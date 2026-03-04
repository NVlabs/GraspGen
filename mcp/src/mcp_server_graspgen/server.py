# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""MCP server that bridges LLM tool-calling to the GraspGen ZMQ inference server.

Architecture:
    LLM (Cursor / Claude) <-- MCP (stdio) --> This server <-- ZMQ (tcp) --> GraspGen GPU server

The GraspGen ZMQ server must be running separately (see client-server/README.md).
This MCP server is lightweight — no CUDA or model weights required.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Annotated, Optional

import numpy as np
import trimesh
import viser
import viser.transforms as vtf
import zmq
import msgpack
import msgpack_numpy

from mcp.shared.exceptions import McpError
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    ErrorData,
    TextContent,
    Tool,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from pydantic import BaseModel, Field

msgpack_numpy.patch()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ZMQ client helpers (inlined to keep this package self-contained)
# ---------------------------------------------------------------------------

_zmq_ctx: zmq.Context | None = None
_zmq_socket: zmq.Socket | None = None
_server_addr: str = ""


def _get_server_addr() -> str:
    global _server_addr
    if not _server_addr:
        host = os.environ.get("GRASPGEN_HOST", "localhost")
        port = os.environ.get("GRASPGEN_PORT", "5556")
        _server_addr = f"tcp://{host}:{port}"
    return _server_addr


def _get_socket(timeout_ms: int = 30_000) -> zmq.Socket:
    global _zmq_ctx, _zmq_socket
    if _zmq_socket is None:
        _zmq_ctx = zmq.Context()
        _zmq_socket = _zmq_ctx.socket(zmq.REQ)
        _zmq_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        _zmq_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        _zmq_socket.setsockopt(zmq.LINGER, 0)
        _zmq_socket.connect(_get_server_addr())
    return _zmq_socket


def _reset_socket() -> None:
    global _zmq_socket, _zmq_ctx
    if _zmq_socket is not None:
        _zmq_socket.close()
        _zmq_socket = None
    if _zmq_ctx is not None:
        _zmq_ctx.term()
        _zmq_ctx = None


def _zmq_request(payload: dict, timeout_ms: int = 60_000) -> dict:
    """Send a msgpack request to the GraspGen ZMQ server and return the response."""
    try:
        sock = _get_socket(timeout_ms)
        sock.send(msgpack.packb(payload, use_bin_type=True))
        raw = sock.recv()
        response = msgpack.unpackb(raw, raw=False)
        if "error" in response:
            raise McpError(
                ErrorData(code=INTERNAL_ERROR, message=f"GraspGen server error: {response['error']}")
            )
        return response
    except zmq.error.Again:
        _reset_socket()
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"GraspGen server at {_get_server_addr()} timed out. Is it running?",
            )
        )
    except zmq.error.ZMQError as e:
        _reset_socket()
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"ZMQ communication error with GraspGen server: {e}",
            )
        )


# ---------------------------------------------------------------------------
# Viser visualization state
# ---------------------------------------------------------------------------

_viser_server: Optional[viser.ViserServer] = None
_viser_port: Optional[int] = None
_viser_lock = threading.Lock()

GRIPPER_DIMS: dict[str, tuple[float, float]] = {
    "robotiq_2f_140": (0.136, 0.195),
    "franka_panda": (0.08, 0.1034),
    "single_suction_cup_30mm": (0.03, 0.10),
}


def _gripper_polyline(width: float, depth: float) -> np.ndarray:
    """Return a 7-point polyline representing a parallel-jaw gripper.

    The shape traces: right_tip -> right_base -> mid -> origin -> mid -> left_base -> left_tip
    with approach direction along +Z.
    """
    hw, hd = width / 2, depth / 2
    return np.array(
        [
            [hw, 0, depth],
            [hw, 0, hd],
            [0, 0, hd],
            [0, 0, 0],
            [0, 0, hd],
            [-hw, 0, hd],
            [-hw, 0, depth],
        ],
        dtype=np.float32,
    )


def _get_or_create_viser(port: int) -> viser.ViserServer:
    global _viser_server, _viser_port
    with _viser_lock:
        if _viser_server is not None and _viser_port == port:
            _viser_server.scene.reset()
            return _viser_server
        _viser_server = viser.ViserServer(port=port)
        _viser_port = port
        return _viser_server


def _visualize_grasps_in_viser(
    point_cloud: np.ndarray,
    grasps: np.ndarray,
    confidences: np.ndarray,
    gripper_name: str,
    port: int,
) -> str:
    """Populate a viser scene with the point cloud and color-coded grasp poses."""
    vis = _get_or_create_viser(port)

    heights = point_cloud[:, 2]
    h_min, h_max = heights.min(), heights.max()
    h_range = h_max - h_min if h_max > h_min else 1.0
    t = (heights - h_min) / h_range
    pc_colors = np.zeros((len(point_cloud), 3), dtype=np.uint8)
    pc_colors[:, 0] = (80 * (1 - t)).astype(np.uint8)
    pc_colors[:, 1] = (180 * t + 60 * (1 - t)).astype(np.uint8)
    pc_colors[:, 2] = (220 * t + 120 * (1 - t)).astype(np.uint8)

    vis.scene.add_point_cloud(
        "point_cloud",
        points=point_cloud.astype(np.float32),
        colors=pc_colors,
        point_size=0.003,
    )

    w, d = GRIPPER_DIMS.get(gripper_name, (0.10, 0.15))
    ctrl_pts = _gripper_polyline(w, d)
    n_pts = len(ctrl_pts)
    segments = np.zeros((n_pts - 1, 2, 3), dtype=np.float32)
    for j in range(n_pts - 1):
        segments[j, 0] = ctrl_pts[j]
        segments[j, 1] = ctrl_pts[j + 1]

    for i, (grasp, conf) in enumerate(zip(grasps, confidences)):
        grasp = grasp.copy()
        grasp[3, 3] = 1.0
        color = (int((1 - conf) * 255), int(conf * 255), 0)

        so3 = vtf.SO3.from_matrix(grasp[:3, :3].astype(np.float64))
        wxyz = so3.wxyz
        position = grasp[:3, 3].astype(np.float64)

        vis.scene.add_line_segments(
            f"grasps/{i:03d}",
            points=segments,
            colors=color,
            line_width=0.6,
            wxyz=wxyz,
            position=position,
        )

    url = f"http://localhost:{port}"
    return url


# ---------------------------------------------------------------------------
# Tool input schemas
# ---------------------------------------------------------------------------


class GenerateGraspsFromMesh(BaseModel):
    """Generate 6-DOF grasp poses for an object given its mesh file."""

    mesh_file: Annotated[
        str,
        Field(description="Absolute path to the mesh file (.obj, .stl, .ply, .glb)"),
    ]
    mesh_scale: Annotated[
        float,
        Field(default=1.0, description="Scale factor to apply to the mesh before sampling"),
    ]
    num_sample_points: Annotated[
        int,
        Field(default=2000, description="Number of points to sample from the mesh surface", gt=0),
    ]
    num_grasps: Annotated[
        int,
        Field(default=200, description="Number of grasps for the diffusion model to sample", gt=0),
    ]
    topk_num_grasps: Annotated[
        int,
        Field(
            default=100,
            description="Return only top-k grasps ranked by confidence. -1 to return all.",
        ),
    ]


class GenerateGraspsFromPointCloud(BaseModel):
    """Generate 6-DOF grasp poses for an object given a point cloud file."""

    point_cloud_file: Annotated[
        str,
        Field(
            description=(
                "Absolute path to a point cloud file. "
                "Supported formats: .npy (N,3 float32 array), .npz (must contain key 'point_cloud' or 'xyz'), "
                ".ply / .pcd (loaded via trimesh)."
            )
        ),
    ]
    num_grasps: Annotated[
        int,
        Field(default=200, description="Number of grasps for the diffusion model to sample", gt=0),
    ]
    topk_num_grasps: Annotated[
        int,
        Field(
            default=100,
            description="Return only top-k grasps ranked by confidence. -1 to return all.",
        ),
    ]


class VisualizeGrasps(BaseModel):
    """Generate grasps and visualize them in a 3D viser web viewer."""

    mesh_file: Annotated[
        Optional[str],
        Field(default=None, description="Absolute path to a mesh file (.obj, .stl, .ply, .glb)"),
    ]
    point_cloud_file: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Absolute path to a point cloud file (.npy, .npz, .ply, .pcd)",
        ),
    ]
    mesh_scale: Annotated[
        float,
        Field(default=1.0, description="Scale factor for the mesh (only used with mesh_file)"),
    ]
    num_sample_points: Annotated[
        int,
        Field(default=2000, description="Points to sample from the mesh surface", gt=0),
    ]
    num_grasps: Annotated[
        int,
        Field(default=200, description="Number of grasps for the diffusion model to sample", gt=0),
    ]
    topk_num_grasps: Annotated[
        int,
        Field(default=100, description="Return only top-k grasps ranked by confidence. -1 for all."),
    ]
    viser_port: Annotated[
        int,
        Field(default=8080, description="Port for the viser 3D visualization server"),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_pcd_ascii(path: str) -> np.ndarray:
    """Minimal ASCII PCD reader (FIELDS x y z)."""
    points = []
    in_data = False
    with open(path, "r") as f:
        for line in f:
            if in_data:
                vals = line.strip().split()
                if len(vals) >= 3:
                    points.append([float(vals[0]), float(vals[1]), float(vals[2])])
            elif line.strip().startswith("DATA"):
                in_data = True
    if not points:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=f"No points found in PCD file: {path}"))
    return np.array(points, dtype=np.float32)


def _load_point_cloud_from_file(path: str) -> np.ndarray:
    """Load a point cloud from various file formats, returning (N, 3) float32."""
    if not os.path.isfile(path):
        raise McpError(ErrorData(code=INVALID_PARAMS, message=f"File not found: {path}"))

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        pc = np.load(path).astype(np.float32)
    elif ext == ".npz":
        data = np.load(path)
        for key in ("point_cloud", "xyz", "points", "pc"):
            if key in data:
                pc = data[key].astype(np.float32)
                break
        else:
            keys = list(data.keys())
            if len(keys) == 1:
                pc = data[keys[0]].astype(np.float32)
            else:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"NPZ file has keys {keys}; expected one of: point_cloud, xyz, points, pc",
                    )
                )
    elif ext == ".pcd":
        pc = _read_pcd_ascii(path)
    elif ext in (".ply", ".obj", ".stl", ".glb", ".gltf"):
        mesh_or_pc = trimesh.load(path)
        if hasattr(mesh_or_pc, "vertices"):
            pc = np.array(mesh_or_pc.vertices, dtype=np.float32)
        else:
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message=f"Could not extract points from {path}")
            )
    else:
        raise McpError(
            ErrorData(code=INVALID_PARAMS, message=f"Unsupported point cloud format: {ext}")
        )

    if pc.ndim != 2 or pc.shape[1] != 3:
        raise McpError(
            ErrorData(code=INVALID_PARAMS, message=f"Point cloud must be (N, 3), got {pc.shape}")
        )
    return pc


def _load_and_sample_mesh(mesh_file: str, scale: float, num_points: int) -> np.ndarray:
    """Load a mesh, scale it, sample surface points, and center them."""
    if not os.path.isfile(mesh_file):
        raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Mesh file not found: {mesh_file}"))

    try:
        mesh = trimesh.load(mesh_file)
    except Exception as e:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Failed to load mesh: {e}"))

    mesh.apply_scale(scale)
    xyz, _ = trimesh.sample.sample_surface(mesh, num_points)
    xyz = np.array(xyz, dtype=np.float32)
    xyz -= xyz.mean(axis=0)
    return xyz


def _format_grasp_results(grasps: np.ndarray, confidences: np.ndarray, timing: dict) -> str:
    """Format grasp inference results into human-readable text for the LLM."""
    n = len(grasps)
    if n == 0:
        return "No grasps were generated. The point cloud may be too sparse or the object too small."

    lines = [
        f"Generated {n} grasp(s).",
        f"Confidence range: {confidences.min():.4f} – {confidences.max():.4f}",
        f"Inference time: {timing.get('infer_ms', 0):.0f} ms",
        "",
        "Top grasps (sorted by confidence, highest first):",
    ]

    show_n = min(n, 5)
    for i in range(show_n):
        pos = grasps[i][:3, 3]
        conf = confidences[i]
        lines.append(
            f"  #{i+1}: confidence={conf:.4f}, "
            f"position=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
        )

    if n > show_n:
        lines.append(f"  ... and {n - show_n} more grasps")

    lines.append("")
    lines.append("Each grasp is a 4x4 homogeneous transformation matrix (SE(3) pose).")
    lines.append(
        "The full grasp data (poses + confidences) can be used downstream "
        "for motion planning with tools like cuRobo."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


async def serve(host: str = "localhost", port: int = 5556) -> None:
    """Run the GraspGen MCP server over stdio."""
    global _server_addr
    _server_addr = f"tcp://{host}:{port}"

    server = Server("mcp-graspgen")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="generate_grasps_from_mesh",
                description=(
                    "Generate 6-DOF robotic grasp poses for an object given its 3D mesh file. "
                    "The mesh is sampled into a point cloud and sent to the GraspGen diffusion model. "
                    "Returns ranked grasp poses (SE(3) matrices) with confidence scores. "
                    "Supported mesh formats: .obj, .stl, .ply, .glb."
                ),
                inputSchema=GenerateGraspsFromMesh.model_json_schema(),
            ),
            Tool(
                name="generate_grasps_from_point_cloud",
                description=(
                    "Generate 6-DOF robotic grasp poses for an object given a point cloud file. "
                    "The point cloud is sent directly to the GraspGen diffusion model. "
                    "Returns ranked grasp poses (SE(3) matrices) with confidence scores. "
                    "Supported formats: .npy, .npz, .ply, .pcd."
                ),
                inputSchema=GenerateGraspsFromPointCloud.model_json_schema(),
            ),
            Tool(
                name="graspgen_health_check",
                description=(
                    "Check if the GraspGen inference server is running and responsive. "
                    "Returns the server status."
                ),
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="graspgen_server_info",
                description=(
                    "Get metadata about the running GraspGen server, including the loaded "
                    "gripper name (e.g. franka_panda, robotiq_2f_140, single_suction_cup_30mm) "
                    "and model configuration."
                ),
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="visualize_grasps",
                description=(
                    "Generate 6-DOF grasp poses and visualize them interactively in a 3D viser "
                    "web viewer. Accepts a mesh file OR a point cloud file. The point cloud and "
                    "gripper poses are rendered in a browser at the returned URL. "
                    "Grasps are color-coded by confidence (green=high, red=low)."
                ),
                inputSchema=VisualizeGrasps.model_json_schema(),
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "generate_grasps_from_mesh":
            return await _handle_generate_grasps_from_mesh(arguments)
        elif name == "generate_grasps_from_point_cloud":
            return await _handle_generate_grasps_from_point_cloud(arguments)
        elif name == "graspgen_health_check":
            return _handle_health_check()
        elif name == "graspgen_server_info":
            return _handle_server_info()
        elif name == "visualize_grasps":
            return await _handle_visualize_grasps(arguments)
        else:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown tool: {name}"))

    async def _handle_generate_grasps_from_mesh(arguments: dict) -> list[TextContent]:
        try:
            args = GenerateGraspsFromMesh(**arguments)
        except ValueError as e:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=str(e)))

        logger.info("Loading mesh: %s (scale=%.2f)", args.mesh_file, args.mesh_scale)
        point_cloud = _load_and_sample_mesh(args.mesh_file, args.mesh_scale, args.num_sample_points)

        response = _zmq_request({
            "action": "infer",
            "point_cloud": point_cloud,
            "num_grasps": args.num_grasps,
            "topk_num_grasps": args.topk_num_grasps,
        })

        grasps = np.asarray(response["grasps"], dtype=np.float32)
        confidences = np.asarray(response["confidences"], dtype=np.float32)
        timing = response.get("timing", {})

        text = (
            f"Mesh: {args.mesh_file} (scale={args.mesh_scale}, "
            f"sampled {args.num_sample_points} points)\n\n"
            + _format_grasp_results(grasps, confidences, timing)
        )

        return [TextContent(type="text", text=text)]

    async def _handle_generate_grasps_from_point_cloud(arguments: dict) -> list[TextContent]:
        try:
            args = GenerateGraspsFromPointCloud(**arguments)
        except ValueError as e:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=str(e)))

        logger.info("Loading point cloud: %s", args.point_cloud_file)
        point_cloud = _load_point_cloud_from_file(args.point_cloud_file)

        response = _zmq_request({
            "action": "infer",
            "point_cloud": point_cloud,
            "num_grasps": args.num_grasps,
            "topk_num_grasps": args.topk_num_grasps,
        })

        grasps = np.asarray(response["grasps"], dtype=np.float32)
        confidences = np.asarray(response["confidences"], dtype=np.float32)
        timing = response.get("timing", {})

        text = (
            f"Point cloud: {args.point_cloud_file} ({len(point_cloud)} points)\n\n"
            + _format_grasp_results(grasps, confidences, timing)
        )

        return [TextContent(type="text", text=text)]

    async def _handle_visualize_grasps(arguments: dict) -> list[TextContent]:
        try:
            args = VisualizeGrasps(**arguments)
        except ValueError as e:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=str(e)))

        if args.mesh_file and args.point_cloud_file:
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message="Provide mesh_file OR point_cloud_file, not both.")
            )
        if not args.mesh_file and not args.point_cloud_file:
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message="Provide either mesh_file or point_cloud_file.")
            )

        if args.mesh_file:
            logger.info("Visualize: loading mesh %s (scale=%.2f)", args.mesh_file, args.mesh_scale)
            point_cloud = _load_and_sample_mesh(args.mesh_file, args.mesh_scale, args.num_sample_points)
            input_desc = f"Mesh: {args.mesh_file} (scale={args.mesh_scale})"
        else:
            logger.info("Visualize: loading point cloud %s", args.point_cloud_file)
            point_cloud = _load_point_cloud_from_file(args.point_cloud_file)
            input_desc = f"Point cloud: {args.point_cloud_file}"

        response = _zmq_request({
            "action": "infer",
            "point_cloud": point_cloud,
            "num_grasps": args.num_grasps,
            "topk_num_grasps": args.topk_num_grasps,
        })

        grasps = np.asarray(response["grasps"], dtype=np.float32)
        confidences = np.asarray(response["confidences"], dtype=np.float32)
        n = len(grasps)

        if n == 0:
            return [TextContent(type="text", text="No grasps generated — nothing to visualize.")]

        metadata = _zmq_request({"action": "metadata"}, timeout_ms=5_000)
        gripper_name = metadata.get("gripper_name", "robotiq_2f_140")

        url = _visualize_grasps_in_viser(
            point_cloud, grasps, confidences, gripper_name, args.viser_port,
        )

        text = (
            f"{input_desc}\n"
            f"Generated {n} grasp(s), confidence range: {confidences.min():.4f} – {confidences.max():.4f}\n\n"
            f"Viser 3D visualization is running at: {url}\n"
            f"Open this URL in a browser to interactively view the point cloud and grasps.\n"
            f"Grasps are color-coded: green = high confidence, red = low confidence."
        )
        return [TextContent(type="text", text=text)]

    def _handle_health_check() -> list[TextContent]:
        try:
            response = _zmq_request({"action": "health"}, timeout_ms=5_000)
            status = response.get("status", "unknown")
            return [TextContent(type="text", text=f"GraspGen server status: {status}")]
        except McpError:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"GraspGen server at {_get_server_addr()} is not reachable. "
                        "Make sure it is running (see GraspGen client-server/README.md)."
                    ),
                )
            ]

    def _handle_server_info() -> list[TextContent]:
        response = _zmq_request({"action": "metadata"}, timeout_ms=5_000)
        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)
