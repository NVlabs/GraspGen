#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Client that loads a mesh or point cloud, sends it to the GraspGen ZMQ
server, and prints (and optionally visualizes) the returned grasps.

Usage:
    # From a mesh file:
    python client-server/graspgen_client.py \
        --mesh_file /path/to/box.obj --mesh_scale 1.0 \
        --host localhost --port 5556

    # From a PCD file:
    python client-server/graspgen_client.py \
        --pcd_file assets/objects/example_object.pcd \
        --host localhost --port 5556

    # With visualization:
    python client-server/graspgen_client.py \
        --mesh_file /path/to/box.obj --mesh_scale 1.0 \
        --host localhost --port 5556 --visualize
"""

import argparse
import logging
import sys
import time

import numpy as np
import trimesh

from grasp_gen.serving.zmq_client import GraspGenClient

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send a mesh or point cloud to the GraspGen ZMQ server and print grasp results",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--mesh_file", type=str, help="Path to a mesh file (.obj / .stl)"
    )
    input_group.add_argument(
        "--pcd_file", type=str, help="Path to a point cloud file (.pcd / .ply / .xyz / .npy)"
    )

    parser.add_argument(
        "--mesh_scale", type=float, default=1.0, help="Scale factor for the mesh (only used with --mesh_file)"
    )
    parser.add_argument(
        "--num_sample_points",
        type=int,
        default=2000,
        help="Number of points to sample from the mesh surface (only used with --mesh_file)",
    )
    parser.add_argument(
        "--num_grasps", type=int, default=200, help="Number of grasps to request"
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=-1.0,
        help="Confidence threshold (-1.0 = use top-k instead)",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=100,
        help="Return only top-k grasps",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5556, help="Server port")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize point cloud and grasps in viser (http://localhost:8080)",
    )
    parser.add_argument(
        "--viser_port",
        type=int,
        default=8080,
        help="Port for the viser visualization server (default: 8080)",
    )
    return parser.parse_args()


def load_point_cloud_from_mesh(mesh_file: str, scale: float, num_points: int) -> np.ndarray:
    """Load a mesh, scale it, sample surface points, and center them."""
    mesh = trimesh.load(mesh_file)
    mesh.apply_scale(scale)
    xyz, _ = trimesh.sample.sample_surface(mesh, num_points)
    xyz = np.array(xyz, dtype=np.float32)
    xyz -= xyz.mean(axis=0)
    return xyz


def load_point_cloud_from_file(pcd_file: str) -> np.ndarray:
    """Load a point cloud from .pcd, .ply, .xyz, or .npy file and center it."""
    ext = pcd_file.rsplit(".", 1)[-1].lower()

    if ext == "npy":
        xyz = np.load(pcd_file).astype(np.float32)
    elif ext == "xyz":
        xyz = np.loadtxt(pcd_file, dtype=np.float32)
    elif ext == "pcd":
        xyz = _read_pcd_ascii(pcd_file)
    elif ext == "ply":
        cloud = trimesh.load(pcd_file)
        xyz = np.array(cloud.vertices, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported point cloud format: .{ext}")

    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Expected (N, 3+) array, got shape {xyz.shape}")
    xyz = xyz[:, :3]
    xyz -= xyz.mean(axis=0)
    return xyz


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
    return np.array(points, dtype=np.float32)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    if args.mesh_file:
        input_source = args.mesh_file
        logger.info("Loading mesh: %s (scale=%.2f)", args.mesh_file, args.mesh_scale)
        point_cloud = load_point_cloud_from_mesh(
            args.mesh_file, args.mesh_scale, args.num_sample_points
        )
        logger.info("Sampled %d points from mesh surface", len(point_cloud))
    else:
        input_source = args.pcd_file
        logger.info("Loading point cloud: %s", args.pcd_file)
        point_cloud = load_point_cloud_from_file(args.pcd_file)
        logger.info("Loaded %d points from file", len(point_cloud))

    logger.info("Connecting to GraspGen server at %s:%d ...", args.host, args.port)
    with GraspGenClient(host=args.host, port=args.port) as client:
        metadata = client.server_metadata
        logger.info("Server metadata: %s", metadata)

        logger.info("Sending inference request ...")
        t0 = time.monotonic()
        grasps, confidences = client.infer(
            point_cloud,
            num_grasps=args.num_grasps,
            grasp_threshold=args.grasp_threshold,
            topk_num_grasps=args.topk_num_grasps,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        print(f"\n{'='*60}")
        print(f"  GraspGen ZMQ Client Results")
        print(f"{'='*60}")
        print(f"  Input           : {input_source}")
        print(f"  Points sent     : {len(point_cloud)}")
        print(f"  Grasps returned : {len(grasps)}")
        if len(grasps) > 0:
            print(f"  Confidence range: {confidences.min():.4f} - {confidences.max():.4f}")
            print(f"  Best grasp pose :")
            print(f"    {grasps[0]}")
        print(f"  Round-trip time : {elapsed_ms:.1f} ms")
        print(f"{'='*60}\n")

    if args.visualize and len(grasps) > 0:
        visualize_results(
            point_cloud,
            grasps,
            confidences,
            gripper_name=metadata.get("gripper_name", "franka_panda"),
            viser_port=args.viser_port,
        )

    return 0 if len(grasps) > 0 else 1


def visualize_results(
    point_cloud: np.ndarray,
    grasps: np.ndarray,
    confidences: np.ndarray,
    gripper_name: str,
    viser_port: int,
):
    """Visualize the point cloud and grasps using the GraspGen viser utilities."""
    from grasp_gen.utils.viser_utils import (
        create_visualizer,
        get_color_from_score,
        visualize_grasp,
        visualize_pointcloud,
    )

    vis = create_visualizer(port=viser_port)

    pc_color = np.ones((len(point_cloud), 3), dtype=np.uint8) * 200
    visualize_pointcloud(vis, "point_cloud", point_cloud, pc_color, size=0.003)

    scores = get_color_from_score(confidences, use_255_scale=True)
    for i, grasp in enumerate(grasps):
        grasp = grasp.copy()
        grasp[3, 3] = 1.0
        visualize_grasp(
            vis,
            f"grasps/{i:03d}",
            grasp,
            color=scores[i],
            gripper_name=gripper_name,
            linewidth=0.6,
        )

    print(f"\nViser visualization running at http://localhost:{viser_port}")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
