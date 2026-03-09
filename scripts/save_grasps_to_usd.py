# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Save predicted grasps as Xform poses into a USD file.

Each grasp is stored as a UsdGeom.Xform under /world/grasps/grasp_NNN with its
full 4x4 transform and a confidence custom attribute.  The object mesh (if
present in the USD) is left untouched.

Optionally, when gripper_name is provided, adds /world/grasps_visualization
with the same poses and gripper wireframe geometry (BasisCurves) at each pose,
matching the format used in grasp_gen.utils.viser_utils.visualize_grasp.

This can also be used standalone to convert an Isaac-format YAML grasp file into
USD grasp Xforms appended to an existing USD scene.
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import trimesh.transformations as tra

from pxr import Gf, Sdf, Usd, UsdGeom, Vt


GRASPS_ROOT_PATH = "/world/grasps"
GRASPS_VIS_ROOT_PATH = "/world/grasps_visualization"

# Default radius/width for grasp wireframe curves in meters (thin lines in Omniverse).
GRASP_WIREFRAME_WIDTH_M = 0.001


def save_grasps_to_usd(
    usd_path: str,
    grasps: np.ndarray,
    confidences: np.ndarray,
    output_path: Optional[str] = None,
    grasps_root: str = GRASPS_ROOT_PATH,
    gripper_name: Optional[str] = None,
    add_visualization: bool = True,
    wireframe_width_m: float = GRASP_WIREFRAME_WIDTH_M,
):
    """Write grasp poses as Xforms into a USD file.

    Args:
        usd_path: Path to an existing USD file to augment (or the output path
            if creating from scratch).
        grasps: (N, 4, 4) array of homogeneous grasp transforms.
        confidences: (N,) array of per-grasp confidence scores.
        output_path: If provided, save to this path instead of overwriting
            usd_path.  If None, overwrites usd_path in place.
        grasps_root: Prim path under which to create grasp Xforms.
        gripper_name: If set, also add /world/grasps_visualization with gripper
            wireframe at each pose (same format as viser_utils.visualize_grasp).
        add_visualization: If False, skip adding grasps_visualization even when
            gripper_name is set.
        wireframe_width_m: Width/radius of the grasp wireframe curves in meters
            (default 0.001 = 1mm). Override if lines look too thick in the viewer.

    Returns:
        The path to the written USD file.
    """
    assert len(grasps) == len(confidences)
    assert grasps.ndim == 3 and grasps.shape[1:] == (4, 4)

    if output_path is None:
        output_path = usd_path

    if os.path.abspath(usd_path) != os.path.abspath(output_path):
        import shutil
        shutil.copy2(usd_path, output_path)
        stage = Usd.Stage.Open(output_path)
    else:
        stage = Usd.Stage.Open(usd_path)

    _remove_existing_grasps(stage, grasps_root)

    UsdGeom.Xform.Define(stage, grasps_root)

    for i, (grasp, conf) in enumerate(zip(grasps, confidences)):
        grasp_path = f"{grasps_root}/grasp_{i:03d}"
        xform = UsdGeom.Xform.Define(stage, grasp_path)

        xform_op = xform.AddTransformOp()
        mat = _numpy_to_gf_matrix4d(grasp)
        xform_op.Set(mat)

        prim = xform.GetPrim()
        conf_attr = prim.CreateAttribute(
            "graspgen:confidence", Sdf.ValueTypeNames.Float
        )
        conf_attr.Set(float(conf))

    if gripper_name and add_visualization:
        add_grasps_visualization_to_stage(
            stage, grasps, confidences, gripper_name, vis_root=GRASPS_VIS_ROOT_PATH,
            wireframe_width_m=wireframe_width_m,
        )
        print(f"Saved grasps_visualization to {output_path} under {GRASPS_VIS_ROOT_PATH}")

    stage.GetRootLayer().Save()
    print(f"Saved {len(grasps)} grasps to {output_path} under {grasps_root}")
    return output_path


def load_grasps_from_usd(
    usd_path: str,
    grasps_root: str = GRASPS_ROOT_PATH,
):
    """Read grasp Xform poses back from a USD file.

    Args:
        usd_path: Path to the USD file.
        grasps_root: Prim path under which grasp Xforms were stored.

    Returns:
        grasps: (N, 4, 4) numpy array of grasp transforms.
        confidences: (N,) numpy array of confidence values.
    """
    stage = Usd.Stage.Open(usd_path)
    root_prim = stage.GetPrimAtPath(grasps_root)
    if not root_prim:
        return np.empty((0, 4, 4)), np.empty((0,))

    grasps = []
    confidences = []

    for child in root_prim.GetChildren():
        xf = UsdGeom.Xformable(child)
        mat4 = xf.GetLocalTransformation()
        grasps.append(_gf_matrix4d_to_numpy(mat4))

        conf_attr = child.GetAttribute("graspgen:confidence")
        conf = conf_attr.Get() if conf_attr else 0.0
        confidences.append(float(conf))

    return np.array(grasps), np.array(confidences)


def _numpy_to_gf_matrix4d(m: np.ndarray) -> Gf.Matrix4d:
    """Convert a 4x4 numpy array to a pxr Gf.Matrix4d (row-major)."""
    return Gf.Matrix4d(*m.T.flatten().tolist())


def _gf_matrix4d_to_numpy(m: Gf.Matrix4d) -> np.ndarray:
    """Convert a pxr Gf.Matrix4d back to a 4x4 numpy array."""
    return np.array(m, dtype=np.float64).T


def _remove_existing_grasps(stage: Usd.Stage, grasps_root: str):
    """Remove any pre-existing grasps prim tree so we can write fresh."""
    prim = stage.GetPrimAtPath(grasps_root)
    if prim:
        stage.RemovePrim(grasps_root)


def _get_gripper_wireframe_segments(gripper_name: str) -> np.ndarray:
    """Build line segments for gripper wireframe in local frame (meters).

    Same logic as grasp_gen.utils.viser_utils.visualize_grasp: consecutive
    points in each polyline from load_control_points_for_visualization become
    segments. Returns (num_segments, 2, 3) in gripper local frame.
    """
    from grasp_gen.robot import load_control_points_for_visualization

    segments_list: List[np.ndarray] = []
    for ctrl_pts in load_control_points_for_visualization(gripper_name):
        pts = np.array(ctrl_pts, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        if pts.shape[0] < 2:
            continue
        for j in range(pts.shape[0] - 1):
            segments_list.append(pts[j : j + 2])
    if not segments_list:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.stack(segments_list, axis=0)


def _add_basis_curves_prim(
    stage: Usd.Stage,
    parent_prim_path: str,
    name: str,
    segments: np.ndarray,
    color_rgb: Tuple[float, float, float],
    width_m: float = GRASP_WIREFRAME_WIDTH_M,
) -> Usd.Prim:
    """Add a UsdGeom.BasisCurves prim for line segments under parent_prim_path.

    segments: (N, 2, 3) in meters. color_rgb: (r, g, b) in [0, 1].
    width_m: curve width/radius in meters (default thin for wireframe).
    """
    if segments.shape[0] == 0:
        curves_path = f"{parent_prim_path}/{name}"
        curves = UsdGeom.BasisCurves.Define(stage, curves_path)
        curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([]))
        curves.CreatePointsAttr().Set(Vt.Vec3fArray([]))
        curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        prim = curves.GetPrim()
    else:
        curve_vertex_counts = [2] * segments.shape[0]
        points_flat = segments.reshape(-1, 3)
        points_list = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points_flat]
        curves_path = f"{parent_prim_path}/{name}"
        curves = UsdGeom.BasisCurves.Define(stage, curves_path)
        curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray(curve_vertex_counts))
        curves.CreatePointsAttr().Set(Vt.Vec3fArray(points_list))
        curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        # Thin lines: constant width for entire prim (in meters).
        curves.CreateWidthsAttr().Set(Vt.FloatArray([width_m]))
        prim = curves.GetPrim()
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        display_color = primvars_api.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3f
        )
        display_color.Set(Vt.Vec3fArray([Gf.Vec3f(*color_rgb)]))
    return prim


def add_grasps_visualization_to_stage(
    stage: Usd.Stage,
    grasps: np.ndarray,
    confidences: np.ndarray,
    gripper_name: str,
    vis_root: str = GRASPS_VIS_ROOT_PATH,
    wireframe_width_m: float = GRASP_WIREFRAME_WIDTH_M,
) -> None:
    """Add /world/grasps_visualization with gripper wireframe at each grasp pose.

    Creates vis_root/grasp_NNN (Xform with grasp transform) and under each
    a BasisCurves child with the gripper wireframe. displayColor is set from
    confidence (red = low, green = high).
    """
    _remove_existing_grasps(stage, vis_root)
    segments = _get_gripper_wireframe_segments(gripper_name)

    UsdGeom.Xform.Define(stage, vis_root)

    for i, (grasp, conf) in enumerate(zip(grasps, confidences)):
        grasp_vis_path = f"{vis_root}/grasp_{i:03d}"
        xform = UsdGeom.Xform.Define(stage, grasp_vis_path)
        xform_op = xform.AddTransformOp()
        xform_op.Set(_numpy_to_gf_matrix4d(grasp))

        # Color from confidence: (1-c, c, 0) -> red (low) to green (high)
        c = float(np.clip(conf, 0.0, 1.0))
        color_rgb = (1.0 - c, c, 0.0)
        _add_basis_curves_prim(
            stage, grasp_vis_path, "wireframe", segments, color_rgb,
            width_m=wireframe_width_m,
        )


def _load_yaml_grasps(yaml_path: str):
    """Load grasps from an Isaac-format YAML file."""
    import yaml

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    grasp_entries = data.get("grasps", {})
    grasps, confidences = [], []
    for _, g in sorted(grasp_entries.items()):
        pos = np.array(g["position"])
        qw = g["orientation"]["w"]
        qxyz = g["orientation"]["xyz"]
        T = tra.translation_matrix(pos) @ tra.quaternion_matrix([qw] + qxyz)
        grasps.append(T)
        confidences.append(g.get("confidence", 0.0))

    return np.array(grasps), np.array(confidences)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save grasps as Xform poses into a USD file"
    )
    parser.add_argument(
        "--usd_file", type=str, required=True, help="Path to the USD file to augment"
    )
    parser.add_argument(
        "--grasps_yaml",
        type=str,
        required=True,
        help="Path to Isaac-format YAML grasp file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output USD path (default: overwrite usd_file)",
    )
    parser.add_argument(
        "--gripper_name",
        type=str,
        default=None,
        help="Gripper name for grasps_visualization wireframe (e.g. robotiq_2f_140, franka_panda). If set, adds /world/grasps_visualization.",
    )
    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="Do not add grasps_visualization even when --gripper_name is set",
    )
    parser.add_argument(
        "--wireframe_width",
        type=float,
        default=GRASP_WIREFRAME_WIDTH_M,
        help=f"Width/radius of grasp wireframe curves in meters (default: {GRASP_WIREFRAME_WIDTH_M}). Use e.g. 0.0005 for thinner lines.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.usd_file):
        print(f"Error: USD file not found: {args.usd_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.grasps_yaml):
        print(f"Error: Grasps file not found: {args.grasps_yaml}", file=sys.stderr)
        sys.exit(1)

    grasps, confidences = _load_yaml_grasps(args.grasps_yaml)
    output = args.output if args.output else None
    save_grasps_to_usd(
        args.usd_file,
        grasps,
        confidences,
        output_path=output,
        gripper_name=args.gripper_name,
        add_visualization=not args.no_visualization,
        wireframe_width_m=args.wireframe_width,
    )
