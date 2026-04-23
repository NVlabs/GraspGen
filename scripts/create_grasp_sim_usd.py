# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Create a USD scene for Omniverse/Isaac Sim with multiple envs: each env has
the object and one gripper at a predicted grasp pose. Use with
scripts/run_grasp_sim_omniverse.py in Isaac Sim so that on Play the grippers close
and grasp the object.
"""

import argparse
import math
import os
import sys

# Allow running from repo root: python scripts/create_grasp_sim_usd.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from save_grasps_to_usd import _load_yaml_grasps

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

WORLD_ROOT = "/World"
ENV_SPACING_DEFAULT = 0.6
# Robotiq 2F-85 USD root prim (note: file may say 2F_86 internally)
GRIPPER_DEFAULT_PRIM = "/Robotiq_2F_86"
OBJECT_DEFAULT_PRIM = "/world"


def _numpy_to_gf_matrix4d(m: np.ndarray) -> Gf.Matrix4d:
    return Gf.Matrix4d(*m.T.flatten().tolist())


def _env_offset_index(i: int, num_envs: int, env_spacing: float):
    """Return (x, y, 0) world offset for env i in a 2D grid (row-major)."""
    cols = math.ceil(math.sqrt(num_envs))
    row = i // cols
    col = i % cols
    return (col * env_spacing, row * env_spacing, 0.0)


def create_grasp_sim_usd(
    object_usd_path: str,
    grasps: np.ndarray,
    confidences: np.ndarray,
    output_path: str,
    gripper_usd_path: str,
    object_prim_path: str = OBJECT_DEFAULT_PRIM,
    gripper_prim_path: str = GRIPPER_DEFAULT_PRIM,
    num_envs: int = 10,
    env_spacing: float = ENV_SPACING_DEFAULT,
) -> str:
    """Build a USD with num_envs environments: each has object + gripper at grasp pose.

    Args:
        object_usd_path: Path to object USD (e.g. box.usd).
        grasps: (N, 4, 4) grasp transforms in object frame.
        confidences: (N,) confidence scores (unused but kept for API).
        output_path: Output USD path (e.g. box_with_grasps_sim.usd).
        gripper_usd_path: Path to gripper USD (e.g. robotiq_2f_85.usd).
        object_prim_path: Prim path in object USD to reference (default /world).
        gripper_prim_path: Prim path in gripper USD to reference (default /Robotiq_2F_86).
        num_envs: Number of envs to create (default 10).
        env_spacing: Spacing between env origins in meters.

    Returns:
        output_path.
    """
    assert grasps.ndim == 3 and grasps.shape[1:] == (4, 4)
    n = min(num_envs, len(grasps))
    grasps = grasps[:n]
    confidences = confidences[:n]

    stage = Usd.Stage.CreateInMemory()
    stage.SetDefaultPrim(stage.DefinePrim(WORLD_ROOT, "Xform"))
    root = stage.GetPrimAtPath(WORLD_ROOT)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    def _ref_path(abs_path: str) -> str:
        p = os.path.abspath(abs_path)
        return os.path.relpath(p, out_dir) if out_dir else p

    object_ref = _ref_path(object_usd_path)
    gripper_ref = _ref_path(gripper_usd_path)

    for i in range(n):
        env_path = f"{WORLD_ROOT}/Env_{i}"
        env_xform = UsdGeom.Xform.Define(stage, env_path)
        offset = _env_offset_index(i, n, env_spacing)
        env_xform.AddTranslateOp().Set(Gf.Vec3d(*offset))

        obj_path = f"{env_path}/object"
        obj_prim = stage.DefinePrim(obj_path, "Xform")
        obj_prim.GetReferences().AddReference(Sdf.Reference(object_ref, object_prim_path))

        # Override collision approximation to convexHull so PhysX doesn't
        # fall back from triangle mesh (which isn't valid for dynamic bodies).
        coll_over = stage.OverridePrim(f"{obj_path}/object/box_obj/box_obj")
        coll_over.CreateAttribute(
            "physics:approximation", Sdf.ValueTypeNames.Token
        ).Set("convexHull")

        # Override the fixed joint's world-side anchor to the env offset so the
        # joint isn't disjointed (original localPos0 is at the world origin).
        joint_over = stage.OverridePrim(f"{obj_path}/object/world_fixed_joint")
        joint_over.CreateAttribute(
            "physics:localPos0", Sdf.ValueTypeNames.Point3f
        ).Set(Gf.Vec3f(float(offset[0]), float(offset[1]), float(offset[2])))

        # Gripper at grasp pose in env-local frame (env Xform already applies offset)
        gripper_path = f"{env_path}/gripper"
        gripper_xform = UsdGeom.Xform.Define(stage, gripper_path)
        gripper_xform.AddTransformOp().Set(_numpy_to_gf_matrix4d(grasps[i]))
        gripper_prim = stage.GetPrimAtPath(gripper_path)
        gripper_prim.GetReferences().AddReference(
            Sdf.Reference(gripper_ref, gripper_prim_path)
        )

    stage.GetRootLayer().Export(output_path)
    print(f"Saved {n} envs to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create box_with_grasps_sim.usd for Omniverse/Isaac Sim"
    )
    parser.add_argument(
        "--object_usd",
        type=str,
        required=True,
        help="Path to object USD (e.g. assets/objects/box.usd)",
    )
    parser.add_argument(
        "--grasps_yaml",
        type=str,
        required=True,
        help="Path to Isaac-format grasp YAML (e.g. from demo_object_mesh.py --output_file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="box_with_grasps_sim.usd",
        help="Output USD path",
    )
    parser.add_argument(
        "--gripper_usd",
        type=str,
        default="",
        help="Path to gripper USD (default: GraspGen assets/bots/robotiq_2f_85.usd)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=10,
        help="Max number of envs (grasps) to write",
    )
    parser.add_argument(
        "--env_spacing",
        type=float,
        default=ENV_SPACING_DEFAULT,
        help="Spacing between env origins in meters",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)  # GraspGen
    default_gripper = os.path.join(repo_root, "assets", "bots", "robotiq_2f_85.usd")
    gripper_usd = args.gripper_usd or default_gripper

    if not os.path.exists(args.object_usd):
        print(f"Error: Object USD not found: {args.object_usd}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.grasps_yaml):
        print(f"Error: Grasps YAML not found: {args.grasps_yaml}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(gripper_usd):
        print(f"Error: Gripper USD not found: {gripper_usd}", file=sys.stderr)
        sys.exit(1)

    grasps, confidences = _load_yaml_grasps(args.grasps_yaml)
    create_grasp_sim_usd(
        args.object_usd,
        grasps,
        confidences,
        args.output,
        gripper_usd,
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
    )


if __name__ == "__main__":
    main()
