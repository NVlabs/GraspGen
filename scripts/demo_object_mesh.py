# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
from pathlib import Path

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    get_normals_from_mesh,
    make_frame,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal
from grasp_gen.dataset.dataset_utils import sample_points
from grasp_gen.dataset.eval_utils import save_to_isaac_grasp_format


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasps on a single object mesh after GraspGen inference"
    )
    parser.add_argument(
        "--mesh_file",
        type=str,
        required=True,
        help="Path to the mesh file (obj, stl, ply, usd, usda, usdc, or usdz)",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default="",
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=-1,
        help="Threshold for valid grasps. If -1.0, then the top 100 grasps will be ranked and returned",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=400,
        help="Number of grasps to generate",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Whether to return only the top k grasps",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=-1,
        help="Number of top grasps to return when return_topk is True",
    )
    parser.add_argument(
        "--mesh_scale",
        type=float,
        default=1.0,
        help="Scale factor to apply to the mesh",
    )
    parser.add_argument(
        "--num_sample_points",
        type=int,
        default=2000,
        help="Number of points to sample from the mesh surface",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Path to save the output grasps. If empty, will save to outputs/YYYY-MM-DD/latest/output_grasps.yml",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable meshcat visualization",
    )

    return parser.parse_args()


def load_mesh_data(mesh_file, scale, num_sample_points):
    """Load mesh data and sample points from surface."""
    if mesh_file.endswith("ply"):
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(mesh_file)
        xyz = np.array(pcd.points).astype(np.float32)
        pt_idx = sample_points(xyz, num_sample_points)
        xyz = xyz[pt_idx]
        obj = None
    elif mesh_file.endswith((".usd", ".usda", ".usdc", ".usdz")):
        import scene_synthesizer as synth

        asset = synth.Asset(mesh_file)
        obj = asset.mesh()
        obj.apply_scale(scale)
        xyz, _ = trimesh.sample.sample_surface(obj, num_sample_points)
        xyz = np.array(xyz)
    else:
        obj = trimesh.load(mesh_file)
        obj.apply_scale(scale)
        xyz, _ = trimesh.sample.sample_surface(obj, num_sample_points)
        xyz = np.array(xyz)

    # Center point cloud
    T_subtract_pc_mean = tra.translation_matrix(-xyz.mean(axis=0))
    xyz = tra.transform_points(xyz, T_subtract_pc_mean)
    if obj is not None:
        obj.apply_transform(T_subtract_pc_mean)

    # Create dummy RGB values (white)
    rgb = np.ones((len(xyz), 3)) * 255

    return xyz, rgb, obj, T_subtract_pc_mean


if __name__ == "__main__":
    args = parse_args()

    if args.gripper_config == "":
        raise ValueError("Gripper config is required")

    if not os.path.exists(args.gripper_config):
        raise ValueError(f"Gripper config {args.gripper_config} does not exist")

    valid_extensions = (".stl", ".obj", ".ply", ".usd", ".usda", ".usdc", ".usdz")
    if not args.mesh_file.endswith(valid_extensions):
        raise ValueError(f"Mesh file must be one of {valid_extensions}")

    # Handle return_topk logic
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    # Create visualizer unless visualization is disabled
    vis = None if args.no_visualization else create_visualizer()

    # Load grasp configuration and initialize sampler
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    grasp_sampler = GraspGenSampler(grasp_cfg)

    # Load mesh data
    print(f"Processing mesh file: {args.mesh_file}")
    pc, pc_color, obj_mesh, T_subtract_pc_mean = load_mesh_data(
        args.mesh_file, args.mesh_scale, args.num_sample_points
    )

    # Visualize original mesh
    if not args.no_visualization and obj_mesh is not None:
        visualize_mesh(vis, "object_mesh", obj_mesh, color=[169, 169, 169])
        visualize_pointcloud(vis, "pc", pc, pc_color, size=0.0025)

    # Run inference on point cloud
    grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
        pc,
        grasp_sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
        remove_outliers=False,
    )

    if len(grasps_inferred) > 0:
        grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
        grasps_inferred = grasps_inferred.cpu().numpy()
        scores_inferred = get_color_from_score(grasp_conf_inferred, use_255_scale=True)
        print(
            f"Inferred {len(grasps_inferred)} grasps, with scores ranging from {grasp_conf_inferred.min():.3f} - {grasp_conf_inferred.max():.3f}"
        )

        # Visualize inferred grasps
        if not args.no_visualization:
            for j, grasp in enumerate(grasps_inferred):
                visualize_grasp(
                    vis,
                    f"grasps_objectpc_filtered/{j:03d}/grasp",
                    grasp,
                    color=scores_inferred[j],
                    gripper_name=gripper_name,
                    linewidth=0.6,
                )

        # Convert grasps back to original mesh frame
        grasps_inferred = np.array(
            [tra.inverse_matrix(T_subtract_pc_mean) @ g for g in grasps_inferred]
        )

        # Save grasps to file only if output_file is not empty
        if args.output_file != "":
            print(f"Saving predicted grasps to {args.output_file}")
            save_to_isaac_grasp_format(
                grasps_inferred, grasp_conf_inferred, args.output_file
            )
        else:
            print("No output file specified, skipping grasp saving")

    else:
        print("No grasps found from inference!")
