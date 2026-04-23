# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import time
from typing import Tuple, Dict, List

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
from IPython import embed
from tqdm import tqdm

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    make_frame,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import (
    point_cloud_outlier_removal,
    knn_points,
    depth_and_segmentation_to_point_clouds,
    filter_colliding_grasps,
)
from grasp_gen.robot import get_gripper_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run collision-free grasp inference from depth image and segmentation mask"
    )
    parser.add_argument(
        "--depth_image_path",
        type=str,
        required=True,
        help="Path to the depth image (e.g., .npy or image file)",
    )
    parser.add_argument(
        "--segmentation_mask_path",
        type=str,
        required=True,
        help="Path to the instance segmentation mask (e.g., .npy or image file)",
    )
    parser.add_argument(
        "--rgb_image_path",
        type=str,
        default=None,
        help="Path to the RGB image (optional, for colored point cloud visualization)",
    )
    parser.add_argument(
        "--camera_intrinsics",
        type=float,
        nargs=4,
        required=True,
        default=[1480.589599609375, 1480.589599609375, 960.0, 540.0],
        help="Camera intrinsics as [fx, fy, cx, cy]",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        required=True,
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.8,
        help="Threshold for valid grasps. If -1.0, then the top 100 grasps will be ranked and returned",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
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
        "--collision_threshold",
        type=float,
        default=0.02,  # 2mm
        help="Distance threshold for collision detection (in meters)",
    )
    parser.add_argument(
        "--max_scene_points",
        type=int,
        default=8192,
        help="Maximum number of scene points to use for collision checking (for speed optimization)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize the results",
    )

    return parser.parse_args()


def process_point_cloud(pc, grasps, grasp_conf, pc_colors=None):
    """Process point cloud and grasps by centering them."""
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    print(f"Scores with min {grasp_conf.min():.3f} and max {grasp_conf.max():.3f}")

    # Ensure grasps have correct homogeneous coordinate
    grasps[:, 3, 3] = 1

    # Center point cloud and grasps
    T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, T_subtract_pc_mean)
    grasps_centered = np.array(
        [T_subtract_pc_mean @ np.array(g) for g in grasps.tolist()]
    )

    # Add red tint to colors if RGB data is available
    pc_colors_centered = pc_colors
    if pc_colors is not None:
        pc_colors_centered = pc_colors.copy().astype(np.float32)
        # Add red tint: increase red channel by 40% while keeping it in valid range
        pc_colors_centered[:, 0] = np.clip(pc_colors_centered[:, 0] * 1.4, 0, 255)
        pc_colors_centered = pc_colors_centered.astype(np.uint8)

    return pc_centered, grasps_centered, scores, T_subtract_pc_mean, pc_colors_centered


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    if not os.path.exists(args.gripper_config):
        raise ValueError(f"Gripper config {args.gripper_config} does not exist")

    # Handle return_topk logic
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    print(f"Starting collision-free grasp detection at {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    # Load depth image and segmentation mask
    load_start = time.time()
    if args.depth_image_path.endswith(".npy"):
        depth_image = np.load(args.depth_image_path)
    else:
        # Assume it's an image file that can be loaded with standard libraries
        from PIL import Image

        depth_image = np.array(Image.open(args.depth_image_path))
        # Convert to meters if needed (this may need adjustment based on your data format)
        depth_image = (
            depth_image.astype(np.float32) / 1000.0
        )  # Assuming mm to m conversion

    if args.segmentation_mask_path.endswith(".npy"):
        segmentation_mask = np.load(args.segmentation_mask_path)
    else:
        from PIL import Image

        segmentation_mask = np.array(Image.open(args.segmentation_mask_path))

    # Load RGB image if provided
    rgb_image = None
    if args.rgb_image_path:
        if args.rgb_image_path.endswith(".npy"):
            rgb_image = np.load(args.rgb_image_path)
        else:
            from PIL import Image

            rgb_image = np.array(Image.open(args.rgb_image_path))
        print(f"Loaded RGB image with shape: {rgb_image.shape}")

    load_time = time.time() - load_start
    print(f"Data loading took: {load_time:.2f} seconds")
    print(f"Loaded depth image with shape: {depth_image.shape}")
    print(f"Loaded segmentation mask with shape: {segmentation_mask.shape}")
    print(f"Segmentation unique values: {np.unique(segmentation_mask)}")

    # Unpack camera intrinsics
    fx, fy, cx, cy = args.camera_intrinsics

    # Create point clouds from depth and segmentation
    pc_start = time.time()
    try:
        scene_pc, object_pc, scene_colors, object_colors = (
            depth_and_segmentation_to_point_clouds(
                depth_image=depth_image,
                segmentation_mask=segmentation_mask,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                rgb_image=rgb_image,
                target_object_id=1,  # Assuming object has ID 1
                remove_object_from_scene=True,
            )
        )
    except ValueError as e:
        print(f"Error creating point clouds: {e}")
        exit(1)

    pc_creation_time = time.time() - pc_start
    print(f"Point cloud creation took: {pc_creation_time:.2f} seconds")

    # Load grasp configuration and get gripper info
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    gripper_info = get_gripper_info(gripper_name)
    gripper_collision_mesh = gripper_info.collision_mesh

    print(f"Using gripper: {gripper_name}")
    print(f"Gripper collision mesh has {len(gripper_collision_mesh.vertices)} vertices")

    # Initialize visualization if requested
    if args.visualize:
        vis = create_visualizer()

    # Filter object point cloud to remove outliers
    filter_start = time.time()
    object_pc_torch = torch.from_numpy(object_pc)
    pc_filtered, pc_removed = point_cloud_outlier_removal(object_pc_torch)
    pc_filtered = pc_filtered.numpy()
    pc_removed = pc_removed.numpy()

    filter_time = time.time() - filter_start
    print(f"Point cloud filtering took: {filter_time:.2f} seconds")
    print(
        f"Filtered object point cloud: {len(pc_filtered)} outlier points (removed {len(pc_removed)})"
    )

    # Initialize GraspGenSampler and run inference
    inference_start = time.time()
    grasp_sampler = GraspGenSampler(grasp_cfg)
    grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
        pc_filtered,
        grasp_sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
    )

    inference_time = time.time() - inference_start
    print(f"Grasp inference took: {inference_time:.2f} seconds")

    if len(grasps_inferred) == 0:
        print("No grasps found from inference!")
        exit(1)

    # Convert to numpy
    grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
    grasps_inferred = grasps_inferred.cpu().numpy()
    grasps_inferred[:, 3, 3] = 1

    print(
        f"Inferred {len(grasps_inferred)} grasps, with scores ranging from {grasp_conf_inferred.min():.3f} - {grasp_conf_inferred.max():.3f}"
    )

    # Process point clouds and grasps for consistent coordinate frame
    pc_centered, grasps_centered, scores, T_center, object_colors_centered = (
        process_point_cloud(
            pc_filtered, grasps_inferred, grasp_conf_inferred, object_colors
        )
    )
    scene_pc_centered = tra.transform_points(scene_pc, T_center)

    # Add red tint to scene colors if RGB data is available
    scene_colors_centered = scene_colors
    if scene_colors is not None:
        scene_colors_centered = scene_colors.copy().astype(np.float32)
        # Add red tint: increase red channel by 40% while keeping it in valid range
        scene_colors_centered[:, 0] = np.clip(scene_colors_centered[:, 0] * 1.4, 0, 255)
        scene_colors_centered = scene_colors_centered.astype(np.uint8)

    # Downsample scene point cloud for faster collision checking (keep full resolution for visualization)
    if len(scene_pc_centered) > args.max_scene_points:
        indices = np.random.choice(
            len(scene_pc_centered), args.max_scene_points, replace=False
        )
        scene_pc_downsampled = scene_pc_centered[indices]
        print(
            f"Downsampled scene point cloud from {len(scene_pc_centered)} to {len(scene_pc_downsampled)} points for collision checking"
        )
    else:
        scene_pc_downsampled = scene_pc_centered
        print(
            f"Scene point cloud has {len(scene_pc_centered)} points (no downsampling needed)"
        )

    # Filter collision grasps using downsampled scene
    collision_start = time.time()
    collision_free_mask = filter_colliding_grasps(
        scene_pc=scene_pc_downsampled,
        grasp_poses=grasps_centered,
        gripper_collision_mesh=gripper_collision_mesh,
        collision_threshold=args.collision_threshold,
    )
    collision_time = time.time() - collision_start
    print(f"Collision detection took: {collision_time:.2f} seconds")

    # Filter grasps to only collision-free ones
    collision_free_grasps = grasps_centered[collision_free_mask]
    collision_free_scores = grasp_conf_inferred[collision_free_mask]
    collision_free_colors = scores[collision_free_mask]

    print(
        f"Final result: {len(collision_free_grasps)} collision-free grasps out of {len(grasps_inferred)} total grasps"
    )

    # Save results
    results = {
        "all_grasps": grasps_centered,
        "all_scores": grasp_conf_inferred,
        "collision_free_mask": collision_free_mask,
        "collision_free_grasps": collision_free_grasps,
        "collision_free_scores": collision_free_scores,
        "scene_pc": scene_pc_centered,
        "object_pc": pc_centered,
        "camera_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
    }

    output_file = "collision_free_grasps_results.npz"
    np.savez(output_file, **results)
    print(f"Results saved to {output_file}")

    # Print timing summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TIMING SUMMARY:")
    print("=" * 60)
    print(f"Data loading:        {load_time:.2f}s")
    print(f"Point cloud creation: {pc_creation_time:.2f}s")
    print(f"Point cloud filtering: {filter_time:.2f}s")
    print(f"Grasp inference:     {inference_time:.2f}s")
    print(f"Collision detection: {collision_time:.2f}s")
    print(f"Total time:          {total_time:.2f}s")
    print("=" * 60)

    # Visualize results if requested
    if args.visualize:
        viz_start = time.time()

        # Visualize scene point cloud - use RGB colors if available, otherwise use gray
        if scene_colors_centered is not None:
            visualize_pointcloud(
                vis, "scene_pc", scene_pc_centered, scene_colors_centered, size=0.002
            )
        else:
            visualize_pointcloud(
                vis, "scene_pc", scene_pc_centered, [128, 128, 128], size=0.002
            )

        # Visualize object point cloud - use RGB colors if available, otherwise use green
        if object_colors_centered is not None:
            visualize_pointcloud(
                vis, "object_pc", pc_centered, object_colors_centered, size=0.0025
            )
        else:
            visualize_pointcloud(
                vis, "object_pc", pc_centered, [0, 255, 0], size=0.0025
            )

        # Visualize collision-free grasps
        for i, (grasp, score) in enumerate(
            zip(collision_free_grasps, collision_free_colors)
        ):
            visualize_grasp(
                vis,
                f"collision_free_grasps/{i:03d}/grasp",
                grasp,
                color=score,
                gripper_name=gripper_name,
                linewidth=0.8,
            )

        # Visualize colliding grasps in red
        colliding_grasps = grasps_centered[~collision_free_mask]
        for i, grasp in enumerate(
            colliding_grasps[:20]
        ):  # Limit to first 20 for clarity
            visualize_grasp(
                vis,
                f"colliding_grasps/{i:03d}/grasp",
                grasp,
                color=[255, 0, 0],
                gripper_name=gripper_name,
                linewidth=0.4,
            )

        viz_time = time.time() - viz_start
        print(f"Visualization setup took: {viz_time:.2f}s")

        if scene_colors_centered is not None or object_colors_centered is not None:
            print("Visualization ready with RGB colors from the original image.")
        else:
            print(
                "Visualization ready. Green point cloud is the target object, gray is the scene."
            )
        print(
            f"Collision-free grasps are shown in their quality colors, colliding grasps (up to 20) are shown in red."
        )
        input("Press Enter to exit...")
