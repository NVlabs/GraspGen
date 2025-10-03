# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import glob
import json
import os

import numpy as np
import omegaconf
import torch
import trimesh.transformations as tra

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    get_normals_from_mesh,
    make_frame,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal_with_color


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasps on a scene point cloud after GraspGen inference, for entire scene"
    )
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        default="",
        help="Directory containing JSON files with point cloud data",
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
        default=0.80,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sample_data_dir == "":
        raise ValueError("sample_data_dir is required")
    if args.gripper_config == "":
        raise ValueError("gripper_config is required")

    if not os.path.exists(args.sample_data_dir):
        raise FileNotFoundError(
            f"sample_data_dir {args.sample_data_dir} does not exist"
        )

    # Handle return_topk logic
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    json_files = glob.glob(os.path.join(args.sample_data_dir, "*.json"))

    # Load gripper config and get gripper name
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name

    # Initialize GraspGenSampler once
    grasp_sampler = GraspGenSampler(grasp_cfg)

    vis = create_visualizer()

    map_method2color = {
        "GraspGen": [0, 185, 0],
    }

    for json_file in json_files:
        print(json_file)
        vis.delete()

        data = json.load(open(json_file, "rb"))

        obj_pc = np.array(data["object_info"]["pc"])
        obj_pc_color = np.array(data["object_info"]["pc_color"])
        grasps = np.array(data["grasp_info"]["grasp_poses"])
        grasp_conf = np.array(data["grasp_info"]["grasp_conf"])

        full_pc_key = "pc_color" if "pc_color" in data["scene_info"] else "full_pc"
        xyz_scene = np.array(data["scene_info"][full_pc_key])[0]
        xyz_scene_color = np.array(data["scene_info"]["img_color"]).reshape(1, -1, 3)[
            0, :, :
        ]

        VIZ_BOUNDS = [[-1.5, -1.25, -0.15], [1.5, 1.25, 2.0]]
        mask_within_bounds = np.all((xyz_scene > VIZ_BOUNDS[0]), 1)
        mask_within_bounds = np.logical_and(
            mask_within_bounds, np.all((xyz_scene < VIZ_BOUNDS[1]), 1)
        )
        # mask_within_bounds = np.ones(xyz_scene.shape[0]).astype(np.bool_)

        xyz_scene = xyz_scene[mask_within_bounds]
        xyz_scene_color = xyz_scene_color[mask_within_bounds]

        visualize_pointcloud(vis, "pc_scene", xyz_scene, xyz_scene_color, size=0.0025)

        obj_pc, pc_removed, obj_pc_color, obj_pc_color_removed = point_cloud_outlier_removal_with_color(
            torch.from_numpy(obj_pc), torch.from_numpy(obj_pc_color)
        )
        obj_pc = obj_pc.cpu().numpy()
        pc_removed = pc_removed.cpu().numpy()
        obj_pc_color = obj_pc_color.cpu().numpy()
        obj_pc_color_removed = obj_pc_color_removed.cpu().numpy()

        visualize_pointcloud(vis, "pc_obj", obj_pc, obj_pc_color, size=0.005)

        method = "GraspGen"
        grasps, grasp_conf = GraspGenSampler.run_inference(
            obj_pc,
            grasp_sampler,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk_num_grasps,
        )

        if len(grasps) > 0:
            grasp_conf = grasp_conf.cpu().numpy()
            grasps = grasps.cpu().numpy()
            grasps[:, 3, 3] = 1
            scores = get_color_from_score(grasp_conf, use_255_scale=True)
            print(
                f"[{method}] Scores with min {grasp_conf.min():.3f} and max {grasp_conf.max():.3f}"
            )

            for j, grasp in enumerate(grasps):
                color = scores[j]
                color = map_method2color[method]

                visualize_grasp(
                    vis,
                    f"{method}/{j:03d}/grasp",
                    grasp,
                    color=color,
                    gripper_name=gripper_name,
                    linewidth=1.5,
                )

            input("Press Enter to continue to next scene...")
        else:
            print(f"[{method}] No grasps found! Skipping to next scene...")
