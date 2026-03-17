# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
from typing import Tuple, Dict

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
from tqdm import tqdm

from grasp_gen.utils.logging_config import get_logger
from grasp_gen.dataset.renderer import depth2points

logger = get_logger(__name__)


# @torch.compile
def knn_points(X: torch.Tensor, K: int, norm: int):
    """
    Computes the K-nearest neighbors for each point in the point cloud X.

    Args:
        X: (N, 3) tensor representing the point cloud.
        K: Number of nearest neighbors.

    Returns:
        dists: (N, K) tensor containing squared Euclidean distances to the K nearest neighbors.
        idxs: (N, K) tensor containing indices of the K nearest neighbors.
    """
    N, _ = X.shape
    if N == 0:
        raise ValueError("Point cloud is empty")

    effective_k = min(K, max(1, N - 1))

    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(X, X, p=norm)  # (N, N)

    # Ignore self-distance (optional, but avoids trivial zero distance)
    self_mask = torch.eye(N, device=X.device, dtype=torch.bool)
    dist_matrix.masked_fill_(self_mask, float("inf"))  # Set self-distances to inf

    # Get the indices of the K-nearest neighbors
    dists, idxs = torch.topk(dist_matrix, effective_k, dim=1, largest=False)

    return dists, idxs


def point_cloud_outlier_removal(
    obj_pc: torch.Tensor, threshold: float = 0.014, K: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove outliers from a point cloud. K-nearest neighbors is used to compute the distance to the nearest neighbor for each point.
    If the distance is greater than a threshold, the point is considered an outlier and removed.

    RANSAC can also be used.

    Args:
        obj_pc (torch.Tensor or np.ndarray): (N, 3) tensor or array representing the point cloud.
        threshold (float): Distance threshold for outlier detection. Points with mean distance to K nearest neighbors greater than this threshold are removed.
        K (int): Number of nearest neighbors to consider for outlier detection.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing filtered and removed point clouds.
    """
    # Convert numpy array to torch tensor if needed
    if isinstance(obj_pc, np.ndarray):
        obj_pc = torch.from_numpy(obj_pc)

    obj_pc = obj_pc.float()
    obj_pc = obj_pc.unsqueeze(0)

    num_points = obj_pc.shape[1]
    if num_points <= max(1, K):
        removed_pc = obj_pc[0].new_empty((0, 3))
        logger.info(
            f"Skipping outlier removal for small point cloud ({num_points} points)"
        )
        return obj_pc[0].view(-1, 3), removed_pc

    nn_dists, _ = knn_points(obj_pc[0], K=K, norm=1)

    mask = nn_dists.mean(1) < threshold
    filtered_pc = obj_pc[0, mask]
    removed_pc = obj_pc[0][~mask]

    if filtered_pc.shape[0] == 0:
        logger.info(
            "Outlier removal removed all points; falling back to original point cloud"
        )
        return obj_pc[0].view(-1, 3), obj_pc[0].new_empty((0, 3))

    filtered_pc = filtered_pc.view(-1, 3)
    removed_pc = removed_pc.view(-1, 3)

    logger.info(
        f"Removed {obj_pc.shape[1] - filtered_pc.shape[0]} points from point cloud"
    )
    return filtered_pc, removed_pc


def point_cloud_outlier_removal_with_color(
    obj_pc: torch.Tensor,
    obj_pc_color: torch.Tensor,
    threshold: float = 0.014,
    K: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove outliers from a point cloud with colors. K-nearest neighbors is used to compute the distance to the nearest neighbor for each point.
    If the distance is greater than a threshold, the point is considered an outlier and removed.

    Args:
        obj_pc (torch.Tensor or np.ndarray): (N, 3) tensor or array representing the point cloud.
        obj_pc_color (torch.Tensor or np.ndarray): (N, 3) tensor or array representing the point cloud color.
        threshold (float): Distance threshold for outlier detection. Points with mean distance to K nearest neighbors greater than this threshold are removed.
        K (int): Number of nearest neighbors to consider for outlier detection.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing filtered and removed point clouds and colors.
    """
    # Convert numpy array to torch tensor if needed
    if isinstance(obj_pc, np.ndarray):
        obj_pc = torch.from_numpy(obj_pc)
    if isinstance(obj_pc_color, np.ndarray):
        obj_pc_color = torch.from_numpy(obj_pc_color)

    obj_pc = obj_pc.float()
    obj_pc = obj_pc.unsqueeze(0)

    obj_pc_color = obj_pc_color.float()
    obj_pc_color = obj_pc_color.unsqueeze(0)

    num_points = obj_pc.shape[1]
    if num_points <= max(1, K):
        removed_pc = obj_pc[0].new_empty((0, 3))
        removed_pc_color = obj_pc_color[0].new_empty((0, 3))
        logger.info(
            f"Skipping outlier removal for small point cloud ({num_points} points)"
        )
        return (
            obj_pc[0].view(-1, 3),
            removed_pc,
            obj_pc_color[0].view(-1, 3),
            removed_pc_color,
        )

    nn_dists, _ = knn_points(obj_pc[0], K=K, norm=1)

    mask = nn_dists.mean(1) < threshold
    filtered_pc = obj_pc[0, mask]
    removed_pc = obj_pc[0][~mask]

    if filtered_pc.shape[0] == 0:
        logger.info(
            "Outlier removal removed all points; falling back to original point cloud/colors"
        )
        return (
            obj_pc[0].view(-1, 3),
            obj_pc[0].new_empty((0, 3)),
            obj_pc_color[0].view(-1, 3),
            obj_pc_color[0].new_empty((0, 3)),
        )

    filtered_pc = filtered_pc.view(-1, 3)
    removed_pc = removed_pc.view(-1, 3)

    filtered_pc_color = obj_pc_color[0, mask]
    removed_pc_color = obj_pc_color[0][~mask]
    filtered_pc_color = filtered_pc_color.view(-1, 3)
    removed_pc_color = removed_pc_color.view(-1, 3)

    logger.info(
        f"Removed {obj_pc.shape[1] - filtered_pc.shape[0]} points from point cloud"
    )
    return filtered_pc, removed_pc, filtered_pc_color, removed_pc_color


def depth_and_segmentation_to_point_clouds(
    depth_image: np.ndarray,
    segmentation_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_image: np.ndarray = None,
    target_object_id: int = 1,
    remove_object_from_scene: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert depth image and instance segmentation mask to scene and object point clouds.

    Args:
        depth_image: HxW depth image in meters
        segmentation_mask: HxW instance segmentation mask with integer labels
        fx, fy, cx, cy: Camera intrinsic parameters
        rgb_image: HxWx3 RGB image (optional, for colored point clouds)
        target_object_id: ID of the target object in the segmentation mask
        remove_object_from_scene: If True, removes object points from scene point cloud

    Returns:
        scene_pc: Nx3 point cloud of the entire scene (excluding object if remove_object_from_scene=True)
        object_pc: Mx3 point cloud of the target object only
        scene_colors: Nx3 RGB colors for scene points (or None)
        object_colors: Mx3 RGB colors for object points (or None)

    Raises:
        ValueError: If no target object found or multiple objects detected
    """
    # Check that segmentation mask contains the target object
    unique_ids = np.unique(segmentation_mask)
    if target_object_id not in unique_ids:
        raise ValueError(
            f"Target object ID {target_object_id} not found in segmentation mask. Available IDs: {unique_ids}"
        )

    # Check that only background (0) and one object (target_object_id) are present
    non_background_ids = unique_ids[unique_ids != 0]
    if len(non_background_ids) > 1:
        raise ValueError(
            f"Multiple objects detected in segmentation mask: {non_background_ids}. Please ensure only one object is present."
        )

    # Convert depth image to point cloud
    pts_data = depth2points(
        depth=depth_image,
        fx=int(fx),
        fy=int(fy),
        cx=int(cx),
        cy=int(cy),
        rgb=rgb_image,
        seg=segmentation_mask,
    )

    xyz = pts_data["xyz"]
    rgb = pts_data["rgb"]
    seg = pts_data["seg"]
    index = pts_data["index"]

    # Filter valid points (non-zero depth)
    xyz_valid = xyz[index]
    seg_valid = seg[index] if seg is not None else None
    rgb_valid = rgb[index] if rgb is not None else None

    # Scene point cloud (all valid points)
    scene_pc = xyz_valid
    scene_colors = rgb_valid

    # Object point cloud (only target object points)
    if seg_valid is not None:
        object_mask = seg_valid.flatten() == target_object_id
        object_pc = xyz_valid[object_mask]
        object_colors = rgb_valid[object_mask] if rgb_valid is not None else None

        # Scene point cloud (optionally excluding object points)
        if remove_object_from_scene:
            scene_mask = ~object_mask  # Invert object mask to get scene-only points
            scene_pc = xyz_valid[scene_mask]
            scene_colors = rgb_valid[scene_mask] if rgb_valid is not None else None
            logger.info(
                f"Removed {np.sum(object_mask)} object points from scene point cloud"
            )
    else:
        raise ValueError("Segmentation data not available from depth2points")

    if len(object_pc) == 0:
        raise ValueError(f"No points found for target object ID {target_object_id}")

    logger.info(f"Scene point cloud: {len(scene_pc)} points")
    logger.info(f"Object point cloud: {len(object_pc)} points")

    return scene_pc, object_pc, scene_colors, object_colors


def filter_colliding_grasps(
    scene_pc: np.ndarray,
    grasp_poses: np.ndarray,
    gripper_collision_mesh: trimesh.Trimesh,
    collision_threshold: float = 0.002,
    num_collision_samples: int = 2000,
) -> np.ndarray:
    """
    Filter grasps based on collision detection with scene point cloud.

    Args:
        scene_pc: Nx3 scene point cloud
        grasp_poses: Kx4x4 array of grasp poses
        gripper_collision_mesh: Trimesh of gripper collision geometry
        collision_threshold: Distance threshold for collision detection (meters)
        num_collision_samples: Number of points to sample from gripper mesh surface

    Returns:
        collision_mask: K-length boolean array, True if grasp is collision-free
    """
    # Sample points from gripper collision mesh surface
    gripper_surface_points, _ = trimesh.sample.sample_surface(
        gripper_collision_mesh, num_collision_samples
    )
    gripper_surface_points = np.array(gripper_surface_points)

    # Convert inputs to torch tensors
    scene_pc_torch = torch.from_numpy(scene_pc).float()
    collision_free_mask = []

    logger.info(
        f"Checking collision for {len(grasp_poses)} grasps against {len(scene_pc)} scene points..."
    )

    for i, grasp_pose in tqdm(
        enumerate(grasp_poses), total=len(grasp_poses), desc="Collision checking"
    ):
        # Transform gripper collision points to grasp pose
        gripper_points_transformed = tra.transform_points(
            gripper_surface_points, grasp_pose
        )
        gripper_points_torch = torch.from_numpy(gripper_points_transformed).float()

        # For each gripper point, find distance to closest scene point
        min_distances = []

        # Process in batches to avoid memory issues
        batch_size = 100
        for j in range(0, len(gripper_points_torch), batch_size):
            batch_gripper_points = gripper_points_torch[j : j + batch_size]

            # Compute distances from batch of gripper points to all scene points
            distances = torch.cdist(
                batch_gripper_points, scene_pc_torch, p=2
            )  # Euclidean distance
            batch_min_distances = torch.min(distances, dim=1)[0]
            min_distances.append(batch_min_distances)

        # Concatenate all minimum distances
        all_min_distances = torch.cat(min_distances)

        # Check if any gripper point is within collision threshold of scene points
        collision_detected = torch.any(all_min_distances < collision_threshold)
        collision_free_mask.append(not collision_detected.item())

    collision_free_mask = np.array(collision_free_mask)
    num_collision_free = np.sum(collision_free_mask)
    logger.info(f"Found {num_collision_free}/{len(grasp_poses)} collision-free grasps")

    return collision_free_mask


def depth_and_segmentation_to_point_clouds(
    depth_image: np.ndarray,
    segmentation_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_image: np.ndarray = None,
    target_object_id: int = 1,
    remove_object_from_scene: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert depth image and instance segmentation mask to scene and object point clouds.

    Args:
        depth_image: HxW depth image in meters
        segmentation_mask: HxW instance segmentation mask with integer labels
        fx, fy, cx, cy: Camera intrinsic parameters
        rgb_image: HxWx3 RGB image (optional, for colored point clouds)
        target_object_id: ID of the target object in the segmentation mask
        remove_object_from_scene: If True, removes object points from scene point cloud

    Returns:
        scene_pc: Nx3 point cloud of the entire scene (excluding object if remove_object_from_scene=True)
        object_pc: Mx3 point cloud of the target object only
        scene_colors: Nx3 RGB colors for scene points (or None)
        object_colors: Mx3 RGB colors for object points (or None)

    Raises:
        ValueError: If no target object found or multiple objects detected
    """
    # Check that segmentation mask contains the target object
    unique_ids = np.unique(segmentation_mask)
    if target_object_id not in unique_ids:
        raise ValueError(
            f"Target object ID {target_object_id} not found in segmentation mask. Available IDs: {unique_ids}"
        )

    # Check that only background (0) and one object (target_object_id) are present
    non_background_ids = unique_ids[unique_ids != 0]
    if len(non_background_ids) > 1:
        raise ValueError(
            f"Multiple objects detected in segmentation mask: {non_background_ids}. Please ensure only one object is present."
        )

    # Convert depth image to point cloud
    pts_data = depth2points(
        depth=depth_image,
        fx=int(fx),
        fy=int(fy),
        cx=int(cx),
        cy=int(cy),
        rgb=rgb_image,
        seg=segmentation_mask,
    )

    xyz = pts_data["xyz"]
    rgb = pts_data["rgb"]
    seg = pts_data["seg"]
    index = pts_data["index"]

    # Filter valid points (non-zero depth)
    xyz_valid = xyz[index]
    seg_valid = seg[index] if seg is not None else None
    rgb_valid = rgb[index] if rgb is not None else None

    # Scene point cloud (all valid points)
    scene_pc = xyz_valid
    scene_colors = rgb_valid

    # Object point cloud (only target object points)
    if seg_valid is not None:
        object_mask = seg_valid.flatten() == target_object_id
        object_pc = xyz_valid[object_mask]
        object_colors = rgb_valid[object_mask] if rgb_valid is not None else None

        # Scene point cloud (optionally excluding object points)
        if remove_object_from_scene:
            scene_mask = ~object_mask  # Invert object mask to get scene-only points
            scene_pc = xyz_valid[scene_mask]
            scene_colors = rgb_valid[scene_mask] if rgb_valid is not None else None
            logger.info(
                f"Removed {np.sum(object_mask)} object points from scene point cloud"
            )
    else:
        raise ValueError("Segmentation data not available from depth2points")

    if len(object_pc) == 0:
        raise ValueError(f"No points found for target object ID {target_object_id}")

    logger.info(f"Scene point cloud: {len(scene_pc)} points")
    logger.info(f"Object point cloud: {len(object_pc)} points")

    return scene_pc, object_pc, scene_colors, object_colors


def filter_colliding_grasps(
    scene_pc: np.ndarray,
    grasp_poses: np.ndarray,
    gripper_collision_mesh: trimesh.Trimesh,
    collision_threshold: float = 0.002,
    num_collision_samples: int = 2000,
) -> np.ndarray:
    """
    Filter grasps based on collision detection with scene point cloud.

    Args:
        scene_pc: Nx3 scene point cloud
        grasp_poses: Kx4x4 array of grasp poses
        gripper_collision_mesh: Trimesh of gripper collision geometry
        collision_threshold: Distance threshold for collision detection (meters)
        num_collision_samples: Number of points to sample from gripper mesh surface

    Returns:
        collision_mask: K-length boolean array, True if grasp is collision-free
    """
    # Sample points from gripper collision mesh surface
    gripper_surface_points, _ = trimesh.sample.sample_surface(
        gripper_collision_mesh, num_collision_samples
    )
    gripper_surface_points = np.array(gripper_surface_points)

    # Convert inputs to torch tensors
    scene_pc_torch = torch.from_numpy(scene_pc).float()
    collision_free_mask = []

    logger.info(
        f"Checking collision for {len(grasp_poses)} grasps against {len(scene_pc)} scene points..."
    )

    for i, grasp_pose in tqdm(
        enumerate(grasp_poses), total=len(grasp_poses), desc="Collision checking"
    ):
        # Transform gripper collision points to grasp pose
        gripper_points_transformed = tra.transform_points(
            gripper_surface_points, grasp_pose
        )
        gripper_points_torch = torch.from_numpy(gripper_points_transformed).float()

        # For each gripper point, find distance to closest scene point
        min_distances = []

        # Process in batches to avoid memory issues
        batch_size = 100
        for j in range(0, len(gripper_points_torch), batch_size):
            batch_gripper_points = gripper_points_torch[j : j + batch_size]

            # Compute distances from batch of gripper points to all scene points
            distances = torch.cdist(
                batch_gripper_points, scene_pc_torch, p=2
            )  # Euclidean distance
            batch_min_distances = torch.min(distances, dim=1)[0]
            min_distances.append(batch_min_distances)

        # Concatenate all minimum distances
        all_min_distances = torch.cat(min_distances)

        # Check if any gripper point is within collision threshold of scene points
        collision_detected = torch.any(all_min_distances < collision_threshold)
        collision_free_mask.append(not collision_detected.item())

    collision_free_mask = np.array(collision_free_mask)
    num_collision_free = np.sum(collision_free_mask)
    logger.info(f"Found {num_collision_free}/{len(grasp_poses)} collision-free grasps")

    return collision_free_mask
