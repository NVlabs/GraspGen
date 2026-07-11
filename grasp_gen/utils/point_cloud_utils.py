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


def filter_colliding_grasps_fast(
    scene_pc: np.ndarray,
    grasp_poses: np.ndarray,
    gripper_collision_mesh: "trimesh.Trimesh | None" = None,
    collision_threshold: float = 0.02,
    num_collision_samples: int = 2000,
    batch_size: int = 16,
    gripper_surface_points: "np.ndarray | None" = None,
    device: "str | torch.device | None" = None,
) -> np.ndarray:
    """GPU-vectorized collision filter — drop-in replacement for
    :func:`filter_colliding_grasps` that is ~10-50x faster on GPU.

    Algorithm:
      1. Uniformly sample ``num_collision_samples`` points (= M) on the
         gripper collision mesh surface (gripper-local frame). Skipped when
         ``gripper_surface_points`` is supplied.
      2. Vectorized over grasps in chunks of ``batch_size`` poses:
         transform the M gripper samples into world frame via the chunk's
         rotations + translations, then run a single ``torch.cdist``
         against the scene point cloud on GPU. The cdist intermediate is
         ``(batch_size * M, N)`` floats — chunk size bounds memory.
      3. A grasp is collision-free iff every one of its M samples'
         nearest-scene-point distance is ≥ ``collision_threshold``.

    Args:
        scene_pc: (N, 3) scene point cloud (target object's points should
                  already be removed by the caller).
        grasp_poses: (K, 4, 4) grasp poses in the same frame as scene_pc.
        gripper_collision_mesh: trimesh.Trimesh of the gripper geometry.
                                Required only if ``gripper_surface_points``
                                is not provided.
        collision_threshold: meters; gripper samples within this distance of
                             any scene point count as a collision.
        num_collision_samples: number of points sampled on the gripper. Used
                               only when ``gripper_surface_points`` is None.
        batch_size: grasps per vectorized cdist call. The per-call distance
                    matrix is ``batch_size * M * N`` fp32 entries — keep
                    ``batch_size * M * N * 4 bytes`` under available GPU
                    memory. Default 16 ≈ 1 GB at M=2000, N=8192.
        gripper_surface_points: optional (M, 3) array of pre-sampled gripper
                                surface points in gripper-local frame. When
                                supplied, the per-call
                                ``trimesh.sample.sample_surface`` is skipped
                                — sample once outside the hot loop and
                                reuse.
        device: torch device for the cdist work. Defaults to "cuda" if
                available, else "cpu".

    Returns:
        (K,) bool ndarray; True == collision-free.
    """
    K = len(grasp_poses)
    if K == 0:
        return np.zeros((0,), dtype=bool)
    if len(scene_pc) == 0:
        logger.info("[collision_fast] scene_pc empty → all grasps marked collision-free.")
        return np.ones((K,), dtype=bool)

    if gripper_surface_points is None:
        if gripper_collision_mesh is None:
            raise ValueError(
                "filter_colliding_grasps_fast: must provide gripper_collision_mesh "
                "or pre-sampled gripper_surface_points"
            )
        sampled, _ = trimesh.sample.sample_surface(
            gripper_collision_mesh, num_collision_samples
        )
        gripper_surface_points = np.asarray(sampled, dtype=np.float32)
    else:
        gripper_surface_points = np.asarray(gripper_surface_points, dtype=np.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    scene_t = torch.as_tensor(scene_pc, dtype=torch.float32, device=device)
    pts_local = torch.as_tensor(
        gripper_surface_points, dtype=torch.float32, device=device
    )
    poses = torch.as_tensor(
        np.asarray(grasp_poses, dtype=np.float32), dtype=torch.float32, device=device
    )

    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    M = pts_local.shape[0]

    logger.info(
        f"[collision_fast] checking {K} grasps against {len(scene_pc)} scene "
        f"points (thr={collision_threshold:.3f}m, samples={M}, "
        f"device={device}, chunk={batch_size})"
    )

    collision_free = torch.empty((K,), dtype=torch.bool, device=device)
    for s in range(0, K, batch_size):
        e = min(s + batch_size, K)
        Kc = e - s
        pts_world = torch.einsum("kij,mj->kmi", R[s:e], pts_local) + t[s:e].unsqueeze(1)
        flat = pts_world.reshape(Kc * M, 3)
        d = torch.cdist(flat, scene_t, p=2)
        min_d = d.amin(dim=1).view(Kc, M)
        collision_free[s:e] = ~torch.any(min_d < collision_threshold, dim=1)

    out = collision_free.detach().cpu().numpy()
    logger.info(f"[collision_fast] {int(out.sum())}/{K} grasps collision-free")
    return out


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
