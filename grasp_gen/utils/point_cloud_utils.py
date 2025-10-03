# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
from typing import Tuple

import numpy as np
import torch

from grasp_gen.utils.logging_config import get_logger

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

    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(X, X, p=norm)  # (N, N)

    # Ignore self-distance (optional, but avoids trivial zero distance)
    self_mask = torch.eye(N, device=X.device, dtype=torch.bool)
    dist_matrix.masked_fill_(self_mask, float("inf"))  # Set self-distances to inf

    # Get the indices of the K-nearest neighbors
    dists, idxs = torch.topk(dist_matrix, K, dim=1, largest=False)

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

    nn_dists, _ = knn_points(obj_pc[0], K=K, norm=1)

    mask = nn_dists.mean(1) < threshold
    filtered_pc = obj_pc[0, mask]
    removed_pc = obj_pc[0][~mask]
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

    nn_dists, _ = knn_points(obj_pc[0], K=K, norm=1)

    mask = nn_dists.mean(1) < threshold
    filtered_pc = obj_pc[0, mask]
    removed_pc = obj_pc[0][~mask]
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