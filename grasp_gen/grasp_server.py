# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys

import time
import numpy as np
import omegaconf
import torch
from pathlib import Path

from grasp_gen.dataset.dataset import collate
from grasp_gen.models.grasp_gen import GraspGen
from grasp_gen.models.m2t2 import M2T2
from grasp_gen.utils.point_cloud_utils import knn_points, point_cloud_outlier_removal
from grasp_gen.robot import load_control_points_for_visualization
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


def score_grasps_with_discriminator(
    model,
    points_centered: torch.Tensor,
    grasps_centered: torch.Tensor,
) -> torch.Tensor:
    """Run only the discriminator on a batch of object-centered grasps.

    Shared by ``GraspGenSampler.sample`` (for ranking diffusion grasps) and by
    the GraspMoE / OBB planner (for ranking OBB-swept candidates against the
    same point cloud).

    Args:
        model: ``GraspGen`` model instance (must have ``grasp_discriminator``).
        points_centered: (N, 3) point cloud centered at its own mean, on the
            same device as ``model``.
        grasps_centered: (M, 4, 4) grasp poses with translations already
            decentered by the same ``obj_pcd_center`` used for ``points_centered``.

    Returns:
        (M,) float32 tensor of discriminator confidences in [0, 1].
    """
    color_zeros = torch.zeros_like(points_centered)
    data = {
        "task": "pick",
        "points": points_centered,
        "inputs": torch.cat([points_centered, color_zeros[:, :3]], dim=-1).float(),
        "grasps": grasps_centered,
        "grasp_key": "grasps",
    }
    data_batch = collate([data])
    data_batch["grasp_key"] = "grasps"
    with torch.inference_mode():
        out, _, _ = model.grasp_discriminator.infer(data_batch)
    return out["grasp_confidence"][0, :, 0].float()


def load_grasp_cfg(gripper_config: str) -> omegaconf.DictConfig:
    """
    Loads the grasp configuration file and updates the checkpoint paths to be relative to the gripper config file.

    Assumes that the checkpoint paths are in the same directory as the gripper config file.

    Args:
        gripper_config: Path to the gripper configuration file
    Returns:
        grasp_cfg: Hydra config object with updated checkpoint paths
    """
    cfg = omegaconf.OmegaConf.load(gripper_config)
    ckpt_root_dir = Path(gripper_config).parent
    cfg.eval.checkpoint = str(ckpt_root_dir / cfg.eval.checkpoint)
    cfg.discriminator.checkpoint = str(ckpt_root_dir / cfg.discriminator.checkpoint)
    assert (
        cfg.data.gripper_name
        == cfg.diffusion.gripper_name
        == cfg.discriminator.gripper_name
    )
    return cfg


class GraspGenSampler:
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
    ):
        """
        Args:
            cfg: Hydra config object
        """

        self.cfg = cfg
        # Initialize model and load checkpoint
        if cfg.eval.model_name == "m2t2":
            model = M2T2.from_config(cfg.m2t2)
            ckpt = torch.load(cfg.eval.checkpoint)
            model.load_state_dict(ckpt["model"])
        elif cfg.eval.model_name == "diffusion-discriminator":
            model = GraspGen.from_config(
                cfg.diffusion,
                cfg.discriminator,
            )
            if not os.path.exists(cfg.eval.checkpoint):
                raise FileNotFoundError(
                    f"Checkpoint {cfg.eval.checkpoint} does not exist"
                )
            if not os.path.exists(cfg.discriminator.checkpoint):
                raise FileNotFoundError(
                    f"Checkpoint {cfg.discriminator.checkpoint} does not exist"
                )

            model.load_state_dict(cfg.eval.checkpoint, cfg.discriminator.checkpoint)
            model.eval()
        else:
            raise NotImplementedError(
                f"Model name not implemented {cfg.eval.model_name}"
            )

        self.model = model.cuda().eval()

    @staticmethod
    def run_inference(
        object_pc: np.ndarray | torch.Tensor,
        grasp_sampler: "GraspGenSampler",
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = -1,
        min_grasps: int = 40,
        max_tries: int = 6,
        remove_outliers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run grasp generation inference on a point cloud.

        Args:
            object_pc: Point cloud to generate grasps for
            grasp_sampler: Initialized GraspGenSampler instance
            grasp_threshold: Threshold for valid grasps. If -1.0, then the top topk_grasps grasps will be ranked and returned
            num_grasps: Number of grasps to generate
            topk_grasps: Maximum number of grasps to return
            min_grasps: Minimum number of grasps required. If fewer grasps are found, inference will be retried
            max_tries: Maximum number of inference attempts to make before returning results

        Returns:
            grasps: Generated grasp poses
            grasp_conf: Confidence scores for the grasps
        """
        if type(object_pc) == np.ndarray:
            object_pc = torch.from_numpy(object_pc).cuda().float()

        if grasp_threshold == -1.0 and topk_num_grasps == -1:
            topk_num_grasps = 100

        all_grasps = []
        all_conf = []
        num_tries = 0

        while sum(len(g) for g in all_grasps) < min_grasps and num_tries < max_tries:
            num_tries += 1
            t0 = time.time()
            output = grasp_sampler.sample(
                object_pc,
                threshold=grasp_threshold,
                num_grasps=num_grasps,
                remove_outliers=remove_outliers,
            )
            grasp_conf = output[1]
            grasps = output[0]

            # Sort and prune grasps within this iteration
            if topk_num_grasps != -1 and len(grasps) > 0:
                grasp_conf, grasps = zip(
                    *sorted(zip(grasp_conf, grasps), key=lambda x: x[0], reverse=True)
                )
                grasps = torch.stack(grasps)
                grasp_conf = torch.stack(grasp_conf)
                grasps = grasps[:topk_num_grasps]
                grasp_conf = grasp_conf[:topk_num_grasps]

            all_grasps.append(grasps)
            all_conf.append(grasp_conf)

            logger.info(
                f"Found {len(grasps)} grasps in iteration {len(all_grasps)}, total grasps: {sum(len(g) for g in all_grasps)}"
            )
            t1 = time.time()
            logger.info(f"Time taken for inference: {t1 - t0} seconds")

        if len(all_grasps) == 0:
            return torch.tensor([]), torch.tensor([])

        # Concatenate all grasps and confidences
        grasps = torch.cat(all_grasps, dim=0)
        grasp_conf = torch.cat(all_conf, dim=0)
        grasps[:, 3, 3] = 1  # TODO: Fix this in grasp_gen.py later.

        return grasps, grasp_conf

    @staticmethod
    def run_inference_batch(
        object_pcs: list,
        grasp_sampler: "GraspGenSampler",
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = -1,
        remove_outliers: bool = True,
    ) -> list:
        """Batched :meth:`run_inference`: one diffusion + discriminator forward
        pass over N object PCs. Equivalent to
        ``[run_inference(pc, ...) for pc in object_pcs]`` but folds the
        reverse-diffusion loop into a single batched call.

        Each input PC is resampled (with replacement) to
        ``grasp_sampler.cfg.data.num_points`` so that ``collate`` can stack them
        (it uses ``torch.stack``). Per-object centers are tracked and added back
        to the predicted grasps so every output is in the input world frame.

        Only the ``diffusion-discriminator`` model is supported (m2t2 has no
        batched path here).

        Args:
            object_pcs: list of N point clouds, each (Mi, 3) np.ndarray or
                torch.Tensor in the same world frame.
            grasp_threshold: per-object discriminator threshold; -1.0 keeps
                everything then prunes by top-k.
            num_grasps: diffusion samples per object.
            topk_num_grasps: cap per object after thresholding; -1 means no
                top-k cap unless ``grasp_threshold == -1.0`` (in which case it
                defaults to 100, matching :meth:`run_inference`).
            remove_outliers: run outlier removal per object before resampling.

        Returns:
            List of ``(grasps, grasp_conf)`` tuples, one per input PC, in input
            order. Shapes: grasps (Ki, 4, 4), grasp_conf (Ki,) on the sampler's
            device.
        """
        if len(object_pcs) == 0:
            return []
        if grasp_sampler.cfg.eval.model_name != "diffusion-discriminator":
            raise NotImplementedError(
                "run_inference_batch only supports the diffusion-discriminator "
                f"model, got '{grasp_sampler.cfg.eval.model_name}'."
            )
        if grasp_threshold == -1.0 and topk_num_grasps == -1:
            topk_num_grasps = 100

        device = next(grasp_sampler.model.parameters()).device
        target_n = int(grasp_sampler.cfg.data.num_points)

        centers: list = []
        data_items: list = []
        for pc in object_pcs:
            if isinstance(pc, np.ndarray):
                pc_t = torch.from_numpy(pc).float()
            else:
                pc_t = pc.float()
            if remove_outliers:
                pc_t, _ = point_cloud_outlier_removal(pc_t)
            # Resample with replacement to the training-time budget so all
            # batch items have the same shape (collate uses torch.stack).
            n = pc_t.shape[0]
            if n != target_n:
                idx = torch.randint(0, n, (target_n,))
                pc_t = pc_t[idx]
            pc_t = pc_t.to(device)
            center = pc_t.mean(dim=0)
            centers.append(center)
            centered = pc_t - center[None]
            color = torch.zeros_like(centered)

            data = {}
            data["task"] = "pick"
            data["inputs"] = torch.cat([centered, color[:, :3]], dim=-1).float()
            data["points"] = centered
            data_items.append(data)

        data_batch = collate(data_items)
        data_batch["grasp_key"] = "grasps_pred"

        # Set the diffusion sampler to produce num_grasps per object; the
        # model's batched forward returns (N, num_grasps, 4, 4).
        grasp_sampler.model.grasp_generator.num_grasps_per_object = num_grasps

        with torch.inference_mode():
            model_outputs, _, _ = grasp_sampler.model.infer(data_batch)

        grasps_pred = model_outputs["grasps_pred"]  # (N, K, 4, 4)
        grasp_conf = model_outputs["grasp_confidence"][..., 0]  # (N, K)

        outputs: list = []
        for i, center in enumerate(centers):
            g_i = grasps_pred[i].to(device)
            c_i = grasp_conf[i].to(device)
            if grasp_threshold > 0.0:
                keep = c_i >= grasp_threshold
                g_i = g_i[keep]
                c_i = c_i[keep]
            if topk_num_grasps != -1 and len(g_i) > topk_num_grasps:
                order = torch.argsort(c_i, descending=True)[:topk_num_grasps]
                g_i = g_i[order]
                c_i = c_i[order]
            # Restore world frame.
            if len(g_i) > 0:
                g_i = g_i.clone()
                g_i[:, :3, 3] = g_i[:, :3, 3] + center.to(g_i.device)
                g_i[:, 3, 3] = 1.0
            outputs.append((g_i, c_i))
        return outputs

    @torch.inference_mode()
    def sample(
        self,
        obj_pcd: np.ndarray,
        threshold: float = -1.0,
        num_grasps: int = 200,
        remove_outliers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obj_pcd: np.array of shape (N, 3)
            obj_pts_color (Optional): np.array of shape (N, 4)

        Returns:
            grasps: torch.tensor of shape (M, 6)
            grasp_conf: torch.tensor of shape (M,)
            grasp_contacts: torch.tensor of shape (M, 3)
        """

        if isinstance(obj_pcd, torch.Tensor):
            obj_pcd = obj_pcd.detach().cpu().numpy()

        if remove_outliers:
            obj_pcd, _ = point_cloud_outlier_removal(obj_pcd)

        obj_pcd = torch.from_numpy(obj_pcd).float()
        device = next(self.model.parameters()).device
        obj_pcd = obj_pcd.to(device)

        obj_pcd_center = obj_pcd.mean(dim=0)
        obj_pts_color = torch.zeros_like(obj_pcd)
        obj_mean_points = obj_pcd - obj_pcd_center[None]

        data = {}
        data["task"] = "pick"
        data["inputs"] = torch.cat(
            [obj_mean_points, obj_pts_color[:, :3].squeeze(1)], dim=-1
        ).float()
        data["points"] = obj_mean_points

        data_batch = collate([data])
        grasp_key = "grasps"
        with torch.inference_mode():
            if self.cfg.eval.model_name == "m2t2":
                model_outputs = self.model.infer(data_batch, self.cfg.eval)
            elif self.cfg.eval.model_name == "diffusion-discriminator":
                grasp_key = "grasps_pred"
                self.model.grasp_generator.num_grasps_per_object = num_grasps

                model_outputs, _, _ = self.model.infer(data_batch)

            else:
                raise NotImplementedError(f"Invalid model {self.cfg.eval.model_name}!")

        if len(model_outputs[grasp_key][0]) == 0:
            device = next(self.model.parameters()).device
            return (
                torch.empty(0, 4, 4, device=device),
                torch.empty(0, device=device),
                torch.empty(0, 3, device=device),
            )

        grasps = model_outputs[grasp_key][0]

        if self.cfg.eval.model_name == "diffusion-discriminator":
            grasp_conf = model_outputs["grasp_confidence"][0][:, 0]
            logger.info(
                f"Confidences min: {grasp_conf.min():.5f}, max: {grasp_conf.max():.5f}"
            )
            mask_best_grasps = grasp_conf >= threshold
            logger.info(
                f"Thresholding grasps @ {threshold}. Only {mask_best_grasps.sum()}/{mask_best_grasps.shape[0]} grasps remaining"
            )

            grasps = grasps[mask_best_grasps]
            grasp_conf = grasp_conf[mask_best_grasps]

        elif self.cfg.eval.model_name == "m2t2":
            grasps = grasps[0]
            grasp_conf = model_outputs["grasp_confidence"][0][0]
        else:
            raise NotImplementedError(f"Invalid model {self.cfg.eval.model_name}!")
        grasps[:, :3, 3] += obj_pcd_center
        return grasps, grasp_conf, None

    # def get_grasp_points(self, pose_array: Pose, gripper_name = None):
    #     if gripper_name is None:
    #         gripper_name = self.cfg.data.gripper_name
    #     line_points = load_control_points_for_visualization(gripper_name)
    #     line_points = torch.as_tensor(line_points, device="cuda", dtype=torch.float32).view(-1,3)

    #     line_points = line_points.unsqueeze(0).repeat(pose_array.shape[0],1,1)

    #     line_points = pose_array.batch_transform_points(line_points)

    #     return line_points
