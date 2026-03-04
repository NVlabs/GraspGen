#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Test inference installation by checking if GraspGen can run end-to-end inference
with random weights (no checkpoints needed). Validates that:

1. All dependencies are correctly installed (torch, pointnet2_ops, etc.)
2. Models can be initialized with random weights for both pointnet and ptv3 backbones
3. Inference pipeline runs correctly end-to-end
4. Expected number of grasps (100) are generated
"""

import pytest
import torch
import numpy as np
import trimesh
from pathlib import Path
from omegaconf import DictConfig

from grasp_gen.models.grasp_gen import GraspGen
from grasp_gen.dataset.dataset import collate


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def box_mesh_path():
    """Path to the test box mesh in assets/."""
    assets_dir = Path(__file__).parent.parent / "assets" / "objects"
    box_path = assets_dir / "box.obj"
    if not box_path.exists():
        pytest.skip(f"Test mesh not found at {box_path}")
    return str(box_path)


@pytest.fixture
def point_cloud_from_box(box_mesh_path):
    """Load box.obj and sample 2000 centered points from its surface."""
    mesh = trimesh.load(box_mesh_path)
    mesh.apply_translation(-mesh.center_mass)
    points, _ = trimesh.sample.sample_surface(mesh, 2000)
    return torch.tensor(points, dtype=torch.float32)


def _make_generator_cfg(backbone: str) -> DictConfig:
    """Build a minimal generator config for the given backbone."""
    return DictConfig({
        "num_embed_dim": 256,
        "num_obs_dim": 512,
        "diffusion_embed_dim": 512,
        "image_size": 256,
        "num_diffusion_iters": 10,       # small for fast testing
        "num_diffusion_iters_eval": 10,
        "obs_backbone": backbone,
        "compositional_schedular": False,
        "loss_pointmatching": True,
        "loss_l1_pos": False,
        "loss_l1_rot": False,
        "grasp_repr": "r3_6d",
        "kappa": -1.0,
        "clip_sample": True,
        "beta_schedule": "squaredcos_cap_v2",
        "attention": "cat",
        "grid_size": 0.02,
        "gripper_name": "franka_panda",
        "pose_repr": "mlp",
        "num_grasps_per_object": 100,
        "checkpoint_object_encoder_pretrained": None,
        "ptv3": DictConfig({"grid_size": 0.02}),
    })


def _make_discriminator_cfg(backbone: str) -> DictConfig:
    """Build a minimal discriminator config for the given backbone."""
    return DictConfig({
        "num_obs_dim": 512,
        "obs_backbone": backbone,
        "grasp_repr": "r3_6d",
        "grid_size": 0.01,
        "sample_embed_dim": 512,
        "pose_repr": "mlp",
        "topk_ratio": 0.40,
        "checkpoint_object_encoder_pretrained": None,
        "kappa": 3.30,
        "gripper_name": "franka_panda",
        "ptv3": DictConfig({"grid_size": 0.01}),
    })


def _prepare_batch(point_cloud: torch.Tensor, device: torch.device) -> dict:
    """Prepare a collated batch dict from a point cloud, matching grasp_server.sample()."""
    pc = point_cloud.to(device)
    pc_center = pc.mean(dim=0)
    pc_centered = pc - pc_center[None]
    pc_color = torch.zeros_like(pc)

    data = {
        "task": "pick",
        "inputs": torch.cat([pc_centered, pc_color[:, :3]], dim=-1).float(),
        "points": pc_centered,
    }
    return collate([data])


def _run_inference(model, data_batch, num_grasps: int = 100):
    """Run model inference and return the predicted grasps tensor."""
    model.grasp_generator.num_grasps_per_object = num_grasps
    with torch.inference_mode():
        outputs, _, _ = model.infer(data_batch)

    assert "grasps_pred" in outputs, "Missing 'grasps_pred' key in model outputs"
    grasps = outputs["grasps_pred"][0]  # first (only) batch element
    return grasps


# ─── Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backbone", ["pointnet", "ptv3"])
def test_inference_100_grasps(backbone, point_cloud_from_box):
    """
    End-to-end inference test: create a model with random weights, feed in
    the box point cloud, and verify we get back exactly 100 grasps that are
    valid 4×4 homogeneous matrices.
    """
    device = torch.device("cuda")

    # Build model with random weights
    gen_cfg = _make_generator_cfg(backbone)
    disc_cfg = _make_discriminator_cfg(backbone)
    model = GraspGen.from_config(gen_cfg, disc_cfg).to(device).eval()

    # Prepare data & run inference
    data_batch = _prepare_batch(point_cloud_from_box, device)
    grasps = _run_inference(model, data_batch, num_grasps=100)

    # ── Check count ──
    assert len(grasps) == 100, (
        f"[{backbone}] Expected 100 grasps, got {len(grasps)}"
    )

    # ── Check shape: each grasp is a 4×4 matrix ──
    assert grasps.shape == torch.Size([100, 4, 4]), (
        f"[{backbone}] Expected shape [100, 4, 4], got {list(grasps.shape)}"
    )

    # ── Check valid homogeneous matrices ──
    bottom_rows = grasps[:, 3, :]
    expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    assert torch.allclose(bottom_rows, expected_bottom.expand_as(bottom_rows), atol=1e-5), (
        f"[{backbone}] Grasp matrices have invalid bottom row"
    )

    print(f"✅ [{backbone}] Generated {len(grasps)} valid 4×4 grasps")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backbone", ["pointnet", "ptv3"])
def test_model_components(backbone):
    """Verify generator and discriminator are properly initialised."""
    gen_cfg = _make_generator_cfg(backbone)
    disc_cfg = _make_discriminator_cfg(backbone)
    model = GraspGen.from_config(gen_cfg, disc_cfg)

    assert hasattr(model, "grasp_generator"), "Missing grasp_generator"
    assert hasattr(model, "grasp_discriminator"), "Missing grasp_discriminator"
    assert model.grasp_generator.obs_backbone == backbone
    assert model.grasp_discriminator.obs_backbone == backbone
    assert model.grasp_generator.num_grasps_per_object == 100

    print(f"✅ [{backbone}] Model components OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
