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
Inference performance benchmark for GraspGen.

Loads the model and a test object, benchmarks inference across acceleration
modes (None / torch.compile), and reports a comparison table with latency,
frequency, and speedup.

Usage:
    # Default mesh (assets/objects/box.obj):
    pytest tests/test_inference_perf.py -v -s

    # Custom mesh:
    pytest tests/test_inference_perf.py -v -s --mesh assets/objects/banana.obj
    pytest tests/test_inference_perf.py -v -s --mesh /absolute/path/to/object.obj

Benchmark parameters:
    WARMUP_SECONDS   = 30.0   Wall-clock warmup time per mode (seconds).
    WARMUP_MIN_ITERS = 100    Minimum warmup iterations before timing starts.
    BENCH_ITERS      = 500    Number of timed iterations per mode.
    Backbones        = pointnet, ptv3
    Acceleration     = None (eager baseline)
    Num grasps       = 100
    Diffusion steps  = 10

Each timed iteration is bracketed by torch.cuda.synchronize() calls
for accurate GPU measurement.

Reference numbers (NVIDIA GeForce RTX 3090, random weights, 100 grasps, 10 diffusion steps):

  box.obj:
    Backbone | Latency (ms)        | Frequency (Hz)
    ---------|---------------------|--------------------
    pointnet |  13.22 +/- 0.63     |  75.66 +/- 3.61
    ptv3     |  36.74 +/- 0.75     |  27.22 +/- 0.55

  banana.obj:
    Backbone | Latency (ms)        | Frequency (Hz)
    ---------|---------------------|--------------------
    pointnet |  12.35 +/- 0.39     |  80.94 +/- 2.55
    ptv3     |  35.82 +/- 1.48     |  27.92 +/- 1.16

Note: torch.compile silently falls back to eager for both backbones due to
extensive graph breaks (45+ each). Root causes:
  - pointnet: pointnet2_ops custom autograd Functions (grouping_operation,
    ball_query, etc.) cannot be traced by Dynamo.
  - ptv3: spconv PyCapsule C++ ops and addict dynamic __setattr__.
"""

import time
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import trimesh
from omegaconf import DictConfig
from pathlib import Path

from grasp_gen.models.grasp_gen import GraspGen
from grasp_gen.dataset.dataset import collate


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_generator_cfg(backbone: str) -> DictConfig:
    return DictConfig({
        "num_embed_dim": 256,
        "num_obs_dim": 512,
        "diffusion_embed_dim": 512,
        "image_size": 256,
        "num_diffusion_iters": 10,
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


@dataclass
class BenchResult:
    latencies_s: np.ndarray
    warmup_iters: int
    warmup_elapsed: float

    @property
    def mean_ms(self) -> float:
        return self.latencies_s.mean() * 1000.0

    @property
    def std_ms(self) -> float:
        return self.latencies_s.std() * 1000.0

    @property
    def hz_mean(self) -> float:
        return 1.0 / self.latencies_s.mean()

    @property
    def hz_std(self) -> float:
        return self.latencies_s.std() / (self.latencies_s.mean() ** 2)


def _bench_loop(model, data_batch, warmup_seconds, warmup_min_iters, bench_iters) -> BenchResult:
    warmup_start = time.perf_counter()
    warmup_iters = 0
    while True:
        with torch.inference_mode():
            model.infer(data_batch)
        torch.cuda.synchronize()
        warmup_iters += 1
        if warmup_iters >= warmup_min_iters and (time.perf_counter() - warmup_start) >= warmup_seconds:
            break
    warmup_elapsed = time.perf_counter() - warmup_start

    latencies = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.infer(data_batch)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    return BenchResult(np.array(latencies), warmup_iters, warmup_elapsed)


ACCEL_MODES = [
    ("None", None),
]


def _print_table(backbone: str, mesh_name: str, rows):
    """Print a formatted comparison table."""
    col_w = [17, 22, 22, 9]
    sep = "  "

    header = sep.join([
        f"{'Backbone':<10}",
        f"{'Acceleration':<{col_w[0]}}",
        f"{'Latency (ms)':>{col_w[1]}}",
        f"{'Frequency (Hz)':>{col_w[2]}}",
        f"{'Speedup':>{col_w[3]}}",
    ])
    divider = sep.join([
        "-" * 10,
        "-" * col_w[0],
        "-" * col_w[1],
        "-" * col_w[2],
        "-" * col_w[3],
    ])

    lines = [
        f"\n{'=' * len(header)}",
        f"  Mesh: {mesh_name}",
        header,
        divider,
    ]
    for name, result, speedup in rows:
        lat_str = f"{result.mean_ms:.2f} +/- {result.std_ms:.2f}"
        hz_str = f"{result.hz_mean:.2f} +/- {result.hz_std:.2f}"
        sp_str = f"{speedup:.2f}x"
        lines.append(sep.join([
            f"{backbone:<10}",
            f"{name:<{col_w[0]}}",
            f"{lat_str:>{col_w[1]}}",
            f"{hz_str:>{col_w[2]}}",
            f"{sp_str:>{col_w[3]}}",
        ]))
    lines.append("=" * len(header))
    print("\n".join(lines))


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mesh_path(request):
    """Resolve the mesh path from --mesh or fall back to assets/objects/box.obj."""
    custom = request.config.getoption("--mesh")
    if custom is not None:
        p = Path(custom)
        if not p.is_absolute():
            p = Path(__file__).parent.parent / p
        if not p.exists():
            pytest.skip(f"Mesh not found at {p}")
        return p
    default = Path(__file__).parent.parent / "assets" / "objects" / "box.obj"
    if not default.exists():
        pytest.skip(f"Default mesh not found at {default}")
    return default


@pytest.fixture
def point_cloud_from_mesh(mesh_path):
    mesh = trimesh.load(str(mesh_path))
    mesh.apply_translation(-mesh.center_mass)
    points, _ = trimesh.sample.sample_surface(mesh, 2000)
    return torch.tensor(points, dtype=torch.float32)


# ─── Benchmark ───────────────────────────────────────────────────────────────

WARMUP_SECONDS = 30.0
WARMUP_MIN_ITERS = 100
BENCH_ITERS = 500


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backbone", ["pointnet", "ptv3"])
def test_inference_perf(backbone, point_cloud_from_mesh, mesh_path):
    """
    Benchmark model.infer() across acceleration modes.

    For each mode (None, torch.compile) on the given backbone:
      1. Build / transform the model.
      2. Warmup for >= WARMUP_MIN_ITERS and >= WARMUP_SECONDS.
      3. Time BENCH_ITERS calls with torch.cuda.synchronize().
    Then print a comparison table with latency, frequency, and speedup
    relative to the None (baseline) mode.
    """
    device = torch.device("cuda")

    gen_cfg = _make_generator_cfg(backbone)
    disc_cfg = _make_discriminator_cfg(backbone)
    base_model = GraspGen.from_config(gen_cfg, disc_cfg).to(device).eval()
    base_model.grasp_generator.num_grasps_per_object = 100

    data_batch = _prepare_batch(point_cloud_from_mesh, device)

    rows = []
    baseline_mean_s = None

    for name, transform in ACCEL_MODES:
        model = transform(base_model) if transform is not None else base_model
        result = _bench_loop(model, data_batch, WARMUP_SECONDS, WARMUP_MIN_ITERS, BENCH_ITERS)

        if baseline_mean_s is None:
            baseline_mean_s = result.latencies_s.mean()

        speedup = baseline_mean_s / result.latencies_s.mean()
        rows.append((name, result, speedup))

    _print_table(backbone, mesh_path.name, rows)

    for name, result, _ in rows:
        assert result.mean_ms > 0, f"{name}: latency must be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
