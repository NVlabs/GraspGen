#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Tests for the GraspMoE / OBB planner ported from GraspGenX.
#
# - Tests 1-7 are pure numpy/scipy and run on CPU in seconds.
# - Test 8 builds a GraspGen discriminator with random weights (no checkpoint
#   needed) and exercises the scoring-path plumbing end-to-end.
# - Test 9 spot-checks the gripper-width YAML lookup for the supported grippers.
# - Test 10 verifies the suction gripper raises a clear error.
# - Test 11 (optional, skipped if checkpoints missing) is a real end-to-end smoke.

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from grasp_gen.samplers.graspmoe import (
    _build_face_candidates,
    _compute_obb,
    _interior_positions,
    _long_axis_positions,
    _resolve_gripper_geometry,
    _run_obb_branch,
    _score_grasps_world,
    _world_aligned_top_down_grasp,
    run_graspmoe,
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_box_pc(extents=(0.08, 0.06, 0.04), n=2000, seed=0) -> np.ndarray:
    """Uniform points on the surface of an axis-aligned box centered at origin."""
    rng = np.random.RandomState(seed)
    hx, hy, hz = (e / 2.0 for e in extents)
    faces = [
        (("x", +1.0), hx, [-hy, hy], [-hz, hz]),
        (("x", -1.0), -hx, [-hy, hy], [-hz, hz]),
        (("y", +1.0), hy, [-hx, hx], [-hz, hz]),
        (("y", -1.0), -hy, [-hx, hx], [-hz, hz]),
        (("z", +1.0), hz, [-hx, hx], [-hy, hy]),
        (("z", -1.0), -hz, [-hx, hx], [-hy, hy]),
    ]
    per_face = n // 6
    pts = []
    for (axis_sign, axis_val, u_range, v_range) in faces:
        axis = axis_sign[0]
        u = rng.uniform(u_range[0], u_range[1], per_face)
        v = rng.uniform(v_range[0], v_range[1], per_face)
        if axis == "x":
            block = np.column_stack([np.full(per_face, axis_val), u, v])
        elif axis == "y":
            block = np.column_stack([u, np.full(per_face, axis_val), v])
        else:
            block = np.column_stack([u, v, np.full(per_face, axis_val)])
        pts.append(block)
    return np.concatenate(pts, axis=0).astype(np.float32)


def _rotate_pc_z(pc: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (pc @ R.T).astype(np.float32)


def make_cuboid_on_table(
    extents=(0.10, 0.04, 0.02), n=3000, seed=0
) -> np.ndarray:
    """A 10cm x 4cm x 2cm cuboid resting with its largest (10x4) face on the
    table (z = 0), i.e. 2cm tall. The top-down grasper closes across the 4cm
    width. Shared by the table-grasp test and the viser demo script."""
    pc = _make_box_pc(extents, n=n, seed=seed)
    pc[:, 2] += extents[2] / 2.0  # lift so the bottom face rests on z = 0
    return pc


# ─── Tests 1-3: OBB compute ─────────────────────────────────────────────────


def test_obb_box_aligned():
    """Axis-aligned box → OBB matches the box's center and extents."""
    pc = _make_box_pc((0.08, 0.06, 0.04))
    center, half_extent, R = _compute_obb(pc, mode="advanced")

    assert np.allclose(center, [0.0, 0.0, 0.0], atol=5e-3), (
        f"Expected center near origin, got {center}"
    )
    # X and Y order may swap if half_extent[0] < half_extent[1]; check sorted match.
    he_sorted = sorted(half_extent[:2].tolist())
    expected_sorted = sorted([0.04, 0.03])
    assert np.allclose(he_sorted, expected_sorted, atol=5e-3), (
        f"Sorted XY half_extents {he_sorted} != expected {expected_sorted}"
    )
    assert abs(half_extent[2] - 0.02) < 5e-3
    # R should be a valid rotation matrix.
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert abs(np.linalg.det(R) - 1.0) < 1e-5


def test_obb_rotated():
    """Box rotated 30° about Z → OBB recovers the rotation."""
    pc = _rotate_pc_z(_make_box_pc((0.10, 0.06, 0.04)), np.deg2rad(30.0))
    center, half_extent, R = _compute_obb(pc, mode="advanced")
    assert np.allclose(center, [0.0, 0.0, 0.0], atol=5e-3)
    # Half-extents (sorted) should still match (0.05, 0.03, 0.02).
    he_sorted = sorted(half_extent.tolist())
    expected_sorted = sorted([0.05, 0.03, 0.02])
    assert np.allclose(he_sorted, expected_sorted, atol=8e-3), (
        f"Sorted half_extents {he_sorted} != expected {expected_sorted}"
    )
    # R columns are the OBB axes — at least one of the XY columns should align
    # with cos(30°)/sin(30°) (modulo sign and X<->Y swap).
    target = np.array([np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0)), 0.0])
    cos_sims = [abs(target @ R[:, i]) for i in range(3)]
    assert max(cos_sims) > 0.95, f"No OBB axis aligned with the 30° rotation: {cos_sims}"


def test_obb_pca_fallback():
    """Forcing mode='pca' on a tiny cloud succeeds (no exception)."""
    pc = _make_box_pc((0.05, 0.04, 0.03), n=60)
    center, half_extent, R = _compute_obb(pc, mode="pca")
    assert center.shape == (3,)
    assert half_extent.shape == (3,)
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)


# ─── Test 4: skip-OBB rule ──────────────────────────────────────────────────


def test_skip_obb_rule_when_object_too_wide():
    """All extents > gripper_width → OBB branch returns 0 candidates."""
    pc = _make_box_pc((0.30, 0.30, 0.30))  # all extents 0.30 m
    grasps, scores, obb, skipped = _run_obb_branch(
        pc_filtered=pc,
        pc_filtered_centered=torch.from_numpy(pc),  # not used since we skip
        pc_center=pc.mean(axis=0).astype(np.float64),
        grasp_sampler=None,  # not invoked when skipped
        num_yaws=12,
        z_offsets_cm=(0.0,),
        obb_mode="advanced",
        gripper_width_m=0.136,  # robotiq_2f_140 width
        gripper_depth_m=0.195,
        skip_obb_rule="auto",
    )
    assert len(grasps) == 0
    assert len(scores) == 0
    assert skipped is True
    assert obb is not None  # OBB was computed even though branch was skipped


# ─── Test 5: top-down pose convention ───────────────────────────────────────


def test_top_down_pose_convention():
    """Top-down base pose has gripper Z pointing -world_Z."""
    center = np.array([0.1, 0.2, 0.05])
    half_extent = np.array([0.04, 0.03, 0.02])
    R = np.eye(3)
    T = _world_aligned_top_down_grasp(center, half_extent, R, z_offset=0.0)
    assert T.shape == (4, 4)
    # gripper Z column should be [0, 0, -1]
    assert np.allclose(T[:3, 2], [0.0, 0.0, -1.0], atol=1e-6)
    # X above OBB top (center.z + half_extent.z = 0.07)
    assert T[2, 3] >= 0.07 - 1e-6


# ─── Tests 6-7: candidate count + face convention ───────────────────────────


def test_face_candidate_count_via_builders():
    """Use _build_face_candidates directly to count poses for one face."""
    yaws = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    z_offsets_m = np.array([-0.02, 0.0, 0.02])
    positions_local = np.array([-0.02, 0.0, 0.02])
    poses = _build_face_candidates(
        face_origin_world=np.array([0.0, 0.0, 0.02]),
        approach_dir_world=np.array([0.0, 0.0, 1.0]),
        in_plane_axis_world=np.array([1.0, 0.0, 0.0]),
        positions_local=positions_local,
        yaws=yaws,
        z_offsets_m=z_offsets_m,
        gripper_depth_m=0.10,
    )
    assert poses.shape == (
        len(positions_local) * len(yaws) * len(z_offsets_m),
        4,
        4,
    )
    # Gripper Z (column 2) should be -approach_dir = [0, 0, -1] for every pose.
    z_columns = poses[:, :3, 2]
    expected = np.tile(np.array([0.0, 0.0, -1.0]), (len(poses), 1))
    assert np.allclose(z_columns, expected, atol=1e-5), (
        "Gripper Z must equal -approach_dir for every face candidate"
    )


def test_face_candidate_side_face_approach():
    """A side face approached along +x → gripper Z should be -x for every pose."""
    yaws = np.array([0.0, np.pi / 2])
    z_offsets_m = np.array([0.0])
    positions_local = np.array([0.0])
    poses = _build_face_candidates(
        face_origin_world=np.array([0.05, 0.0, 0.0]),
        approach_dir_world=np.array([1.0, 0.0, 0.0]),  # approach +x
        in_plane_axis_world=np.array([0.0, 1.0, 0.0]),  # along Y
        positions_local=positions_local,
        yaws=yaws,
        z_offsets_m=z_offsets_m,
        gripper_depth_m=0.10,
    )
    z_columns = poses[:, :3, 2]
    # Gripper Z = -approach_dir = [-1, 0, 0]
    assert np.allclose(z_columns, np.tile([-1.0, 0.0, 0.0], (2, 1)), atol=1e-5)


def test_interior_positions_short_axis():
    """A very short axis returns [0.0] instead of an empty array."""
    out = _interior_positions(half=0.005, spacing_m=0.01)
    assert np.array_equal(out, np.array([0.0]))


def test_long_axis_picks_longer_x_or_y():
    long_idx, pos = _long_axis_positions(np.array([0.05, 0.03, 0.02]), spacing_m=0.01)
    assert long_idx == 0
    long_idx, pos = _long_axis_positions(np.array([0.02, 0.05, 0.02]), spacing_m=0.01)
    assert long_idx == 1


# ─── Tests 9-10: gripper geometry lookup ────────────────────────────────────


def test_gripper_geometry_lookup_franka():
    width, depth = _resolve_gripper_geometry("franka_panda")
    assert abs(width - 0.10537486) < 1e-6
    assert depth > 0.0


def test_gripper_geometry_lookup_robotiq_2f_140():
    width, depth = _resolve_gripper_geometry("robotiq_2f_140")
    assert abs(width - 0.13603458) < 1e-6
    assert depth > 0.0


def test_gripper_geometry_lookup_robotiq_2f_85():
    """robotiq_2f_85 now has `width: 0.085` in its YAML (added by this PR)."""
    width, _ = _resolve_gripper_geometry("robotiq_2f_85")
    assert abs(width - 0.085) < 1e-6


def test_gripper_geometry_suction_raises():
    """Suction gripper has no `width` → planner refuses with a clear error."""
    with pytest.raises(ValueError, match="suction|width"):
        _resolve_gripper_geometry("single_suction_cup_30mm")


# ─── Top-down grasp on a single cuboid resting on a table ───────────────────


def test_topdown_cuboid_on_table_obb_geometry():
    """A 4cm x 2cm x 10cm cuboid lying flat on the table (largest face down):
    the OBB recovers its dimensions, sits 1cm above the table, and the
    top-down grasp is centered over the object pointing straight down."""
    pc = make_cuboid_on_table((0.10, 0.04, 0.02), n=3000)
    center, half_extent, R = _compute_obb(pc, mode="advanced")

    # Half-extents (sorted) recover (0.05, 0.02, 0.01) for the 10/4/2 cm box.
    he_sorted = sorted(half_extent.tolist())
    assert np.allclose(he_sorted, [0.01, 0.02, 0.05], atol=5e-3), (
        f"Sorted half_extents {he_sorted} != expected [0.01, 0.02, 0.05]"
    )
    # The box rests on the table: its center sits ~1cm above z=0.
    assert abs(center[2] - 0.01) < 5e-3, f"center.z {center[2]} != ~0.01"

    # Top-down base grasp: gripper Z points straight down, centered in XY,
    # and anchored above the 2cm-tall top face (top_z ~= 0.02).
    T = _world_aligned_top_down_grasp(center, half_extent, R, z_offset=0.0)
    assert np.allclose(T[:3, 2], [0.0, 0.0, -1.0], atol=1e-6)
    assert abs(T[0, 3] - center[0]) < 1e-6 and abs(T[1, 3] - center[1]) < 1e-6
    assert T[2, 3] >= 0.02 - 2e-3, f"grasp z {T[2,3]} not above the 2cm top face"


def test_topdown_cuboid_on_table_extents_fit_gripper():
    """The 4cm width is well under any supported gripper aperture, so the
    object extents are a precondition for the OBB top-down branch not being
    skipped."""
    pc = make_cuboid_on_table((0.10, 0.04, 0.02), n=3000)
    _, half_extent, _ = _compute_obb(pc, mode="advanced")
    full_extent = 2.0 * half_extent
    # The shortest horizontal extent (the 4cm width) is graspable by franka
    # (0.105m) and both robotiq grippers (>=0.085m); skip only triggers when
    # *every* extent exceeds the aperture.
    assert not np.all(full_extent > 0.085), (
        f"extents {full_extent} should not all exceed the gripper aperture"
    )


# ─── Test 8: discriminator scoring with random weights ──────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_score_grasps_with_random_weights():
    """End-to-end plumbing check: random-weights discriminator scores some
    OBB candidates without crashing and returns confidences in [0, 1]."""
    from grasp_gen.models.grasp_gen import GraspGen
    from tests.test_inference_installation import (
        _make_discriminator_cfg,
        _make_generator_cfg,
    )

    device = torch.device("cuda")
    gen_cfg = _make_generator_cfg("pointnet")
    disc_cfg = _make_discriminator_cfg("pointnet")
    model = GraspGen.from_config(gen_cfg, disc_cfg).to(device).eval()

    # Build a fake sampler stub with a `.model` and `.cfg.data.gripper_name`.
    class _StubSampler:
        pass

    stub = _StubSampler()
    stub.model = model
    stub.cfg = DictConfig({"data": {"gripper_name": "franka_panda"}})

    # Make some grasps and a PC.
    pc_np = _make_box_pc((0.06, 0.06, 0.04), n=1024)
    pc_center = pc_np.mean(axis=0).astype(np.float64)
    pc_centered_t = (torch.from_numpy(pc_np) - torch.from_numpy(pc_center.astype(np.float32))).to(device)

    # 16 random grasps near the object surface (world frame).
    rng = np.random.RandomState(0)
    grasps_world = np.tile(np.eye(4), (16, 1, 1)).astype(np.float32)
    grasps_world[:, :3, 3] = rng.uniform(-0.05, 0.05, size=(16, 3)).astype(np.float32)

    scores = _score_grasps_world(grasps_world, pc_centered_t, pc_center, stub)
    assert scores.shape == (16,)
    assert scores.dtype == np.float32
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


# ─── Test 11: real end-to-end smoke with checkpoints (optional) ─────────────


_CKPT_DIR_DEFAULTS = [
    Path("/home/admurali/xgrasp_transfer/GraspGenModels/checkpoints"),
    Path.home() / "xgrasp_transfer/GraspGenModels/checkpoints",
    Path.home() / "GraspGenModels/checkpoints",
]


def _find_ckpt_yml(gripper: str):
    for d in _CKPT_DIR_DEFAULTS:
        candidate = d / f"graspgen_{gripper}.yml"
        if candidate.exists():
            return candidate
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_run_graspmoe_e2e_smoke():
    """If real franka_panda checkpoints are present, run a tiny end-to-end
    GraspMoE pass over a synthetic box PC and assert the contract."""
    ckpt_yml = _find_ckpt_yml("franka_panda")
    if ckpt_yml is None:
        pytest.skip("no franka_panda checkpoint found in standard paths")

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    cfg = load_grasp_cfg(str(ckpt_yml))
    sampler = GraspGenSampler(cfg)

    pc = _make_box_pc((0.06, 0.06, 0.04), n=2000)
    out = run_graspmoe(
        pc,
        sampler,
        grasp_threshold=-1.0,
        num_grasps=64,
        topk_num_grasps=-1,
        num_yaws=8,
        z_offsets_cm=(-2.0, 0.0),
        obb_density="sparse",
    )
    assert set(out.keys()) >= {
        "grasps_diff",
        "scores_diff",
        "grasps_obb",
        "scores_obb",
        "pc_removed",
        "obb",
        "skipped_obb",
    }
    # OBB shouldn't be skipped for a 6cm box vs 0.105m franka aperture.
    assert out["skipped_obb"] is False
    n_obb = len(out["grasps_obb"])
    assert n_obb == 8 * 2, f"Expected 16 OBB candidates (8 yaws x 2 Zs), got {n_obb}"
    if n_obb > 0:
        assert (out["scores_obb"] >= 0.0).all()
        assert (out["scores_obb"] <= 1.0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_run_graspmoe_topdown_cuboid_on_table():
    """End-to-end (checkpoint-gated): run the top-down GraspMoE planner on a
    4cm x 2cm x 10cm cuboid resting on a table and assert it produces a full
    sparse top-down sweep that the discriminator scores in [0, 1]."""
    ckpt_yml = _find_ckpt_yml("franka_panda")
    if ckpt_yml is None:
        pytest.skip("no franka_panda checkpoint found in standard paths")

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    cfg = load_grasp_cfg(str(ckpt_yml))
    sampler = GraspGenSampler(cfg)

    pc = make_cuboid_on_table((0.10, 0.04, 0.02), n=3000)
    num_yaws, z_offsets_cm = 12, (-4.0, -2.0, 0.0)
    out = run_graspmoe(
        pc,
        sampler,
        grasp_threshold=-1.0,
        num_grasps=64,
        topk_num_grasps=-1,
        num_yaws=num_yaws,
        z_offsets_cm=z_offsets_cm,
        obb_density="sparse",
    )
    # 4cm width << franka aperture → top-down branch runs.
    assert out["skipped_obb"] is False
    assert out["obb"] is not None
    # Sparse top-down = num_yaws x len(z_offsets) candidates.
    n_obb = len(out["grasps_obb"])
    assert n_obb == num_yaws * len(z_offsets_cm), (
        f"Expected {num_yaws * len(z_offsets_cm)} OBB candidates, got {n_obb}"
    )
    assert (out["scores_obb"] >= 0.0).all() and (out["scores_obb"] <= 1.0).all()
    # Every OBB grasp approaches top-down (gripper Z ~ -world Z).
    z_cols = out["grasps_obb"][:, :3, 2]
    assert np.allclose(z_cols, np.tile([0.0, 0.0, -1.0], (n_obb, 1)), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
