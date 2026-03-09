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
End-to-end test for the USD grasp pipeline:

1. Convert assets/objects/box.obj to a temporary USD file.
2. Load the USD, sample a point cloud, and generate synthetic grasps.
3. Save those grasps as Xform poses back into the USD file.
4. Re-open the USD and verify the grasp poses and confidences round-trip correctly.
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
import trimesh.transformations as tra

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from convert_obj_to_usd import convert_mesh_to_usd
from save_grasps_to_usd import (
    GRASPS_ROOT_PATH,
    load_grasps_from_usd,
    save_grasps_to_usd,
)


ASSETS_DIR = Path(__file__).parent.parent / "assets" / "objects"
BOX_OBJ = ASSETS_DIR / "box.obj"


@pytest.fixture
def box_obj_path():
    if not BOX_OBJ.exists():
        pytest.skip(f"Test mesh not found at {BOX_OBJ}")
    return str(BOX_OBJ)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def box_usd_path(box_obj_path, tmp_dir):
    """Convert box.obj to box.usd in a temp directory."""
    usd_path = os.path.join(tmp_dir, "box.usd")
    convert_mesh_to_usd(box_obj_path, usd_path, scale=1.0)
    assert os.path.exists(usd_path), "USD conversion failed"
    return usd_path


def _make_synthetic_grasps(n: int = 10, seed: int = 42):
    """Create N random but valid 4x4 grasp transforms and confidences."""
    rng = np.random.RandomState(seed)
    grasps = np.zeros((n, 4, 4))
    for i in range(n):
        angle = rng.uniform(-np.pi, np.pi)
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        R = tra.rotation_matrix(angle, axis)
        T = tra.translation_matrix(rng.uniform(-0.1, 0.1, size=3))
        grasps[i] = T @ R
    confidences = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    return grasps, confidences


# ─── Tests ──────────────────────────────────────────────────────────────────


class TestObjToUsdConversion:
    """Verify OBJ -> USD conversion preserves geometry."""

    def test_usd_file_created(self, box_usd_path):
        assert os.path.exists(box_usd_path)
        assert os.path.getsize(box_usd_path) > 0

    def test_usd_mesh_matches_obj(self, box_obj_path, box_usd_path):
        import scene_synthesizer as synth

        obj_mesh = trimesh.load(box_obj_path)
        usd_asset = synth.Asset(box_usd_path)
        usd_mesh = usd_asset.mesh()

        assert obj_mesh.vertices.shape == usd_mesh.vertices.shape, (
            f"Vertex count mismatch: OBJ {obj_mesh.vertices.shape} vs USD {usd_mesh.vertices.shape}"
        )
        assert obj_mesh.faces.shape == usd_mesh.faces.shape, (
            f"Face count mismatch: OBJ {obj_mesh.faces.shape} vs USD {usd_mesh.faces.shape}"
        )

        np.testing.assert_allclose(
            np.sort(obj_mesh.vertices, axis=0),
            np.sort(usd_mesh.vertices, axis=0),
            atol=1e-5,
            err_msg="Vertex positions don't match between OBJ and USD",
        )


class TestSaveGraspsToUsd:
    """Verify grasps can be written to and read from a USD file."""

    def test_save_and_load_roundtrip(self, box_usd_path):
        grasps, confs = _make_synthetic_grasps(n=15)

        save_grasps_to_usd(box_usd_path, grasps, confs)
        loaded_grasps, loaded_confs = load_grasps_from_usd(box_usd_path)

        assert loaded_grasps.shape == grasps.shape, (
            f"Grasp shape mismatch: expected {grasps.shape}, got {loaded_grasps.shape}"
        )
        assert loaded_confs.shape == confs.shape, (
            f"Confidence shape mismatch: expected {confs.shape}, got {loaded_confs.shape}"
        )

        np.testing.assert_allclose(
            loaded_grasps, grasps, atol=1e-6,
            err_msg="Grasp transforms don't round-trip through USD",
        )
        np.testing.assert_allclose(
            loaded_confs, confs, atol=1e-6,
            err_msg="Grasp confidences don't round-trip through USD",
        )

    def test_grasp_transforms_are_valid_homogeneous(self, box_usd_path):
        grasps, confs = _make_synthetic_grasps(n=5)
        save_grasps_to_usd(box_usd_path, grasps, confs)
        loaded_grasps, _ = load_grasps_from_usd(box_usd_path)

        for i, g in enumerate(loaded_grasps):
            np.testing.assert_allclose(
                g[3, :], [0, 0, 0, 1], atol=1e-7,
                err_msg=f"Grasp {i} has invalid bottom row: {g[3, :]}",
            )
            R = g[:3, :3]
            np.testing.assert_allclose(
                R @ R.T, np.eye(3), atol=1e-5,
                err_msg=f"Grasp {i} rotation is not orthogonal",
            )
            np.testing.assert_allclose(
                np.linalg.det(R), 1.0, atol=1e-5,
                err_msg=f"Grasp {i} rotation determinant != 1",
            )

    def test_overwrite_replaces_old_grasps(self, box_usd_path):
        grasps1, confs1 = _make_synthetic_grasps(n=10, seed=1)
        save_grasps_to_usd(box_usd_path, grasps1, confs1)

        grasps2, confs2 = _make_synthetic_grasps(n=3, seed=2)
        save_grasps_to_usd(box_usd_path, grasps2, confs2)

        loaded, _ = load_grasps_from_usd(box_usd_path)
        assert len(loaded) == 3, (
            f"Expected 3 grasps after overwrite, got {len(loaded)}"
        )

    def test_save_to_separate_output(self, box_usd_path, tmp_dir):
        grasps, confs = _make_synthetic_grasps(n=5)
        output_path = os.path.join(tmp_dir, "box_with_grasps.usd")

        save_grasps_to_usd(box_usd_path, grasps, confs, output_path=output_path)

        assert os.path.exists(output_path)
        loaded, _ = load_grasps_from_usd(output_path)
        assert len(loaded) == 5

        original, _ = load_grasps_from_usd(box_usd_path)
        assert len(original) == 0, "Original USD should be unchanged"


class TestUsdGraspPrimStructure:
    """Verify the USD prim hierarchy is correct when grasps are saved."""

    def test_grasps_root_exists(self, box_usd_path):
        from pxr import Usd

        grasps, confs = _make_synthetic_grasps(n=3)
        save_grasps_to_usd(box_usd_path, grasps, confs)

        stage = Usd.Stage.Open(box_usd_path)
        root_prim = stage.GetPrimAtPath(GRASPS_ROOT_PATH)
        assert root_prim, f"Grasps root prim not found at {GRASPS_ROOT_PATH}"
        assert root_prim.GetTypeName() == "Xform"

    def test_each_grasp_is_xform_with_confidence(self, box_usd_path):
        from pxr import Usd, UsdGeom

        grasps, confs = _make_synthetic_grasps(n=5)
        save_grasps_to_usd(box_usd_path, grasps, confs)

        stage = Usd.Stage.Open(box_usd_path)
        root_prim = stage.GetPrimAtPath(GRASPS_ROOT_PATH)
        children = list(root_prim.GetChildren())
        assert len(children) == 5

        for i, child in enumerate(children):
            assert child.GetTypeName() == "Xform", (
                f"Grasp prim {child.GetPath()} is {child.GetTypeName()}, expected Xform"
            )
            conf_attr = child.GetAttribute("graspgen:confidence")
            assert conf_attr, f"Missing confidence attribute on {child.GetPath()}"
            assert isinstance(conf_attr.Get(), float)

    def test_object_mesh_preserved(self, box_usd_path):
        """Saving grasps should not destroy the existing mesh geometry."""
        import scene_synthesizer as synth

        mesh_before = synth.Asset(box_usd_path).mesh()

        grasps, confs = _make_synthetic_grasps(n=10)
        save_grasps_to_usd(box_usd_path, grasps, confs)

        mesh_after = synth.Asset(box_usd_path).mesh()

        np.testing.assert_allclose(
            np.sort(mesh_before.vertices, axis=0),
            np.sort(mesh_after.vertices, axis=0),
            atol=1e-6,
            err_msg="Mesh geometry was corrupted by grasp saving",
        )


class TestEndToEndPipeline:
    """Full pipeline: OBJ -> USD -> synthetic grasps -> save -> verify."""

    def test_full_pipeline(self, box_obj_path, tmp_dir):
        import scene_synthesizer as synth

        # Step 1: Convert OBJ to USD
        usd_path = os.path.join(tmp_dir, "box_pipeline.usd")
        convert_mesh_to_usd(box_obj_path, usd_path)

        # Step 2: Load mesh from USD and sample point cloud
        asset = synth.Asset(usd_path)
        mesh = asset.mesh()
        xyz, _ = trimesh.sample.sample_surface(mesh, 2000)
        xyz = np.array(xyz)
        assert xyz.shape == (2000, 3)

        # Step 3: Generate synthetic grasps (standing in for real inference)
        grasps, confs = _make_synthetic_grasps(n=20, seed=99)

        # Step 4: Save grasps into the USD
        output_usd = os.path.join(tmp_dir, "box_pipeline_grasps.usd")
        save_grasps_to_usd(usd_path, grasps, confs, output_path=output_usd)

        # Step 5: Re-open and verify everything
        loaded_grasps, loaded_confs = load_grasps_from_usd(output_usd)

        assert loaded_grasps.shape == (20, 4, 4)
        assert loaded_confs.shape == (20,)

        np.testing.assert_allclose(loaded_grasps, grasps, atol=1e-6)
        np.testing.assert_allclose(loaded_confs, confs, atol=1e-6)

        # Verify mesh still intact
        mesh_after = synth.Asset(output_usd).mesh()
        np.testing.assert_allclose(
            np.sort(mesh.vertices, axis=0),
            np.sort(mesh_after.vertices, axis=0),
            atol=1e-5,
        )

        print(
            f"Pipeline OK: {len(loaded_grasps)} grasps saved, "
            f"confidences [{loaded_confs.min():.3f}, {loaded_confs.max():.3f}], "
            f"mesh {mesh_after.vertices.shape[0]} vertices preserved"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
