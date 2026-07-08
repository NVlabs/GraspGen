"""Tests for GraspRecorder (scripts/record_grasp.py)."""

import json
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.record_grasp import GraspRecorder


def make_random_grasp():
    T = np.eye(4)
    T[:3, 3] = np.random.randn(3) * 0.1
    Q, _ = np.linalg.qr(np.random.randn(3, 3))
    T[:3, :3] = Q
    return T


def make_random_point_cloud(n=512):
    return np.random.randn(n, 3).astype(np.float32) * 0.1


def test_save_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.9, success=True)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.3, success=False)
        rec.save()
        assert os.path.exists(os.path.join(tmpdir, "rollouts.h5"))
        assert os.path.exists(os.path.join(tmpdir, "grasps.json"))


def test_h5_fields_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.9, success=True)
        rec.save()
        with h5py.File(os.path.join(tmpdir, "rollouts.h5"), "r") as f:
            grp = f["objects/banana"]
            for field in ["pred_grasps", "gt_grasps", "confidence", "collision", "point_clouds"]:
                assert field in grp


def test_h5_shapes():
    N = 5
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        for _ in range(N):
            rec.record(make_random_point_cloud(512), make_random_grasp(), 0.5, success=True)
        rec.save()
        with h5py.File(os.path.join(tmpdir, "rollouts.h5"), "r") as f:
            grp = f["objects/banana"]
            assert grp["pred_grasps"].shape  == (N, 4, 4)
            assert grp["confidence"].shape   == (N,)
            assert grp["point_clouds"].shape == (N, 512, 3)


def test_critical_constraint_no_collisions():
    """len(json transforms) == sum(~collision) — required by load_onpolicy_dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.9, success=True,  collided=False)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.8, success=False, collided=False)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.2, success=False, collided=True)
        rec.save()

        with h5py.File(os.path.join(tmpdir, "rollouts.h5"), "r") as f:
            n_noncolliding = int(np.logical_not(f["objects/banana"]["collision"][...]).sum())
        with open(os.path.join(tmpdir, "grasps.json")) as f:
            n_transforms = len(json.load(f)["grasps"]["transforms"])

        assert n_noncolliding == n_transforms


def test_success_labels_correct():
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.9, success=True,  collided=False)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.5, success=False, collided=False)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.1, success=False, collided=True)
        rec.save()
        with open(os.path.join(tmpdir, "grasps.json")) as f:
            labels = json.load(f)["grasps"]["object_in_gripper"]
        assert labels == [1, 0]


def test_empty_recorder_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(AssertionError):
            GraspRecorder("banana", tmpdir).save()


def test_wrong_grasp_shape_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(AssertionError):
            GraspRecorder("banana", tmpdir).record(
                make_random_point_cloud(), np.eye(3), 0.5, success=True)


def test_wrong_point_cloud_shape_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(AssertionError):
            GraspRecorder("banana", tmpdir).record(
                np.random.randn(512), make_random_grasp(), 0.5, success=True)


def test_load_onpolicy_dataset_logic():
    """Round-trip: save then manually run load_onpolicy_dataset() logic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        for _ in range(3):
            rec.record(make_random_point_cloud(), make_random_grasp(), 0.9, success=True,  collided=False)
        for _ in range(2):
            rec.record(make_random_point_cloud(), make_random_grasp(), 0.3, success=False, collided=False)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.1, success=False, collided=True)
        rec.save()

        h5 = h5py.File(os.path.join(tmpdir, "rollouts.h5"), "r")
        h5_obj = h5["objects"]["banana"]
        pred_grasps = h5_obj["pred_grasps"][...]
        scores      = h5_obj["confidence"][...]
        collision   = h5_obj["collision"][...]
        mask_not_colliding = np.logical_not(collision)

        data = json.load(open(os.path.join(tmpdir, "grasps.json"), "rb"))
        assert mask_not_colliding.sum() == len(data["grasps"]["transforms"])

        mask_eval_success = np.array(data["grasps"]["object_in_gripper"])
        success_result = np.zeros(len(scores))
        success_result[np.where(mask_not_colliding)[0][np.where(mask_eval_success)[0]]] = 1.0

        positive_grasps = pred_grasps[success_result.astype(np.bool_)]
        negative_grasps = pred_grasps[~success_result.astype(np.bool_)]

        assert positive_grasps.shape == (3, 4, 4)
        assert negative_grasps.shape == (3, 4, 4)


def test_write_cache_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.9, success=True)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.3, success=False)
        rec.save()
        cache_path = os.path.join(tmpdir, "cache.h5")
        rec.write_cache(cache_path, num_renderings=2)
        assert os.path.exists(cache_path)


def test_write_cache_loadable():
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        for _ in range(4):
            rec.record(make_random_point_cloud(), make_random_grasp(), 0.9, success=True)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.3, success=False)
        rec.save()
        cache_path = os.path.join(tmpdir, "cache.h5")
        rec.write_cache(cache_path, num_renderings=3)

        from grasp_gen.dataset.dataset_utils import GraspGenDatasetCache
        cache = GraspGenDatasetCache.load_from_h5_file(cache_path)
        assert "banana" in cache
        _, renderings = cache["banana"]
        assert len(renderings) == 3


def test_write_cache_shapes():
    N_POINTS, N_POS, N_NEG, N_RENDERINGS = 256, 3, 2, 4
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("mug", tmpdir)
        for _ in range(N_POS):
            rec.record(make_random_point_cloud(N_POINTS), make_random_grasp(), 0.9, success=True)
        for _ in range(N_NEG):
            rec.record(make_random_point_cloud(N_POINTS), make_random_grasp(), 0.2, success=False)
        rec.save()
        cache_path = os.path.join(tmpdir, "cache.h5")
        rec.write_cache(cache_path, num_renderings=N_RENDERINGS)

        from grasp_gen.dataset.dataset_utils import GraspGenDatasetCache
        gd, renderings = GraspGenDatasetCache.load_from_h5_file(cache_path)["mug"]
        assert gd.positive_grasps.shape == (N_POS, 4, 4)
        assert gd.negative_grasps.shape == (N_NEG, 4, 4)
        assert gd.positive_grasps_onpolicy.shape == (N_POS, 4, 4)
        assert gd.negative_grasps_onpolicy.shape == (N_NEG, 4, 4)
        assert len(renderings) == N_RENDERINGS
        for r in renderings:
            assert r["points"].shape == (N_POINTS, 3)
            assert r["positive_grasps"].shape == (N_POS, 4, 4)


def test_write_cache_point_clouds_centered():
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("apple", tmpdir)
        for _ in range(3):
            rec.record(make_random_point_cloud(), make_random_grasp(), 0.8, success=True)
        rec.save()
        cache_path = os.path.join(tmpdir, "cache.h5")
        rec.write_cache(cache_path, num_renderings=3)

        from grasp_gen.dataset.dataset_utils import GraspGenDatasetCache
        _, renderings = GraspGenDatasetCache.load_from_h5_file(cache_path)["apple"]
        for r in renderings:
            np.testing.assert_allclose(r["points"].mean(axis=0), 0.0, atol=1e-4)


def test_from_h5_round_trip_attempts():
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        for i in range(6):
            rec.record(make_random_point_cloud(), make_random_grasp(),
                       0.9, success=(i % 2 == 0), collided=(i == 5))
        rec.save()
        restored = GraspRecorder.from_h5(os.path.join(tmpdir, "rollouts.h5"), "banana", tmpdir)
        assert len(restored) == len(rec)


def test_from_h5_round_trip_success_labels():
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.9, success=True,  collided=False)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.5, success=False, collided=False)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.1, success=False, collided=True)
        rec.save()
        restored = GraspRecorder.from_h5(os.path.join(tmpdir, "rollouts.h5"), "banana", tmpdir)
        assert restored.attempts[0].success  is True
        assert restored.attempts[0].collided is False
        assert restored.attempts[1].success  is False
        assert restored.attempts[2].collided is True


def test_from_h5_write_cache_matches_direct():
    """from_h5 + write_cache should produce same shapes as direct write_cache."""
    from grasp_gen.dataset.dataset_utils import GraspGenDatasetCache
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        for i in range(5):
            rec.record(make_random_point_cloud(256), make_random_grasp(),
                       0.9, success=(i < 3), collided=False)
        rec.save()

        cache_direct = os.path.join(tmpdir, "cache_direct.h5")
        rec.write_cache(cache_direct, num_renderings=3)

        restored = GraspRecorder.from_h5(os.path.join(tmpdir, "rollouts.h5"), "banana", tmpdir)
        cache_restored = os.path.join(tmpdir, "cache_restored.h5")
        restored.write_cache(cache_restored, num_renderings=3)

        gd_d, rd_d = GraspGenDatasetCache.load_from_h5_file(cache_direct)["banana"]
        gd_r, rd_r = GraspGenDatasetCache.load_from_h5_file(cache_restored)["banana"]

        assert gd_d.positive_grasps.shape == gd_r.positive_grasps.shape
        assert gd_d.negative_grasps.shape == gd_r.negative_grasps.shape
        assert len(rd_d) == len(rd_r)


def test_write_cache_two_objects_same_file():
    """Two objects written to the same cache file should both be readable."""
    from grasp_gen.dataset.dataset_utils import GraspGenDatasetCache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "cache.h5")
        for obj_id, n_pos, n_neg in [("apple", 3, 2), ("mug", 4, 1)]:
            obj_dir = os.path.join(tmpdir, obj_id)
            rec = GraspRecorder(obj_id, obj_dir)
            for _ in range(n_pos):
                rec.record(make_random_point_cloud(), make_random_grasp(), 0.9, success=True)
            for _ in range(n_neg):
                rec.record(make_random_point_cloud(), make_random_grasp(), 0.2, success=False)
            rec.save()
            rec.write_cache(cache_path, num_renderings=2)

        cache = GraspGenDatasetCache.load_from_h5_file(cache_path)
        assert "apple" in cache
        assert "mug"   in cache
        assert cache["apple"][0].positive_grasps.shape[0] == 3
        assert cache["mug"][0].positive_grasps.shape[0]   == 4


def test_write_cache_multi_object_shapes_independent():
    """Point cloud shapes for different objects don't bleed into each other."""
    from grasp_gen.dataset.dataset_utils import GraspGenDatasetCache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "cache.h5")
        for obj_id, n_pts in [("small_obj", 128), ("large_obj", 512)]:
            obj_dir = os.path.join(tmpdir, obj_id)
            rec = GraspRecorder(obj_id, obj_dir)
            rec.record(make_random_point_cloud(n_pts), make_random_grasp(), 0.9, success=True)
            rec.record(make_random_point_cloud(n_pts), make_random_grasp(), 0.2, success=False)
            rec.save()
            rec.write_cache(cache_path, num_renderings=1)

        cache = GraspGenDatasetCache.load_from_h5_file(cache_path)
        assert cache["small_obj"][1][0]["points"].shape == (128, 3)
        assert cache["large_obj"][1][0]["points"].shape == (512, 3)


def test_prepare_finetune_prints_valid_config_keys():
    """prepare_finetune.py output must use config keys that actually exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        for i in range(6):
            rec.record(make_random_point_cloud(), make_random_grasp(),
                       0.9, success=(i < 3), collided=False)
        rec.save()

        dataset_dir = os.path.join(tmpdir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        with open(os.path.join(dataset_dir, "train.txt"), "w") as f:
            f.write("banana\n")

        result = subprocess.run(
            [sys.executable,
             os.path.join(os.path.dirname(__file__), "..", "scripts", "prepare_finetune.py"),
             "--rollouts",    tmpdir,
             "--cache_dir",   os.path.join(tmpdir, "cache"),
             "--dataset_dir", dataset_dir,
             "--object_id",   "banana"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "train.model_name=discriminator"      in out
        assert "data.load_discriminator_dataset=true" in out
        assert "data.discriminator_ratio="            in out
        assert "discriminator.checkpoint="            in out
        assert "train.checkpoint="                    in out


def test_asset_path_written_to_h5():
    """asset_path must be in HDF5 — load_onpolicy_dataset() reads it to build UUID mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = GraspRecorder("banana", tmpdir)
        rec.record(make_random_point_cloud(), make_random_grasp(), 0.9, success=True)
        rec.save()
        with h5py.File(os.path.join(tmpdir, "rollouts.h5"), "r") as f:
            grp = f["objects"]["banana"]
            assert "asset_path" in grp
            assert grp["asset_path"][...].item().decode("utf-8") == "banana"
