"""
Verify one end-to-end discriminator training step with onpolicy data.

No mocks. Constructs a minimal dataset, loads one batch, runs one
forward + backward pass through the discriminator, checks loss is finite.

Run on the GCP VM inside Docker:
    docker run --rm --gpus all \\
        -v /opt/GraspGen:/code \\
        graspgen:latest \\
        bash -c "pip install -q -e /code --no-deps 2>/dev/null && \\
                 pip install -q viser 2>/dev/null && \\
                 python /code/scripts/test_training_step.py"
"""

import json
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


OBJECT_KEY   = "banana.obj"
NUM_POINTS   = 512
NUM_GRASPS   = 20
GRIPPER_NAME = "robotiq_2f_140"
GRIPPER_DEPTH = 0.195   # from config/grippers/robotiq_2f_140.yaml
CKPT_PATH    = "/code/checkpoints/graspgen_robotiq_2f_140_dis.pth"  # mount checkpoints at /code/checkpoints inside Docker


def make_grasp(z_offset=0.0):
    """Grasp whose tool tip lands near origin (inside the point cloud)."""
    T = np.eye(4)
    T[2, 3] = -GRIPPER_DEPTH + z_offset
    return T.tolist()


def setup_dataset(root):
    from scripts.record_grasp import GraspRecorder

    object_dir  = os.path.join(root, "objects")
    grasp_dir   = os.path.join(root, "grasps")
    rollout_dir = os.path.join(root, "onpolicy")
    dataset_dir = os.path.join(root, "dataset")
    cache_dir   = os.path.join(root, "cache")
    for d in [object_dir, grasp_dir, rollout_dir, dataset_dir, cache_dir]:
        os.makedirs(d, exist_ok=True)

    # Object mesh
    banana_mesh = os.path.join(os.path.dirname(__file__), "..", "assets", "objects", "banana.obj")
    banana_mesh = os.path.abspath(banana_mesh)
    assert os.path.exists(banana_mesh), f"Mesh not found: {banana_mesh}"
    os.symlink(banana_mesh, os.path.join(object_dir, OBJECT_KEY))

    # Grasp JSON
    rng = np.random.default_rng(0)
    grasps_data = {
        "object": {"file": OBJECT_KEY, "scale": 1.0},
        "grasps": {
            "transforms": [make_grasp(rng.uniform(-0.01, 0.01)) for _ in range(NUM_GRASPS)],
            "object_in_gripper": [i % 2 for i in range(NUM_GRASPS)],
        },
    }
    with open(os.path.join(grasp_dir, "banana_grasps.json"), "w") as f:
        json.dump(grasps_data, f)
    with open(os.path.join(grasp_dir, "map_uuid_to_path.json"), "w") as f:
        json.dump({OBJECT_KEY: "banana_grasps.json"}, f)

    # Onpolicy rollouts
    recorder = GraspRecorder(object_id=OBJECT_KEY, output_dir=rollout_dir)
    for i in range(8):
        pc = rng.standard_normal((512, 3)).astype(np.float32) * 0.05
        T  = np.eye(4); T[2, 3] = -GRIPPER_DEPTH
        recorder.record(pc, T, confidence=0.8, success=(i % 2 == 0), collided=False)
    recorder.save()
    h5_path   = os.path.join(rollout_dir, "rollouts.h5")
    json_path = os.path.join(rollout_dir, "grasps.json")

    # Cache JSON so constructor skips UUID scan
    real_cache_dir = os.path.join(cache_dir, os.path.basename(dataset_dir))
    os.makedirs(real_cache_dir, exist_ok=True)
    cache_json = os.path.join(real_cache_dir, f"{os.path.basename(rollout_dir)}.json")
    with open(cache_json, "w") as f:
        json.dump({OBJECT_KEY: json_path}, f)

    # train.txt
    with open(os.path.join(dataset_dir, "train.txt"), "w") as f:
        f.write(OBJECT_KEY + "\n")

    return object_dir, grasp_dir, rollout_dir, dataset_dir, cache_dir, h5_path


def run():
    import torch
    from grasp_gen.dataset.dataset import ObjectPickDataset
    from grasp_gen.models.discriminator import GraspGenDiscriminator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    with tempfile.TemporaryDirectory() as root:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        object_dir, grasp_dir, rollout_dir, dataset_dir, cache_dir, h5_path = setup_dataset(root)

        # ── Dataset ──────────────────────────────────────────────────────────
        print("\n[1] Building dataset...")
        dataset = ObjectPickDataset(
            root_dir=dataset_dir,
            cache_dir=cache_dir,
            split="train",
            tasks=[],
            num_points=NUM_POINTS,
            num_obj_points=NUM_POINTS,
            cam_coord=False,
            num_rotations=1,
            grid_res=0.005,
            jitter_scale=0.0,
            contact_radius=0.005,
            dist_above_table=0.0,
            offset_bins=None,
            robot_prob=0.0,
            random_seed=42,
            object_root_dir=object_dir,
            grasp_root_dir=grasp_dir,
            dataset_version="v2",
            gripper_name=GRIPPER_NAME,
            num_grasps_per_object=NUM_GRASPS,
            load_discriminator_dataset=True,
            prob_point_cloud=0.0,   # mesh_mode=True avoids pyrender (no display needed)
            onpolicy_dataset_h5_path=h5_path,
            onpolicy_dataset_dir=rollout_dir,
            discriminator_ratio=[0.30, 0.10, 0.15, 0.05, 0.0, 0.20, 0.20],
        )
        print(f"    {len(dataset)} items")

        # ── Load one batch ────────────────────────────────────────────────────
        print("\n[2] Loading one batch...")
        item = dataset[0]
        assert not item.get("invalid", False), "batch returned invalid"
        print(f"    batch keys: {sorted(item.keys())}")

        # ── Load discriminator model ──────────────────────────────────────────
        print("\n[3] Loading discriminator...")
        assert os.path.exists(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"

        from omegaconf import OmegaConf
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        cfg = OmegaConf.load(cfg_path)
        model = GraspGenDiscriminator.from_config(cfg.discriminator).to(device)

        ckpt = torch.load(CKPT_PATH, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.train()
        print(f"    loaded from {CKPT_PATH}")

        # ── Forward + backward pass ───────────────────────────────────────────
        print("\n[4] Forward + backward pass...")
        from grasp_gen.utils.train_utils import to_gpu
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        # Mimic collate_batch_keys from dataset.py for a single-item batch:
        # - "points", "inputs", "seg", "cam_pose" etc. → stacked tensor [1, ...]
        # - "grasps", "labels", "grasp_ids" etc.      → list of tensors [tensor]
        STACKED_KEYS = {"inputs", "points", "seg", "object_inputs", "bottom_center",
                        "cam_pose", "ee_pose", "placement_masks", "placement_region"}
        LIST_KEYS    = {"grasps", "labels", "grasp_ids", "positive_grasps",
                        "negative_grasps", "grasps_ground_truth", "grasps_highres"}

        data = {}
        for k, v in item.items():
            if not torch.is_tensor(v):
                data[k] = v
            elif k in STACKED_KEYS:
                data[k] = v.unsqueeze(0)   # [1, ...]
            elif k in LIST_KEYS:
                data[k] = [v]              # list of 1 tensor, as collate produces
            else:
                data[k] = v.unsqueeze(0)   # default: stack
        to_gpu(data)

        optimizer.zero_grad()
        outputs, losses, stats = model(data, None)
        loss = sum(w * v for w, v in losses.values())
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        print(f"\n[PASS] One training step completed")
        print(f"       loss        = {loss.item():.6f}")
        print(f"       grasp_ids   = {item['grasp_ids'].unique().tolist()}")

    print("\n" + "=" * 60)
    print("TRAINING STEP TEST PASSED")
    print("=" * 60)


def setup_meshless_dataset(root):
    """Minimal dataset with NO mesh — uses GraspRecorder.write_cache() directly."""
    from scripts.record_grasp import GraspRecorder
    from grasp_gen.dataset.dataset import get_cache_prefix

    MESHLESS_KEY = "new_real_object.obj"

    dataset_dir = os.path.join(root, "dataset")
    cache_dir   = os.path.join(root, "cache")
    rollout_dir = os.path.join(root, "onpolicy")
    for d in [dataset_dir, cache_dir, rollout_dir]:
        os.makedirs(d, exist_ok=True)

    # Record rollouts with stored point clouds
    rng = np.random.default_rng(1)
    recorder = GraspRecorder(object_id=MESHLESS_KEY, output_dir=rollout_dir)
    for i in range(8):
        pc = rng.standard_normal((512, 3)).astype(np.float32) * 0.05
        T  = np.eye(4); T[2, 3] = -GRIPPER_DEPTH
        recorder.record(pc, T, confidence=0.8, success=(i % 2 == 0), collided=False)
    recorder.save()

    # Write directly to GraspGenDatasetCache format so preload_dataset=True works
    real_cache_dir = os.path.join(cache_dir, os.path.basename(dataset_dir))
    os.makedirs(real_cache_dir, exist_ok=True)
    prefix = get_cache_prefix(prob_point_cloud=0.0, load_discriminator_dataset=True)
    cache_h5 = os.path.join(real_cache_dir, f"cache_train_{prefix}.h5")
    recorder.write_cache(cache_h5, num_renderings=5)

    with open(os.path.join(dataset_dir, "train.txt"), "w") as f:
        f.write(MESHLESS_KEY + "\n")

    return dataset_dir, cache_dir, cache_h5, MESHLESS_KEY


def run_meshless():
    """One training step for a meshless real-world object using preload_dataset=True."""
    import torch
    from grasp_gen.dataset.dataset import ObjectPickDataset
    from grasp_gen.models.discriminator import GraspGenDiscriminator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    with tempfile.TemporaryDirectory() as root:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        dataset_dir, cache_dir, cache_h5, key = setup_meshless_dataset(root)

        # Dummy dirs — not used since preload_dataset=True reads from cache
        dummy_obj_dir   = os.path.join(root, "objects")
        dummy_grasp_dir = os.path.join(root, "grasps")
        os.makedirs(dummy_obj_dir,   exist_ok=True)
        os.makedirs(dummy_grasp_dir, exist_ok=True)
        with open(os.path.join(dummy_grasp_dir, "map_uuid_to_path.json"), "w") as f:
            json.dump({}, f)

        print("\n[1] Building meshless dataset (preload_dataset=True)...")
        dataset = ObjectPickDataset(
            root_dir=dataset_dir,
            cache_dir=cache_dir,
            split="train",
            tasks=[],
            num_points=NUM_POINTS,
            num_obj_points=NUM_POINTS,
            cam_coord=False,
            num_rotations=1,
            grid_res=0.005,
            jitter_scale=0.0,
            contact_radius=0.005,
            dist_above_table=0.0,
            offset_bins=None,
            robot_prob=0.0,
            random_seed=42,
            object_root_dir=dummy_obj_dir,
            grasp_root_dir=dummy_grasp_dir,
            dataset_version="v2",
            gripper_name=GRIPPER_NAME,
            num_grasps_per_object=NUM_GRASPS,
            load_discriminator_dataset=True,
            prob_point_cloud=0.0,
            preload_dataset=True,          # production default
            discriminator_ratio=[0.30, 0.10, 0.15, 0.05, 0.0, 0.20, 0.20],
        )
        print(f"    {len(dataset)} items in cache")

        print("\n[2] Loading one batch (preload_dataset=True, meshless)...")
        item = dataset[0]
        assert not item.get("invalid", False), "batch returned invalid"
        print(f"    batch keys: {sorted(item.keys())}")

        print("\n[3] Loading discriminator...")
        assert os.path.exists(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"
        from omegaconf import OmegaConf
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        cfg = OmegaConf.load(cfg_path)
        model = GraspGenDiscriminator.from_config(cfg.discriminator).to(device)
        ckpt = torch.load(CKPT_PATH, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.train()

        print("\n[4] Forward + backward pass (meshless, preload_dataset=True)...")
        from grasp_gen.utils.train_utils import to_gpu
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        STACKED_KEYS = {"inputs", "points", "seg", "object_inputs", "bottom_center",
                        "cam_pose", "ee_pose", "placement_masks", "placement_region"}
        LIST_KEYS    = {"grasps", "labels", "grasp_ids", "positive_grasps",
                        "negative_grasps", "grasps_ground_truth", "grasps_highres"}

        data = {}
        for k, v in item.items():
            if not torch.is_tensor(v):
                data[k] = v
            elif k in STACKED_KEYS:
                data[k] = v.unsqueeze(0)
            elif k in LIST_KEYS:
                data[k] = [v]
            else:
                data[k] = v.unsqueeze(0)
        to_gpu(data)

        optimizer.zero_grad()
        outputs, losses, stats = model(data, None)
        loss = sum(w * v for w, v in losses.values())
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        print(f"\n[PASS] Meshless training step completed (preload_dataset=True)")
        print(f"       loss        = {loss.item():.6f}")
        print(f"       grasp_ids   = {item['grasp_ids'].unique().tolist()}")

    print("\n" + "=" * 60)
    print("MESHLESS TRAINING STEP TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run()
    run_meshless()
