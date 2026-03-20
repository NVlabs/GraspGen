"""
Convert real-world rollout data into a training cache and print the fine-tuning command.

Run this after collecting rollouts with GraspRecorder:

    python scripts/prepare_finetune.py \\
        --rollouts   /data/rollouts/banana \\
        --cache_dir  /data/cache \\
        --dataset_dir /data/dataset \\
        --object_id  banana.obj
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts",    required=True)
    p.add_argument("--cache_dir",   required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--object_id",   required=True)
    p.add_argument("--num_renderings", type=int, default=5)
    p.add_argument("--gripper", default="robotiq_2f_140")
    return p.parse_args()


def main():
    args = parse_args()

    h5_path   = os.path.join(args.rollouts, "rollouts.h5")
    json_path = os.path.join(args.rollouts, "grasps.json")
    train_txt = os.path.join(args.dataset_dir, "train.txt")

    assert os.path.exists(h5_path),   f"rollouts.h5 not found: {h5_path}"
    assert os.path.exists(json_path), f"grasps.json not found: {json_path}"

    import h5py
    import numpy as np
    with h5py.File(h5_path, "r") as f:
        grp       = f["objects"][args.object_id]
        n_total   = grp["pred_grasps"].shape[0]
        n_collided = int(grp["collision"][...].sum())
    with open(json_path) as jf:
        labels = json.load(jf)["grasps"]["object_in_gripper"]
    n_success = sum(labels)
    n_noncolliding = n_total - n_collided

    print(f"  {args.object_id}: {n_total} attempts, "
          f"{n_success} success, {n_noncolliding - n_success} fail, "
          f"{n_collided} collided")

    if n_success == 0:
        print("No successful grasps — need at least 1 to build a useful cache.")
        sys.exit(1)

    if n_noncolliding < 10:
        print(f"Only {n_noncolliding} non-colliding attempts — signal may be weak. "
              f"Recommend collecting at least 10.")

    # Validate that the ratio we're about to print is the right length.
    # train_graspgen.py accesses onpolicy slots (indices 5, 6) unconditionally
    # when load_discriminator_dataset=True; a 5-element ratio crashes there.
    _ratio = [0.30, 0.10, 0.15, 0.05, 0.0, 0.20, 0.20]
    assert len(_ratio) == 7, "discriminator_ratio must have 7 elements for onpolicy data"

    from scripts.record_grasp import GraspRecorder
    from grasp_gen.dataset.dataset import get_cache_prefix

    dataset_name = os.path.basename(os.path.abspath(args.dataset_dir))
    cache_subdir = os.path.join(args.cache_dir, dataset_name)
    os.makedirs(cache_subdir, exist_ok=True)

    prefix   = get_cache_prefix(prob_point_cloud=0.0, load_discriminator_dataset=True)
    cache_h5 = os.path.join(cache_subdir, f"cache_train_{prefix}.h5")

    recorder = GraspRecorder.from_h5(h5_path, args.object_id, args.rollouts)
    recorder.write_cache(cache_h5, num_renderings=args.num_renderings)

    os.makedirs(args.dataset_dir, exist_ok=True)
    existing = set()
    if os.path.exists(train_txt):
        with open(train_txt) as f:
            existing = set(f.read().splitlines())
    if args.object_id not in existing:
        with open(train_txt, "a") as f:
            f.write(args.object_id + "\n")

    ckpt_root = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

    print(f"""
Run this to fine-tune:

    python scripts/train_graspgen.py \\
        data.root_dir={args.dataset_dir} \\
        data.cache_dir={args.cache_dir} \\
        data.gripper_name={args.gripper} \\
        data.load_discriminator_dataset=true \\
        data.discriminator_ratio='[0.30,0.10,0.15,0.05,0.0,0.20,0.20]' \\
        train.model_name=discriminator \\
        train.num_epochs=5 \\
        train.checkpoint={ckpt_root}/graspgen_{args.gripper}_dis.pth \\
        discriminator.checkpoint={ckpt_root}/graspgen_{args.gripper}_dis.pth \\
        discriminator.gripper_name={args.gripper}

Loss should drop within the first epoch. If it doesn't move, collect more data.
""")


if __name__ == "__main__":
    main()
