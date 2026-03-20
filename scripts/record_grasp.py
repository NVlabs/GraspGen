"""
Record grasp rollout data for discriminator finetuning on real-world data.

Usage:
    from scripts.record_grasp import GraspRecorder

    recorder = GraspRecorder("banana", "/data/rollouts/banana")
    recorder.record(point_cloud, grasp_pose, confidence, success, collided)
    recorder.save()

Output:
    rollouts.h5   - grasp poses, confidences, collision flags, point clouds
    grasps.json   - success labels for non-colliding grasps

Note: pass a 7-element discriminator_ratio when fine-tuning:
    discriminator_ratio=[0.35, 0.15, 0.20, 0.05, 0.0, 0.15, 0.10]
The default 5-element ratio will crash because the onpolicy slots (indices 5, 6)
are always accessed by load_discriminator_batch_with_stratified_sampling.
"""

import json
import os
from dataclasses import dataclass
from typing import List

import h5py
import numpy as np


@dataclass
class GraspAttempt:
    point_cloud: np.ndarray   # (N, 3)
    grasp_pose: np.ndarray    # (4, 4)
    confidence: float
    success: bool
    collided: bool


class GraspRecorder:
    """Records grasp attempts for a single object and saves them to disk."""

    def __init__(self, object_id: str, output_dir: str):
        self.object_id = object_id
        self.output_dir = output_dir
        self.attempts: List[GraspAttempt] = []
        os.makedirs(output_dir, exist_ok=True)

    def record(
        self,
        point_cloud: np.ndarray,
        grasp_pose: np.ndarray,
        confidence: float,
        success: bool,
        collided: bool = False,
    ):
        """
        Record a single grasp attempt.

        point_cloud: (N, 3) in object-centered frame (same frame as grasp_pose).
                     Pass exactly what GraspGenSampler received — don't re-center.
        grasp_pose:  (4, 4) SE(3), exactly as returned by GraspGenSampler.run_inference().
        collided:    mark True if the gripper hit something before closing.
                     Collided attempts are excluded from training labels.
        """
        assert grasp_pose.shape == (4, 4), "grasp_pose must be (4, 4)"
        assert point_cloud.ndim == 2 and point_cloud.shape[1] == 3, \
            "point_cloud must be (N, 3)"

        self.attempts.append(GraspAttempt(
            point_cloud=point_cloud.astype(np.float32),
            grasp_pose=grasp_pose.astype(np.float64),
            confidence=float(confidence),
            success=bool(success),
            collided=bool(collided),
        ))

    def save(self):
        """Write all attempts to rollouts.h5 and grasps.json."""
        assert len(self.attempts) > 0, "No attempts recorded yet"

        N = len(self.attempts)
        pred_grasps = np.array([a.grasp_pose for a in self.attempts])
        confidences = np.array([a.confidence for a in self.attempts])
        collisions  = np.array([a.collided   for a in self.attempts])
        successes   = np.array([a.success    for a in self.attempts])

        pc_sizes = [a.point_cloud.shape[0] for a in self.attempts]
        M = max(pc_sizes)
        point_clouds = np.zeros((N, M, 3), dtype=np.float32)
        for i, a in enumerate(self.attempts):
            point_clouds[i, :a.point_cloud.shape[0]] = a.point_cloud

        h5_path = os.path.join(self.output_dir, "rollouts.h5")
        with h5py.File(h5_path, "w") as f:
            grp = f.require_group(f"objects/{self.object_id}")
            grp.create_dataset("pred_grasps",  data=pred_grasps)
            grp.create_dataset("gt_grasps",    data=pred_grasps)  # placeholder
            grp.create_dataset("confidence",   data=confidences)
            grp.create_dataset("collision",    data=collisions)
            grp.create_dataset("point_clouds", data=point_clouds)
            grp.create_dataset("pc_sizes",     data=np.array(pc_sizes))
            grp.create_dataset("asset_path",   data=np.bytes_(self.object_id))

        non_colliding = ~collisions
        json_data = {
            "grasps": {
                "transforms": pred_grasps[non_colliding].tolist(),
                "object_in_gripper": successes[non_colliding].astype(int).tolist(),
            }
        }
        json_path = os.path.join(self.output_dir, "grasps.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        print(f"Saved {N} attempts for '{self.object_id}' to {self.output_dir}")
        print(f"  Collided: {collisions.sum()}, Successful: {successes.sum()}")

    def write_cache(self, cache_h5_path: str, num_renderings: int = 5) -> None:
        """
        Write to GraspGenDatasetCache format so train_graspgen.py can consume
        this data directly with preload_dataset=True — no mesh needed.

        Place the output at <cache_dir>/<dataset_name>/cache_train_<prefix>.h5
        (prepare_finetune.py handles this automatically).
        """
        assert len(self.attempts) > 0, "No attempts recorded"
        import trimesh.transformations as tra
        from grasp_gen.dataset.eval_utils import write_info

        non_colliding = [a for a in self.attempts if not a.collided]
        assert len(non_colliding) > 0, "All attempts collided"

        pos_attempts = [a for a in non_colliding if a.success]
        neg_attempts = [a for a in non_colliding if not a.success]

        positive_grasps = (
            np.array([a.grasp_pose for a in pos_attempts], dtype=np.float64)
            if pos_attempts else np.zeros((0, 4, 4), dtype=np.float64)
        )
        negative_grasps = (
            np.array([a.grasp_pose for a in neg_attempts], dtype=np.float64)
            if neg_attempts else np.zeros((0, 4, 4), dtype=np.float64)
        )
        assert len(positive_grasps) > 0, "No successful grasps to write"

        n = min(num_renderings, len(self.attempts))
        idxs = np.random.choice(len(self.attempts), size=n, replace=False)
        renderings = []
        for idx in idxs:
            xyz = self.attempts[idx].point_cloud.astype(np.float32)
            T = tra.translation_matrix(-xyz.mean(axis=0))
            xyz_centered = tra.transform_points(xyz, T).astype(np.float32)
            renderings.append({
                "mesh_mode":          np.bool_(False),
                "load_contact_batch": np.bool_(False),
                "invalid":            np.bool_(False),
                "points":             xyz_centered,
                "T_move_to_pc_mean":  T.astype(np.float32),
                "positive_grasps":    positive_grasps,
            })

        grasp_data = {
            "object_mesh":              None,
            "positive_grasps":          positive_grasps,
            "contacts":                 None,
            "object_asset_path":        self.object_id,
            "object_scale":             float(1.0),
            "negative_grasps":          negative_grasps,
            "positive_grasps_onpolicy": positive_grasps,
            "negative_grasps_onpolicy": negative_grasps,
        }

        key_h5 = self.object_id.replace("/", "____")
        with h5py.File(cache_h5_path, "a") as f:
            if key_h5 in f:
                del f[key_h5]
            grp = f.create_group(key_h5)
            write_info(grp.create_group("grasp_data"), grasp_data)
            grp_r = grp.create_group("renderings")
            for i, rendering in enumerate(renderings):
                write_info(grp_r.create_group(str(i)), rendering)

        print(f"Wrote cache for '{self.object_id}' → {cache_h5_path}")
        print(f"  Positives: {len(positive_grasps)}, Negatives: {len(negative_grasps)}, Renderings: {n}")

    @classmethod
    def from_h5(cls, h5_path: str, object_id: str, output_dir: str) -> "GraspRecorder":
        """Reconstruct a GraspRecorder from a saved rollouts.h5 + grasps.json."""
        import h5py as _h5py
        recorder = cls(object_id=object_id, output_dir=output_dir)
        with _h5py.File(h5_path, "r") as f:
            grp          = f["objects"][object_id]
            pred_grasps  = grp["pred_grasps"][...]
            confidences  = grp["confidence"][...]
            collisions   = grp["collision"][...]
            point_clouds = grp["point_clouds"][...]
            pc_sizes     = grp["pc_sizes"][...] if "pc_sizes" in grp else None

        import json as _json
        json_path = os.path.join(output_dir, "grasps.json")
        with open(json_path) as jf:
            data = _json.load(jf)
        success_labels = np.array(data["grasps"]["object_in_gripper"])

        non_colliding_idx = np.where(~collisions)[0]
        successes = np.zeros(len(pred_grasps), dtype=bool)
        successes[non_colliding_idx] = success_labels.astype(bool)

        for i in range(len(pred_grasps)):
            pc = point_clouds[i]
            if pc_sizes is not None:
                pc = pc[:pc_sizes[i]]
            recorder.attempts.append(GraspAttempt(
                point_cloud=pc.astype(np.float32),
                grasp_pose=pred_grasps[i].astype(np.float64),
                confidence=float(confidences[i]),
                success=bool(successes[i]),
                collided=bool(collisions[i]),
            ))
        return recorder

    def __len__(self):
        return len(self.attempts)
