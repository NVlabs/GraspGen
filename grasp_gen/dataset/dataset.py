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
Data loader for training grasp models.
"""
import glob
import json
import logging
import os
import random
import time
from typing import Tuple, Union

import h5py
import numpy as np

try:
    import pickle5 as pickle
except:
    import pickle
import torch
import trimesh
import trimesh.transformations as tra
from omegaconf import DictConfig
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset
from tqdm import tqdm

from grasp_gen.dataset.dataset_utils import (
    GraspGenDatasetCache,
    ObjectGraspDataset,
    dump_object_list,
    filter_grasps_by_point_cloud_visibility,
    load_from_json,
    load_object_grasp_data,
    GraspJsonDatasetReader,
    get_rotation_augmentation,
)
from grasp_gen.dataset.eval_utils import check_collision
from grasp_gen.dataset.exceptions import DataLoaderError

# NOTE: render_pc is imported lazily where needed to avoid forcing pyrender import in headless environments
from grasp_gen.dataset.visualize_utils import (
    MAPPING_ID2NAME,
    MAPPING_NAME2ID,
    visualize_discriminator_dataset,
    visualize_generator_dataset,
)
from grasp_gen.dataset.webdataset_utils import GraspWebDatasetReader, is_webdataset
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_pointcloud,
    visualize_grasp,
)
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


def get_cache_path(cache_dir: str, root_dir: str) -> str:
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    cache_name = os.path.basename(root_dir)
    cache_path = os.path.join(cache_dir, cache_name)
    os.makedirs(cache_path, exist_ok=True)
    return cache_path


def get_cache_prefix(prob_point_cloud: float, load_discriminator_dataset: bool) -> str:
    cache_name = get_pc_setting_name(prob_point_cloud)
    prefix = "dis" if load_discriminator_dataset else "gen"
    cache_name = f"{cache_name}_{prefix}"
    return cache_name


def get_pc_setting_name(prob_point_cloud: float) -> str:
    name = "mesh" if prob_point_cloud <= 0 else "pc"
    if prob_point_cloud < 1.0 and prob_point_cloud > 0:
        name = "meshandpc"
    return name


def get_denylist_path(
    cache_dir: str,
    root_dir: str,
    prob_point_cloud: float,
    load_discriminator_dataset: bool,
) -> str:
    cache_path = get_cache_path(cache_dir, root_dir)
    prefix = get_cache_prefix(prob_point_cloud, load_discriminator_dataset)
    return os.path.join(cache_path, f"denylist_{prefix}.json")


def is_valid_cache_dir(cfg: DictConfig) -> bool:

    dataset_cache_dir = get_cache_path(cfg.cache_dir, cfg.root_dir)

    if not os.path.exists(dataset_cache_dir):
        return False
    json_files = glob.glob(os.path.join(dataset_cache_dir, "denylist_*.json"))

    if len(json_files) == 0:
        logger.info(
            f"[CACHE VALIDATION] No denylist found in {dataset_cache_dir}, deleting cache dir. It may have been improperly created before."
        )
        os.system(f"rm -rf {dataset_cache_dir}")  # Save from future misery
        return False
    data_files = glob.glob(os.path.join(dataset_cache_dir, "*.h5"))
    if len(data_files) == 0:
        logger.info(
            f"[CACHE VALIDATION] No data files found in {dataset_cache_dir}, deleting cache dir. It may have been improperly created before."
        )
        os.system(f"rm -rf {dataset_cache_dir}")  # Save from future misery
        return False
    if len(data_files) > 0:
        for split in ["train", "valid"]:
            prefix = get_cache_prefix(
                cfg.prob_point_cloud, cfg.load_discriminator_dataset
            )
            data_files = glob.glob(
                os.path.join(dataset_cache_dir, f"cache_{split}_{prefix}.h5")
            )
            if len(data_files) == 0:
                logger.info(
                    f"[CACHE VALIDATION] No cache files found for split {split}, skipping"
                )
                return False

            path_to_h5_file = data_files[0]
            logger.info(
                f"[CACHE VALIDATION] Found cache file for split {split}: {path_to_h5_file}. Loading cache..."
            )
            try:
                dataset_cache = GraspGenDatasetCache.load_from_h5_file(path_to_h5_file)
            except OSError as e:
                logger.error(
                    f"[CACHE VALIDATION] Error loading cache file {path_to_h5_file}: {e}"
                )
                logger.error(
                    f"[CACHE VALIDATION] Cache file {path_to_h5_file} is not valid, deleting cache dir. It may have been improperly created before."
                )
                os.system(f"rm -rf {path_to_h5_file}")  # Save from future misery
                return False
            except KeyError as e:
                logger.error(
                    f"[CACHE VALIDATION] KeyError loading cache file {path_to_h5_file}: {e}"
                )
                logger.error(
                    f"[CACHE VALIDATION] Cache file {path_to_h5_file} is not valid, deleting cache dir. It may have been improperly created before."
                )
                os.system(f"rm -rf {path_to_h5_file}")  # Save from future misery
                return False

            cache_dir = cfg.cache_dir
            root_dir = cfg.root_dir
            prob_point_cloud = cfg.prob_point_cloud
            load_discriminator_dataset = cfg.load_discriminator_dataset

            denylist_path = get_denylist_path(
                cache_dir, root_dir, prob_point_cloud, load_discriminator_dataset
            )
            denylist = load_from_json(denylist_path)

            if len(denylist) > 0:
                denylist = list(denylist.keys())

            splits_file_path = f"{root_dir}/{split}.txt"
            scenes = load_scenes_from_text_file(splits_file_path)
            logger.info(f"Loading dataset from config: {splits_file_path}")

            for idx in tqdm(range(len(scenes))):
                key = scenes[idx]

                if key not in dataset_cache:
                    if key not in denylist:
                        logger.info(
                            f"[CACHE VALIDATION] Cache for split {split} is only partially complete. {key} is not in cache. {len(dataset_cache._cache.keys())}/{int(len(scenes) - len(denylist))} completed"
                        )
                        return False
            logger.info(
                f"[CACHE VALIDATION] Cache for split {split} is complete. It has been properly generated."
            )

    return True


def load_grasps_with_contacts(
    data: dict,
    pts: torch.Tensor,
    seg: torch.Tensor,
    world2cam: torch.Tensor,
    contact_radius: float,
    offset_bins: torch.Tensor,
    inference: bool,
    load_grasp: bool,
) -> dict:
    names, masks, grasping_masks, matched_grasps = [], [], [], []
    contact_dirs = torch.zeros_like(pts)
    approach_dirs = torch.zeros_like(pts)
    offsets = torch.zeros_like(pts[..., 0])

    num_grasps = 0
    for name, info in data["objects"].items():
        seg_id = info["seg_id"][()]
        mask = seg == seg_id

        if seg_id == 1:
            continue

        if mask.sum() == 0:
            # object is not visible, skip
            continue
        masks.append(mask)
        names.append(name)
        has_grasp = False

        if not load_grasp:
            continue

        if "grasps" in info:
            contacts = torch.from_numpy(info["contacts"][...]).float()
            grasps = torch.from_numpy(info["grasps"][...]).float()

            num_grasps += len(grasps)
            if len(grasps) == 0:
                continue

            if world2cam is not None:
                # convert contacts and grasps to camera coordinate
                contacts = contacts @ world2cam[:3, :3].T + world2cam[:3, 3]
                grasps = world2cam @ grasps

            if len(contacts.shape) == 3:

                contact_dir = contacts[:, 1] - contacts[:, 0]

                offset = contact_dir.norm(dim=1)
                contact_dir = contact_dir / offset.unsqueeze(1)
                approach_dir = grasps[:, :3, 2]

                # Mx2x3 -> 2Mx3
                contacts = contacts.transpose(0, 1).reshape(-1, 3)
                contact_dir = torch.cat([contact_dir, -contact_dir])
                approach_dir = torch.cat([approach_dir, approach_dir])
                offset = torch.cat([offset, offset])
                grasps = torch.cat([grasps, grasps])

            else:
                assert len(contacts.shape) == 2
                contact_dir = grasps[:, :3, 0]
                approach_dir = grasps[:, :3, 2]

                offset = 0.02 * torch.ones(len(contact_dir))

            tree = KDTree(contacts.numpy())
            dist, idx = tree.query(pts[mask].numpy())
            matched = dist < contact_radius
            idx = idx[matched]
            grasps = grasps[idx]

            if matched.sum() > 0:
                pt_i = torch.where(mask)[0]
                contact_mask = torch.zeros_like(mask)
                contact_mask[pt_i[matched[:, 0]]] = 1
                contact_dirs[contact_mask] = contact_dir[idx]
                approach_dirs[contact_mask] = approach_dir[idx]
                offsets[contact_mask] = offset[idx]
                grasping_masks.append(contact_mask)
                matched_grasps.append(grasps)
                has_grasp = True
            elif inference:
                grasping_masks.append(torch.zeros_like(mask))
                matched_grasps.append(torch.zeros(0, 4, 4))
            else:
                pass
        elif inference:
            grasping_masks.append(torch.zeros_like(mask))
            matched_grasps.append(torch.zeros(0, 4, 4))

    if len(grasping_masks) > 0:
        grasping_masks = torch.stack(grasping_masks).float()
    else:
        # No grasp, skip
        return {"invalid": True}

    num_matched_grasps = np.sum([len(grasps_obj) for grasps_obj in matched_grasps])
    if num_matched_grasps == 0:
        # TODO - Root cause this issue
        # No grasp, skip
        return {"invalid": True}

    contact_any_obj = grasping_masks.any(dim=0)
    contact_dirs = contact_dirs[contact_any_obj]
    approach_dirs = approach_dirs[contact_any_obj]
    offsets = offsets[contact_any_obj]
    outputs = {
        "names": names,
        "instance_masks": torch.stack(masks).float(),
        "grasping_masks": grasping_masks,
        "contact_dirs": contact_dirs,
        "approach_dirs": approach_dirs,
        "grasps": matched_grasps,
    }

    if offset_bins is not None:
        if inference:
            outputs["offsets"] = offsets
        else:
            labels = torch.bucketize(offsets, torch.tensor(offset_bins)) - 1
            outputs["offsets"] = torch.clip(labels, 0, len(offset_bins) - 1)

            # TODO: Explain clearly - this is for when offsets go OOB - some issue with contact points
            if outputs["offsets"].max().item() == 10:

                return {"invalid": True}
    return outputs


def load_scenes_from_text_file(file_path: str) -> list:
    """
    Load object IDs or relative paths to object meshes from a text file.

    Each line in the text file should contain one object ID or relative path.
    Lines are stripped of whitespace and empty lines are ignored.

    Args:
        file_path: Path to the text file containing object IDs/paths

    Returns:
        List of object IDs or relative paths as strings

    Raises:
        FileNotFoundError: If the text file doesn't exist
    """
    all_objects = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                all_objects.append(line)
    return all_objects


class PickDataset(Dataset):
    def __init__(
        self,
        root_dir,
        cache_dir,
        split,
        tasks,
        num_points,
        num_obj_points,
        cam_coord,
        num_rotations,
        grid_res,
        jitter_scale,
        contact_radius,
        dist_above_table,
        offset_bins,
        robot_prob,
        random_seed,
        inference=False,
        scenes=None,
        rotation_augmentation=False,
        downsample_points=True,
        add_depth_noise=False,
        load_patch=False,
        patch_width=200,
        prob_point_cloud=-1.0,
        object_root_dir="",
        grasp_root_dir="",
        dataset_name="acronym",
        dataset_version="v0",
        gripper_name="franka_panda",
        num_grasps_per_object=20,
        load_discriminator_dataset=False,
        load_contact=False,
        prefiltering=False,
        discriminator_ratio=[0.50, 0.20, 0.25, 0.05, 0.0],
        visualize_batch=False,
        onpolicy_dataset_dir=None,
        onpolicy_dataset_h5_path=None,
        preload_dataset=False,
        redundancy=1,
    ):
        self.split = split
        self.scenes, self.scene2h5 = [], {}
        self.load_patch = load_patch
        cache_dir = cache_dir
        self.patch_width = patch_width
        self.load_contact = load_contact
        self.prob_point_cloud = prob_point_cloud
        self.preload_dataset = preload_dataset
        self.redundancy = redundancy
        self.load_discriminator_dataset = load_discriminator_dataset
        self._prefiltering = prefiltering
        self.denylist_path = get_denylist_path(
            cache_dir, root_dir, prob_point_cloud, load_discriminator_dataset
        )
        self._cache_dir = get_cache_path(cache_dir, root_dir)
        logger.info(f"cache_dir: {self._cache_dir}")
        logger.info(f"Denylist path: {self.denylist_path}")

        self.denylist = load_from_json(self.denylist_path)
        if len(self.denylist) > 0:
            if type(self.denylist) == dict:
                self.denylist = list(self.denylist.keys())

        file_path = f"{root_dir}/{self.split}.txt"
        self.scenes = load_scenes_from_text_file(file_path)
        logger.info(f"Loading dataset from config: {file_path}")

        if len(self.denylist) > 0:
            init_datapoints = len(self.scenes)
            self.scenes = [scene for scene in self.scenes if scene not in self.denylist]
            logger.info(
                f"{init_datapoints - len(self.scenes)} out of {init_datapoints} scenes in denylist. Final number of datapoints {len(self.scenes)}"
            )
        else:
            logger.info("No filtering; All datapoints are used")

        self.overfitting_mode = False
        if len(self.scenes) == 1:
            multiplier = 100 if self.split == "train" else 10
            if self.split != "train" and not rotation_augmentation:
                multiplier = 1
            self.scenes = [self.scenes[0]] * multiplier
            self.overfitting_mode = True
        self.load_patch = load_patch
        self.num_points = num_points
        self.num_obj_points = num_obj_points
        self.cam_coord = cam_coord
        self.num_rotations = num_rotations
        self.grid_res = grid_res
        self.jitter_scale = jitter_scale
        self.robot_prob = robot_prob
        self.contact_radius = contact_radius
        self.dist_above_table = dist_above_table
        self.random_seed = random_seed
        self.inference = inference
        self.rotation_augmentation = rotation_augmentation
        self.downsample_points = downsample_points
        self.add_depth_noise = add_depth_noise
        self.root_dir = root_dir
        self.object_root_dir = object_root_dir
        self.grasp_root_dir = grasp_root_dir
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.gripper_name = gripper_name
        self.num_grasps_per_object = num_grasps_per_object
        self.gripper_info = get_gripper_info(self.gripper_name)
        self.gripper_collision_mesh = self.gripper_info.collision_mesh
        self.gripper_visual_mesh = self.gripper_info.visual_mesh
        self.offset_bins = self.gripper_info.offset_bins
        self.discriminator_ratio = discriminator_ratio
        self.visualize_batch = visualize_batch
        self.onpolicy_dataset_h5_path = onpolicy_dataset_h5_path
        self.onpolicy_dataset_dir = onpolicy_dataset_dir

        # Initialize the appropriate grasp dataset reader
        if is_webdataset(self.grasp_root_dir):
            self.grasp_dataset_reader = GraspWebDatasetReader(self.grasp_root_dir)
        else:
            self.grasp_dataset_reader = GraspJsonDatasetReader(self.grasp_root_dir)

        self.onpolicy_dataset_h5_path = onpolicy_dataset_h5_path
        self.onpolicy_dataset_dir = onpolicy_dataset_dir

        if self.onpolicy_dataset_dir is not None:
            self.load_onpolicy_dataset()

        if self.preload_dataset:
            self.load_cache()

    def load_onpolicy_dataset(self):
        """
        Loads the onpolicy dataset.
        """
        assert self.onpolicy_dataset_h5_path is not None
        assert self.onpolicy_dataset_dir is not None
        # The below is to load the onpolicy dataset for the validation set. if no directory is found, it is set to None
        if type(self.onpolicy_dataset_h5_path) == str and self.split == "valid":
            if self.onpolicy_dataset_h5_path.find("train") >= 0:
                onpolicy_dataset_h5_path_valid = self.onpolicy_dataset_h5_path.replace(
                    "train", "valid"
                )
                onpolicy_dataset_dir_valid = self.onpolicy_dataset_dir.replace(
                    "train", "valid"
                )
                if os.path.exists(onpolicy_dataset_h5_path_valid):
                    logger.info(
                        f"Onpolicy dataset: Loading onpolicy dataset from {onpolicy_dataset_h5_path_valid}"
                    )
                    self.onpolicy_dataset_h5_path = onpolicy_dataset_h5_path_valid
                    self.onpolicy_dataset_dir = onpolicy_dataset_dir_valid
                else:
                    self.onpolicy_dataset_h5_path = None
                    self.onpolicy_dataset_dir = None

        if self.onpolicy_dataset_h5_path is not None:

            # Computing the mapping from object ids to json paths in the onpolicy dataset
            json_path_onpolicy_filename = os.path.join(
                self._cache_dir, f"{os.path.basename(self.onpolicy_dataset_dir)}.json"
            )
            logger.info("Precomputing all the json file mapping for onpolicy dataset")
            t0 = time.time()

            if os.path.exists(json_path_onpolicy_filename):
                logger.info(
                    f"Onpolicy dataset: Loading json path from cached json file {json_path_onpolicy_filename}"
                )
                self.map_key_to_json_path_online_dataset = load_from_json(
                    json_path_onpolicy_filename
                )
            else:
                logger.info(
                    f"Onpolicy dataset: Loading json path from scratch {json_path_onpolicy_filename}"
                )
                import h5py
                from tqdm import tqdm

                from grasp_gen.dataset.dataset_utils import (
                    get_json_file_given_object_id,
                )

                possible_grasp_keys = [
                    "grasps.json",
                    "*suction_grasps.json",
                ]  # This is because ACRONYM pipeline generates grasp jsons with different names
                json_files = []
                for grasp_key in possible_grasp_keys:
                    logger.info(f"Searching using grasp key {grasp_key}")
                    json_files += sorted(
                        glob.glob(
                            f"{self.onpolicy_dataset_dir}/**/{grasp_key}",
                            recursive=True,
                        )
                    )
                    json_files += sorted(
                        glob.glob(
                            f"{self.onpolicy_dataset_dir}/**/**/{grasp_key}",
                            recursive=True,
                        )
                    )  # This is to account for the case when the dataset is built with MapReduce

                h5 = h5py.File(self.onpolicy_dataset_h5_path, "r")
                h5_object_ids = list(h5["objects"].keys())
                map_object_uuid_to_id = {}
                map_object_id_to_uuid = {}
                for h5_object_id in h5_object_ids:
                    uuid = (
                        h5["objects"][h5_object_id]["asset_path"][...]
                        .item()
                        .decode("utf-8")
                    )
                    map_object_uuid_to_id[uuid] = h5_object_id
                    map_object_id_to_uuid[h5_object_id] = uuid

                map_basenameuuid_to_json_file = {}
                logger.info(f"Onpolicy dataset: Parsing json file to UUID mappings")
                for json_file_path in tqdm(json_files):
                    json_path_object_id = load_from_json(json_file_path)["object"][
                        "file"
                    ]
                    map_basenameuuid_to_json_file[
                        os.path.basename(json_path_object_id)
                    ] = json_file_path

                self.map_key_to_json_path_online_dataset = {}
                for idx in tqdm(range(self.__len__())):
                    key = self.scenes[idx]
                    if key not in map_object_id_to_uuid:
                        logger.info(f"Onpolicy dataset: Key {key} not in json dataset")
                        continue
                    try:
                        json_path = map_basenameuuid_to_json_file[
                            os.path.basename(map_object_id_to_uuid[key])
                        ]
                    except:
                        logger.info(
                            f"Onpolicy dataset: File path {map_object_id_to_uuid[key]}, key {key} not in dataset yet "
                        )
                        continue
                    self.map_key_to_json_path_online_dataset[key] = json_path

                dump_object_list(
                    self.map_key_to_json_path_online_dataset,
                    json_path_onpolicy_filename,
                )
            logger.info(f"Onpolicy dataset: That took {time.time() - t0}s. Phew...")

    def load_cache(self, cache_save_freq: int = 2000):
        """
        Converts the dataset into a cached file, loads to system memory.
        """
        from tqdm import tqdm

        assert self.preload_dataset
        logger.info("Preloading dataset to memory")

        cache_save_path = ""
        save_emd_data = False
        all_emd_data = {}
        prefix = get_cache_prefix(
            self.prob_point_cloud, self.load_discriminator_dataset
        )
        cache_name = f"cache_{self.split}_{prefix}.h5"
        cache_save_path = os.path.join(self._cache_dir, cache_name)

        if os.path.exists(cache_save_path):
            self.cache = GraspGenDatasetCache.load_from_h5_file(cache_save_path)
        else:
            logger.info(f"Unable to find cache file: {cache_save_path}")
            self.cache = GraspGenDatasetCache()

        if len(self.cache) > 0:
            logger.info(f"Cache already has {len(self.cache)} objects")

        idx_cache_loaded = 0
        for idx in tqdm(range(self.__len__())):
            key = self.scenes[idx]

            if key in self.cache:
                # Skip if the object is already in the cache
                continue
            else:
                idx_cache_loaded += 1

            onpolicy_json_path = None
            onpolicy_data_found = True
            if self.onpolicy_dataset_dir is not None:
                try:
                    onpolicy_json_path = self.map_key_to_json_path_online_dataset[key]
                except:
                    onpolicy_data_found = False

            error_code, object_grasp_data = load_object_grasp_data(
                key,
                self.object_root_dir,
                self.grasp_root_dir,
                self.dataset_version,
                load_discriminator_dataset=self.load_discriminator_dataset,
                gripper_info=self.gripper_info,
                onpolicy_dataset_dir=self.onpolicy_dataset_dir,
                onpolicy_dataset_h5_path=self.onpolicy_dataset_h5_path,
                onpolicy_json_path=onpolicy_json_path,
                onpolicy_data_found=onpolicy_data_found,
                grasp_dataset_reader=self.grasp_dataset_reader,
            )

            if error_code != DataLoaderError.SUCCESS:
                self.save_to_denylist(self.denylist_path, key, error_code)
                continue

            # This is for plotting only EMD data
            if save_emd_data:
                from grasp_gen.dataset.dataset_utils import compute_emd_data

                emd_data = compute_emd_data(object_grasp_data)
                all_emd_data[key] = emd_data

            rendering_output = []
            POINT_CLOUD_REDUNDANCY = 3
            for _ in range(self.redundancy):

                mesh_mode = (
                    False if np.random.random() <= self.prob_point_cloud else True
                )
                load_contact_batch = self.load_contact

                # Lazy import to avoid forcing pyrender in headless environments
                from grasp_gen.dataset.renderer import render_pc

                outputs, error_code = render_pc(
                    object_grasp_data,
                    self.num_points * POINT_CLOUD_REDUNDANCY,
                    mesh_mode=mesh_mode,
                )

                if error_code == DataLoaderError.RENDERING_SUCCESS:
                    # Check if there are grasps in the visibile area of the point clouds
                    T_move_to_pc_mean = outputs["T_move_to_pc_mean"]
                    grasps = np.array(
                        [
                            T_move_to_pc_mean @ g
                            for g in object_grasp_data.positive_grasps.copy()
                        ]
                    )
                    mask_grasp_visibility = filter_grasps_by_point_cloud_visibility(
                        grasps,
                        outputs["points"],
                        transform_from_base_link_to_tool_tcp=self.gripper_info.transform_from_base_link_to_tool_tcp,
                    )
                    if mask_grasp_visibility is not None:
                        positive_grasps = object_grasp_data.positive_grasps.copy()[
                            mask_grasp_visibility
                        ]
                        outputs["mesh_mode"] = mesh_mode
                        outputs["load_contact_batch"] = load_contact_batch
                        outputs["positive_grasps"] = positive_grasps
                        rendering_output.append(outputs)
                    else:
                        error_code = (
                            DataLoaderError.RENDERING_NO_GRASPS_IN_VISIBLE_POINT_CLOUD
                        )

            if len(rendering_output) == 0:
                logger.error(
                    f"{idx}: Unable to preload object even after sampling pc {self.redundancy} times"
                )
                self.save_to_denylist(self.denylist_path, key, error_code)
                continue
            self.cache[key] = (object_grasp_data, rendering_output)

            if idx_cache_loaded % cache_save_freq == 0:
                self.cache.save_to_h5_file(cache_save_path)

        # NOTE - This round of postprocessing is needed but not sure why. It is a fail safe.
        id_key_pair_missing = []
        denylist = load_from_json(self.denylist_path)
        for idx in tqdm(range(self.__len__())):
            key = self.scenes[idx]
            if key not in self.cache:
                if key not in denylist:
                    logger.info(f"Key not in cache {key} for some reason")
                    self.save_to_denylist(
                        self.denylist_path, key, DataLoaderError.AMBIGUOUS_LOAD_ERROR
                    )
                    id_key_pair_missing.append(key)
        if len(id_key_pair_missing) > 0:
            logger.info(
                f"{len(id_key_pair_missing)} keys were not found in the cache and were forcibly added into the denylist. Need to be rootcaused"
            )
            # print(id_key_pair_missing)

        # If any new data was loaded, save final version for the entire dataset. This will not enter if the cache is already complete.
        if idx_cache_loaded > 0:
            self.cache.save_to_h5_file(cache_save_path)

        if save_emd_data:
            with open(os.path.join(self._cache_dir, "emd_data_100.json"), "w") as f:
                json.dump(all_emd_data, f)

    def save_to_denylist(self, denylist_path, key, error_code):
        if self._prefiltering:
            logger.info(f"Saving key: {key} to denylist at {denylist_path}")
            if os.path.exists(denylist_path):
                denylist = load_from_json(denylist_path)
            else:
                denylist = {}
            denylist[key] = (error_code.code, error_code.description)
            os.system(f"rm -rf {denylist_path}")
            dump_object_list(denylist, denylist_path)

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args["root_dir"] = cfg.root_dir
        args["cache_dir"] = cfg.cache_dir
        args["tasks"] = cfg.tasks
        args["num_points"] = cfg.num_points
        args["num_obj_points"] = cfg.num_object_points
        args["cam_coord"] = cfg.cam_coord
        args["num_rotations"] = cfg.num_rotations
        args["grid_res"] = cfg.grid_resolution
        args["jitter_scale"] = cfg.jitter_scale
        args["contact_radius"] = cfg.contact_radius
        args["dist_above_table"] = cfg.dist_above_table
        args["offset_bins"] = cfg.offset_bins
        args["robot_prob"] = cfg.robot_prob
        args["random_seed"] = cfg.random_seed
        args["rotation_augmentation"] = cfg.rotation_augmentation
        args["downsample_points"] = cfg.downsample_points
        args["add_depth_noise"] = cfg.add_depth_noise
        args["load_patch"] = cfg.load_patch
        args["patch_width"] = cfg.patch_width
        args["prob_point_cloud"] = cfg.prob_point_cloud
        args["object_root_dir"] = cfg.object_root_dir
        args["grasp_root_dir"] = cfg.grasp_root_dir
        args["dataset_name"] = cfg.dataset_name
        args["dataset_version"] = cfg.dataset_version
        args["gripper_name"] = cfg.gripper_name
        args["num_grasps_per_object"] = cfg.num_grasps_per_object
        args["load_discriminator_dataset"] = cfg.load_discriminator_dataset
        args["load_contact"] = cfg.load_contact
        args["prefiltering"] = cfg.prefiltering
        args["discriminator_ratio"] = cfg.discriminator_ratio
        args["visualize_batch"] = cfg.visualize_batch
        args["onpolicy_dataset_dir"] = cfg.onpolicy_dataset_dir
        args["onpolicy_dataset_h5_path"] = cfg.onpolicy_dataset_h5_path
        args["preload_dataset"] = cfg.preload_dataset
        args["redundancy"] = cfg.redundancy
        return args

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented by the subclass")


class ObjectPickDataset(PickDataset):

    def calculate_dataset_kappa(self) -> float:
        """Calculate the mean extent of grasps across all objects in the dataset. See method section in the paper.

        The extent is calculated as the difference between max and min grasp positions
        along each axis, averaged across all objects.

        Returns:
            float: Kappa - Mean grasp extent K across all objects
        """
        from copy import deepcopy as copy

        list_minmax = []
        for j in range(self.__len__()):
            key = self.scenes[j]

            if key in self.cache:
                (object_grasp_data, outputs_red) = self.cache[key]
                outputs = copy(random.choice(outputs_red))
                mesh_mode = outputs["mesh_mode"]
                load_contact_batch = outputs["load_contact_batch"]
                mask = torch.randint(0, outputs["points"].shape[0], (self.num_points,))
                outputs["points"] = outputs["points"][mask]

                T_move_to_pc_mean = outputs["T_move_to_pc_mean"]
                grasps = object_grasp_data.positive_grasps.copy()
                grasps = np.array([T_move_to_pc_mean @ g for g in grasps])
                min_max = grasps[:, :3, 3].max(axis=0) - grasps[:, :3, 3].min(axis=0)

                list_minmax.append(min_max)
        kappa = 1 / np.array(list_minmax).mean()
        return kappa

    def __getitem__(self, idx):
        key = self.scenes[idx]
        if self.preload_dataset:
            from copy import deepcopy as copy

            if key not in self.cache:
                # print(f"Key {key} not in cache")
                denylist = load_from_json(self.denylist_path)
                if key not in denylist:
                    logger.info(f"Key {key} not in cache and not in denylist")
                    self.save_to_denylist(
                        self.denylist_path, key, DataLoaderError.AMBIGUOUS_LOAD_ERROR
                    )
                return {"invalid": True}

            (object_grasp_data, outputs_red) = self.cache[key]

            outputs = copy(random.choice(outputs_red))
            mesh_mode = outputs["mesh_mode"]
            load_contact_batch = outputs["load_contact_batch"]
            mask = torch.randint(0, outputs["points"].shape[0], (self.num_points,))
            outputs["points"] = outputs["points"][mask]

        else:

            onpolicy_json_path = None
            onpolicy_data_found = True
            if self.onpolicy_dataset_dir is not None:
                try:
                    onpolicy_json_path = self.map_key_to_json_path_online_dataset[key]
                except:
                    onpolicy_data_found = False

            error_code, object_grasp_data = load_object_grasp_data(
                key,
                self.object_root_dir,
                self.grasp_root_dir,
                self.dataset_version,
                load_discriminator_dataset=self.load_discriminator_dataset,
                gripper_info=self.gripper_info,
                onpolicy_dataset_dir=self.onpolicy_dataset_dir,
                onpolicy_dataset_h5_path=self.onpolicy_dataset_h5_path,
                onpolicy_json_path=onpolicy_json_path,
                onpolicy_data_found=onpolicy_data_found,
                grasp_dataset_reader=self.grasp_dataset_reader,
            )

            if object_grasp_data is None:
                self.save_to_denylist(self.denylist_path, key, error_code)
                # Invalid datapoint
                return {"invalid": True}

            mesh_mode = False if np.random.random() <= self.prob_point_cloud else True
            load_contact_batch = self.load_contact

            # Lazy import to avoid forcing pyrender in headless environments
            from grasp_gen.dataset.renderer import render_pc

            outputs, error_code = render_pc(
                object_grasp_data, self.num_points, mesh_mode=mesh_mode
            )
            if error_code != DataLoaderError.RENDERING_SUCCESS:
                # Invalid point cloud rendering
                self.save_to_denylist(self.denylist_path, key, error_code)
                return {"invalid": True}

            T_move_to_pc_mean = outputs["T_move_to_pc_mean"]
            grasps = np.array(
                [
                    T_move_to_pc_mean @ g
                    for g in object_grasp_data.positive_grasps.copy()
                ]
            )
            mask_grasp_visibility = filter_grasps_by_point_cloud_visibility(
                grasps,
                outputs["points"],
                transform_from_base_link_to_tool_tcp=self.gripper_info.transform_from_base_link_to_tool_tcp,
            )

            if mask_grasp_visibility is not None:
                positive_grasps = object_grasp_data.positive_grasps.copy()[
                    mask_grasp_visibility
                ]
                outputs["positive_grasps"] = positive_grasps
            else:
                if error_code == DataLoaderError.RENDERING_SUCCESS:
                    error_code = (
                        DataLoaderError.RENDERING_NO_GRASPS_IN_VISIBLE_POINT_CLOUD
                    )
                    # Invalid point cloud rendering
                    self.save_to_denylist(self.denylist_path, key, error_code)
                    return {"invalid": True}

        T_move_to_pc_mean = outputs["T_move_to_pc_mean"]
        grasps = outputs["positive_grasps"]
        grasps = np.array([T_move_to_pc_mean @ g for g in grasps])

        # Extra stuff added to outputs later. TODO - Clean this all up
        xyz = outputs["points"]
        if type(xyz) == np.ndarray:
            xyz = torch.from_numpy(xyz).float()
        num_points = self.num_points
        seg = 5 * np.ones(num_points).astype(np.int32)
        rgb = (
            150.0
            * np.vstack(
                [np.ones(num_points), np.zeros(num_points), np.zeros(num_points)]
            ).T
        )
        cam_pose = np.eye(4)
        rgb = torch.from_numpy(rgb).float()
        seg = torch.from_numpy(seg).float()
        cam_pose = torch.from_numpy(cam_pose).float()
        outputs["inputs"] = torch.cat([xyz, rgb], dim=1)
        outputs["seg"] = seg
        outputs["rgb"] = rgb
        outputs["cam_pose"] = cam_pose

        data = {
            "objects": {
                "obj0": {
                    "seg_id": np.array(5),
                    "grasps": grasps,
                }
            }
        }

        outputs["scene"] = self.scenes[idx]
        outputs["task"] = "pick"
        cam_pose = outputs["cam_pose"] if self.cam_coord else None
        world2cam = outputs["cam_pose"].inverse() if self.cam_coord else None

        grasps_ground_truth = object_grasp_data.positive_grasps.copy()
        grasps_ground_truth = np.array(
            [T_move_to_pc_mean @ g for g in grasps_ground_truth]
        )
        outputs["grasps_ground_truth"] = grasps_ground_truth

        if self.load_discriminator_dataset:
            negative_grasps = object_grasp_data.negative_grasps.copy()
            if negative_grasps is not None:
                negative_grasps = np.array(
                    [T_move_to_pc_mean @ g for g in negative_grasps]
                )
                outputs["negative_grasps"] = negative_grasps

        if load_contact_batch:
            contacts = object_grasp_data.contacts.copy()
            assert contacts is not None
            if len(contacts.shape) == 3:
                contacts[:, 0, :] = tra.transform_points(
                    contacts[:, 0, :], T_move_to_pc_mean
                )
                contacts[:, 1, :] = tra.transform_points(
                    contacts[:, 1, :], T_move_to_pc_mean
                )
            else:
                contacts = tra.transform_points(contacts, T_move_to_pc_mean)
            data["objects"]["obj0"]["contacts"] = contacts

            outputs.update(
                load_grasps_with_contacts(
                    data,
                    outputs["points"],
                    outputs["seg"],
                    world2cam,
                    self.contact_radius,
                    self.offset_bins,
                    self.inference,
                    True,
                )
            )

            # outputs['grasps'] = outputs['grasps'][0]
            outputs.update(
                {
                    "ee_pose": torch.eye(4),
                    "obj_pose": torch.eye(4),
                    "object_inputs": torch.zeros(self.num_obj_points, 6),
                    "bottom_center": torch.zeros(3),
                    "object_center": torch.zeros(3),
                    "placement_masks": torch.zeros(self.num_rotations, self.num_points),
                    "placement_region": torch.zeros(self.num_points),
                }
            )
        else:
            outputs["grasps"] = [
                torch.from_numpy(grasps),
            ]
            outputs["names"] = ["obj0"]

        if "grasps" not in outputs:
            self.save_to_denylist(
                self.denylist_path,
                key,
                DataLoaderError.GRASP_POINT_CLOUD_CORRESPONDENCE_ERROR,
            )
            return {"invalid": True}

        if not self.load_contact:
            # Sanitize the dictionary output for diffusion model training.
            blacklist_keys = [
                "instance_masks",
                "grasping_masks",
                "contact_dirs",
                "approach_dirs",
                "offsets",
                "ee_pose",
                "obj_pose",
                "object_inputs",
                "bottom_center",
                "object_center",
                "placement_masks",
                "placement_region",
            ]
            for key in blacklist_keys:
                if key in outputs:
                    del outputs[key]

        # TODO - make sure rotation_augmentation works with pc?
        if self.rotation_augmentation:
            pc = outputs["points"]
            if type(pc) == torch.Tensor:
                pc = pc.cpu().numpy()

            if len(pc.shape) == 3 and pc.shape[0] == 1:
                pc = pc.squeeze(0)
            T_world_to_pcmean = tra.translation_matrix(-pc.mean(axis=0))

            T_pcmean_to_world = tra.inverse_matrix(T_world_to_pcmean)
            T_rotation = get_rotation_augmentation(stratified_sampling=False)
            T_aug = T_pcmean_to_world @ T_rotation @ T_world_to_pcmean
            pc = tra.transform_points(pc, T_aug)
            xyz = torch.from_numpy(pc).float()

            if "grasps_ground_truth" in outputs:
                grasps_ground_truth = outputs["grasps_ground_truth"]
                outputs["grasps_ground_truth"] = np.array(
                    [T_aug @ g for g in grasps_ground_truth]
                )

            if "grasps" in outputs:
                output_grasps_rotated = []
                for grasps in outputs["grasps"]:
                    if len(grasps) > 0:
                        grasps_rotated = np.array(
                            [T_aug @ g for g in grasps.cpu().numpy()]
                        )
                        grasps_rotated = torch.from_numpy(grasps_rotated).float()
                    else:
                        grasps_rotated = grasps
                    output_grasps_rotated.append(grasps_rotated)
                outputs["grasps"] = output_grasps_rotated

                if self.load_discriminator_dataset:
                    if "negative_grasps" in outputs:
                        negative_grasps = outputs["negative_grasps"]
                        outputs["negative_grasps"] = np.array(
                            [T_aug @ g for g in negative_grasps]
                        )

            outputs["points"] = xyz
            rgb = outputs["rgb"]
            center = xyz.mean(dim=0)
            outputs["inputs"] = torch.cat([xyz - center, rgb], dim=1)

        positive_grasps_onpolicy = None
        negative_grasps_onpolicy = None
        if self.load_discriminator_dataset:
            if object_grasp_data.positive_grasps_onpolicy is not None:
                try:
                    if len(object_grasp_data.positive_grasps_onpolicy) > 0:
                        positive_grasps_onpolicy = np.copy(
                            object_grasp_data.positive_grasps_onpolicy
                        )
                        positive_grasps_onpolicy[:, 3, 3] = 1.0
                        positive_grasps_onpolicy = np.array(
                            [
                                T_move_to_pc_mean @ np.array(g)
                                for g in positive_grasps_onpolicy.tolist()
                            ]
                        )
                        if self.rotation_augmentation:
                            positive_grasps_onpolicy = np.array(
                                [T_aug @ g for g in positive_grasps_onpolicy]
                            )
                except Exception as e:
                    # print(f"Error loading positive grasps onpolicy: {e}")
                    positive_grasps_onpolicy = None

            if object_grasp_data.negative_grasps_onpolicy is not None:
                try:
                    if len(object_grasp_data.negative_grasps_onpolicy) > 0:
                        negative_grasps_onpolicy = np.copy(
                            object_grasp_data.negative_grasps_onpolicy
                        )
                        negative_grasps_onpolicy[:, 3, 3] = 1.0
                        negative_grasps_onpolicy = np.array(
                            [T_move_to_pc_mean @ g for g in negative_grasps_onpolicy]
                        )
                        if self.rotation_augmentation:
                            negative_grasps_onpolicy = np.array(
                                [T_aug @ g for g in negative_grasps_onpolicy]
                            )
                except Exception as e:
                    # print(f"Error loading negative grasps onpolicy: {e}")
                    negative_grasps_onpolicy = None

        if self.inference:
            outputs["object_points"] = torch.rand(self.num_obj_points, 3)

        obj_asset_path = object_grasp_data.object_asset_path
        if obj_asset_path.find(self.object_root_dir) >= 0:
            obj_asset_path_rel_idx = obj_asset_path.find(self.object_root_dir) + len(
                self.object_root_dir
            )
            obj_asset_path_rel = obj_asset_path[obj_asset_path_rel_idx:]
        else:
            obj_asset_path_rel = obj_asset_path

        try:
            trimesh.load(obj_asset_path)
        except:
            logger.error(f"Error loading object asset path: {obj_asset_path}")
        obj_scale = object_grasp_data.object_scale
        obj_pose = T_move_to_pc_mean

        if self.rotation_augmentation:
            obj_pose = T_aug @ obj_pose

        temp_object_path_dir = "/object_dataset/simplified"
        if obj_asset_path.find(temp_object_path_dir) >= 0:
            if os.path.isdir(temp_object_path_dir):
                # if this dir actually exists, skip
                pass
            else:
                assert os.path.isdir(self.object_root_dir)
                obj_asset_path = obj_asset_path.replace(temp_object_path_dir, "")
                if obj_asset_path.startswith("/"):
                    obj_asset_path = obj_asset_path[1:]
                obj_asset_path = os.path.join(self.object_root_dir, obj_asset_path)

        scene_info = {
            "assets": [
                obj_asset_path,
            ],
            "scales": [
                obj_scale,
            ],
            "poses": [
                obj_pose,
            ],
        }

        outputs["scene_info"] = scene_info

        if self.load_discriminator_dataset:

            positive_grasps = outputs["grasps"][0]
            negative_grasps = (
                outputs["negative_grasps"]
                if "negative_grasps" in outputs and len(outputs["negative_grasps"]) != 0
                else None
            )

            batch_data, scene_mesh = load_discriminator_batch_with_stratified_sampling(
                self.num_grasps_per_object,
                positive_grasps,
                self.discriminator_ratio,
                scene_info,
                self.gripper_collision_mesh,
                negative_grasps,
                positive_grasps_onpolicy=positive_grasps_onpolicy,
                negative_grasps_onpolicy=negative_grasps_onpolicy,
            )

            outputs.update(batch_data)

            mask = np.random.randint(0, len(positive_grasps), 1000)
            outputs["grasps_highres"] = positive_grasps

            if self.visualize_batch:
                # if len(object_grasp_data.positive_grasps) < 100:
                logger.info(
                    "Batch Statistics for object: ",
                    object_grasp_data.object_asset_path,
                    "\n",
                )
                logger.info(
                    f"Positive Grasps: {len(object_grasp_data.positive_grasps)}"
                )
                logger.info(
                    f"Negative Grasps: {len(object_grasp_data.negative_grasps)}"
                )
                try:
                    logger.info(
                        f"Positive Grasps Onpolicy: {len(object_grasp_data.positive_grasps_onpolicy)}"
                    )
                    logger.info(
                        f"Negative Grasps Onpolicy: {len(object_grasp_data.negative_grasps_onpolicy)}"
                    )
                except:
                    pass
                visualize_discriminator_dataset(
                    batch_data,
                    scene_mesh,
                    self.gripper_name,
                    self.gripper_visual_mesh,
                    pointcloud=outputs["points"],
                )
        else:

            if self.num_grasps_per_object != -1:
                grasps_gt = outputs["grasps"][0]

                mask_grasps_filtered = np.random.randint(
                    0, len(grasps_gt), self.num_grasps_per_object
                )

                outputs["grasps"] = grasps_gt[mask_grasps_filtered]
                outputs["grasps_highres"] = grasps_gt

            if not load_contact_batch:
                for key in ["points"]:
                    if type(outputs[key]) == np.ndarray:
                        outputs[key] = torch.from_numpy(outputs[key])
                    outputs[key] = outputs[key].unsqueeze(0).repeat(1, 1, 1)
                outputs["grasps"] = torch.from_numpy(
                    np.array(outputs["grasps"]).astype(np.float32)
                )

            if self.visualize_batch:
                logger.info(
                    "Object asset:",
                    self.scenes[idx],
                    object_grasp_data.object_asset_path,
                )
                grasps_gt = outputs["grasps"]
                if type(grasps_gt) == list and len(grasps_gt) == 1:
                    grasps_gt = grasps_gt[0]
                if load_contact_batch:
                    contacts = object_grasp_data.contacts.copy()
                    pc = outputs["points"].cpu().numpy()
                else:
                    contacts = None
                    pc = outputs["points"][0].cpu().numpy()
                visualize_generator_dataset(
                    obj_pose,
                    grasps_gt,
                    self.gripper_name,
                    load_contact_batch,
                    contacts,
                    pc,
                    self.gripper_visual_mesh,
                )
        if type(outputs["points"]) == np.ndarray:
            outputs["points"] = torch.from_numpy(outputs["points"]).float()
        if len(outputs["points"].shape) == 2:
            outputs["points"] = outputs["points"].unsqueeze(0)
        return outputs


def generate_negative_hardnegatives(
    num_grasps: int,
    grasps_starter: np.ndarray,
    scene_mesh: trimesh.base.Trimesh,
    gripper_mesh: trimesh.base.Trimesh,
) -> Union[None, np.ndarray]:
    """Generate hard negative grasp examples by perturbing positive grasps to create collisions.

    This function creates three sets of negative grasp examples:
    1. Grasps with perturbed orientations
    2. Grasps with perturbed translations along z-axis
    3. Grasps with both perturbed orientations and translations

    The function then checks for collisions between these grasps and the scene/gripper
    meshes to identify valid hard negatives.

    Args:
        num_grasps (int): Number of hard negative grasps to generate
        grasps_starter (np.ndarray): Array of initial positive grasps to perturb, shape (N, 4, 4)
        scene_mesh (trimesh.base.Trimesh): Mesh of the scene/object to check collisions against
        gripper_mesh (trimesh.base.Trimesh): Mesh of the gripper to check collisions against

    Returns:
        Union[None, np.ndarray]: Array of hard negative grasps that collide, shape (num_grasps, 4, 4).
        Returns None if no valid colliding grasps are found.
    """
    if num_grasps == 0:
        return

    upsample_factor = 0.70  # NOTE: This number is specifically tuned to ensure there are enough colliding grasps to choose from in the next step, which still being minimized (as it introduces delays)
    num_grasps_upsampled = int(upsample_factor * num_grasps)
    neg_grasps_src = grasps_starter[
        np.random.randint(0, len(grasps_starter), num_grasps_upsampled)
    ]
    # Add grasps by perturbing a little bit, the orientation.  Make sure these collide
    lims = [-np.pi / 6, np.pi / 6]
    neg_grasps_candidate_set_1 = np.array(
        [
            tra.euler_matrix(rot[0], rot[1], rot[2]) @ g
            for rot, g in zip(
                np.random.uniform(lims[0], lims[1], size=(num_grasps_upsampled, 3)),
                neg_grasps_src,
            )
        ]
    )

    # Add grasps by perturbing the -translation, but fixed orientation. Make sure these collide
    lims = [0.03, 0.08]
    neg_grasps_candidate_set_2 = np.array(
        [
            g @ tra.translation_matrix([0, 0, z])
            for z, g in zip(
                np.random.uniform(lims[0], lims[1], size=num_grasps_upsampled),
                neg_grasps_src,
            )
        ]
    )

    # Do a combination of the two above.
    lims_trans = [0.02, 0.06]
    lims_rot = [-np.pi / 6, np.pi / 6]
    neg_grasps_candidate_set_3 = np.array(
        [
            tra.euler_matrix(rot[0], rot[1], rot[2])
            @ g
            @ tra.translation_matrix([0, 0, z])
            for z, rot, g in zip(
                np.random.uniform(
                    lims_trans[0], lims_trans[1], size=num_grasps_upsampled
                ),
                np.random.uniform(
                    lims_rot[0], lims_rot[1], size=(num_grasps_upsampled, 3)
                ),
                neg_grasps_src,
            )
        ]
    )
    neg_grasp_candidates = np.vstack(
        [
            # neg_grasps_candidate_set_0,
            neg_grasps_candidate_set_1,
            neg_grasps_candidate_set_2,
            neg_grasps_candidate_set_3,
        ]
    )

    collision = check_collision(scene_mesh, gripper_mesh, neg_grasp_candidates)
    neg_grasp_hn = neg_grasp_candidates[collision]
    # print(f"Hardnegatives: {len(neg_grasp_hn)} out of {len(neg_grasp_candidates)} grasps are colliding")

    grasps = None
    if len(neg_grasp_hn) != 0:
        mask = np.random.randint(0, len(neg_grasp_hn), num_grasps)
        # if num_grasps >= len(neg_grasp_hn):
        #     print(f"Hardnegatives: Subsampling {num_grasps} from {len(neg_grasp_hn)} grasps")
        grasps = neg_grasp_hn[mask]
    return grasps


def generate_negative_retract(
    num_grasps: int, grasps_starter: np.ndarray
) -> np.ndarray:
    neg_grasps_src = grasps_starter[
        np.random.randint(0, len(grasps_starter), num_grasps)
    ]
    # Add grasps by perturbing the -translation, but fixed orientation. Make sure these collide
    lims = [0.01, 0.03]
    grasps = np.array(
        [
            g @ tra.translation_matrix([0, 0, -z])
            for z, g in zip(
                np.random.uniform(lims[0], lims[1], size=num_grasps), neg_grasps_src
            )
        ]
    )

    return grasps


def generate_negative_freespace(num_grasps: int) -> np.ndarray:
    # Step 3 - Add grasps in the air to get rid of outliers in the model
    lims_trans = [-0.4, 0.4]
    lims_rot = [-np.pi, np.pi]
    grasps = np.array(
        [
            tra.translation_matrix([trans[0], trans[1], trans[2]])
            @ tra.euler_matrix(rot[0], rot[1], rot[2])
            for trans, rot in zip(
                np.random.uniform(lims_trans[0], lims_trans[1], size=(num_grasps, 3)),
                np.random.uniform(lims_rot[0], lims_rot[1], size=(num_grasps, 3)),
            )
        ]
    )
    return grasps


def load_discriminator_batch_with_stratified_sampling(
    num_grasps_per_batch: int,
    grasps_positive: np.ndarray,
    ratio: dict,
    scene_info: dict,
    gripper_mesh: trimesh.base.Trimesh,
    grasps_negative: np.ndarray = None,
    positive_grasps_onpolicy: np.ndarray = None,
    negative_grasps_onpolicy: np.ndarray = None,
):

    N = num_grasps_per_batch

    include_true_neg = grasps_negative is not None
    include_true_pos = grasps_positive is not None

    assert not (
        not include_true_pos and not include_true_neg
    ), "Cannot have a situation where both positive and negative grasps are not in the batch"

    num_pos_true_grasps = int(N * ratio[MAPPING_NAME2ID["pos_true"]])
    num_neg_true_grasps = int(N * ratio[MAPPING_NAME2ID["neg_true"]])
    num_neg_hncolliding = int(N * ratio[MAPPING_NAME2ID["neg_hncolliding"]])
    num_neg_freespace = int(N * ratio[MAPPING_NAME2ID["neg_freespace"]])
    num_neg_hnretract = int(N * ratio[MAPPING_NAME2ID["neg_hnretract"]])

    num_pos_true_onpolicy_grasps = int(N * ratio[MAPPING_NAME2ID["pos_true_onpolicy"]])
    num_neg_true_onpolicy_grasps = int(N * ratio[MAPPING_NAME2ID["neg_true_onpolicy"]])

    if positive_grasps_onpolicy is None:
        num_pos_true_grasps += num_pos_true_onpolicy_grasps
        num_pos_true_onpolicy_grasps = 0

        num_neg_true_grasps += num_neg_true_onpolicy_grasps
        num_neg_true_onpolicy_grasps = 0

    if include_true_pos:
        if torch.is_tensor(grasps_positive):
            grasps_positive = grasps_positive.cpu().numpy()
    else:
        num_neg_true_grasps += num_pos_true_grasps
        num_pos_true_grasps = 0
        num_pos_true_onpolicy_grasps = 0

    if not include_true_neg:
        assert include_true_pos
        num_pos_true_grasps += num_neg_true_grasps
        num_neg_true_grasps = 0

    obj_scale = scene_info["scales"][0]
    obj_asset_path = scene_info["assets"][0]
    obj_pose = scene_info["poses"][0]

    if os.path.exists(obj_asset_path):
        scene_mesh = trimesh.load(obj_asset_path)
        scene_mesh.apply_scale(obj_scale)
        scene_mesh.apply_transform(obj_pose)
    else:
        scene_mesh = None
        num_neg_hncolliding = 0  # mesh-based hard negatives unavailable

    grasps, grasp_ids = None, None

    if num_pos_true_onpolicy_grasps > 0:
        mask = np.random.randint(
            0, len(positive_grasps_onpolicy), num_pos_true_onpolicy_grasps
        )
        grasps_pos_true_onpolicy = positive_grasps_onpolicy[mask]
        grasps_id_pos_true_onpolicy = (
            torch.ones([num_pos_true_onpolicy_grasps, 1])
            * MAPPING_NAME2ID["pos_true_onpolicy"]
        )
        grasps_pos_true_onpolicy = torch.from_numpy(grasps_pos_true_onpolicy)
        if grasps is None:
            grasps = grasps_pos_true_onpolicy
            grasp_ids = grasps_id_pos_true_onpolicy

    if num_pos_true_grasps > 0:
        mask = np.random.randint(0, len(grasps_positive), num_pos_true_grasps)
        grasps_pos_true = grasps_positive[mask]
        grasps_id_pos_true = (
            torch.ones([num_pos_true_grasps, 1]) * MAPPING_NAME2ID["pos_true"]
        )
        grasps_pos_true = torch.from_numpy(grasps_pos_true)
        if grasps is None:
            grasps = grasps_pos_true
            grasp_ids = grasps_id_pos_true
        else:
            grasps = torch.vstack([grasps, grasps_pos_true]).float()
            grasp_ids = torch.vstack([grasp_ids, grasps_id_pos_true]).int()

    if num_neg_true_onpolicy_grasps > 0:
        mask = np.random.randint(
            0, len(negative_grasps_onpolicy), num_neg_true_onpolicy_grasps
        )
        grasps_neg_true_onpolicy = negative_grasps_onpolicy[mask]
        grasps_id_neg_true_onpolicy = (
            torch.ones([num_neg_true_onpolicy_grasps, 1])
            * MAPPING_NAME2ID["neg_true_onpolicy"]
        )
        grasps_neg_true_onpolicy = torch.from_numpy(grasps_neg_true_onpolicy)
        grasps = torch.vstack([grasps, grasps_neg_true_onpolicy]).float()
        grasp_ids = torch.vstack([grasp_ids, grasps_id_neg_true_onpolicy]).int()

    if num_neg_freespace > 0:
        grasps_neg_freespace = generate_negative_freespace(num_neg_freespace)
        grasps_id_neg_freespace = (
            torch.ones([num_neg_freespace, 1]) * MAPPING_NAME2ID["neg_freespace"]
        )
        grasps_neg_freespace = torch.from_numpy(grasps_neg_freespace)
        grasps = torch.vstack([grasps, grasps_neg_freespace]).float()
        grasp_ids = torch.vstack([grasp_ids, grasps_id_neg_freespace]).int()

    if num_neg_hnretract > 0:
        # Step 4 - Add the hard negatives retraction
        grasps_neg_retract = generate_negative_retract(
            num_neg_hnretract, grasps_positive
        )
        grasps_id_neg_retract = (
            torch.ones([num_neg_hnretract, 1]) * MAPPING_NAME2ID["neg_hnretract"]
        )
        grasps_neg_retract = torch.from_numpy(grasps_neg_retract)

        # NOTE - the labels are already correct from the get-go
        grasps = torch.vstack([grasps, grasps_neg_retract]).float()
        grasp_ids = torch.vstack([grasp_ids, grasps_id_neg_retract]).int()

    if include_true_neg:
        # Step 5 - Add the true negatives from the dataset
        negative_grasps = grasps_negative
        grasps_neg_true = negative_grasps[
            np.random.randint(0, len(negative_grasps), num_neg_true_grasps)
        ]
        grasps_neg_true = torch.from_numpy(grasps_neg_true)
        grasps_id_neg_true = (
            torch.ones([num_neg_true_grasps, 1]) * MAPPING_NAME2ID["neg_true"]
        )

        # NOTE - the labels are already correct from the get-go
        grasps = torch.vstack([grasps, grasps_neg_true]).float()
        grasp_ids = torch.vstack([grasp_ids, grasps_id_neg_true]).int()

    # Step 2 - Add the hard negatives
    grasps_neg_hncolliding = generate_negative_hardnegatives(
        num_neg_hncolliding, grasps_positive, scene_mesh, gripper_mesh
    )

    if grasps_neg_hncolliding is not None:

        num_neg_hncolliding_actual = len(grasps_neg_hncolliding)
        grasps_neg_hncolliding = torch.from_numpy(grasps_neg_hncolliding)
        grasps_id_neg_hncolliding = (
            torch.ones([num_neg_hncolliding_actual, 1])
            * MAPPING_NAME2ID["neg_hncolliding"]
        )

        # NOTE - the labels are already correct from the get-go
        grasps = torch.vstack([grasps, grasps_neg_hncolliding]).float()
        grasp_ids = torch.vstack([grasp_ids, grasps_id_neg_hncolliding]).int()
    else:
        num_neg_hncolliding_actual = 0

    num_total_actual = (
        num_pos_true_grasps
        + num_neg_true_grasps
        + num_neg_hncolliding_actual
        + num_neg_freespace
        + num_neg_hnretract
        + num_pos_true_onpolicy_grasps
        + num_neg_true_onpolicy_grasps
    )
    num_total_pos = num_pos_true_grasps + num_pos_true_onpolicy_grasps
    labels = torch.vstack(
        [
            torch.ones([num_total_pos, 1]),
            torch.zeros([num_total_actual - num_total_pos, 1]),
        ]
    ).float()

    assert grasps.shape[0] == num_total_actual
    assert labels.shape[0] == num_total_actual
    assert grasp_ids.shape[0] == num_total_actual

    outputs = {"grasps": grasps, "labels": labels, "grasp_ids": grasp_ids}
    return outputs, scene_mesh


def collate_batch_keys(batch):
    if len(batch) < 1:
        return
    batch = {key: [data[key] for data in batch] for key in batch[0]}
    if "task" in batch:
        task = batch.pop("task")
        batch["task_is_pick"] = torch.stack([torch.tensor(t == "pick") for t in task])
        batch["task_is_place"] = torch.stack([torch.tensor(t == "place") for t in task])
    for key in batch:
        if key in [
            "inputs",
            "points",
            "seg",
            "object_inputs",
            "bottom_center",
            "cam_pose",
            "ee_pose",
            "placement_masks",
            "placement_region",
        ]:
            batch[key] = torch.stack(batch[key])
        if key in ["contact_dirs", "approach_dirs", "offsets"]:
            batch[key] = torch.cat(batch[key])
    return batch


def collate(batch):
    initial_batch_size = len(batch)
    batch = [data for data in batch if not data.get("invalid", False)]
    final_batch_size = len(batch)
    return collate_batch_keys(batch)
