import argparse
import os
import time

import numpy as np
import torch
import trimesh.transformations as tra

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    make_frame,
    visualize_grasp,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import (
    depth_and_segmentation_to_point_clouds,
    filter_colliding_grasps,
    point_cloud_outlier_removal,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture one frame from Orbbec Gemini 2 and run GraspGen inference"
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        required=True,
        help="Path to checkpoint config yml (e.g. /models/checkpoints/graspgen_robotiq_2f_140.yml)",
    )
    parser.add_argument(
        "--segmentation_mask_path",
        type=str,
        default="",
        help="Optional path to segmentation mask (.npy or image). If empty, auto-mask from depth is used.",
    )
    parser.add_argument(
        "--target_object_id",
        type=int,
        default=1,
        help="Object id in segmentation mask used as grasp target",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasp candidates to generate",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=-1.0,
        help="Confidence threshold; use -1 for top-k ranking behavior",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=100,
        help="If threshold is -1, return top-k grasps",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Retained for backward compatibility; top-k mode is enabled by default when threshold is -1",
    )
    parser.add_argument(
        "--collision_filter",
        action="store_true",
        help="Filter colliding grasps against scene point cloud",
    )
    parser.add_argument(
        "--collision_threshold",
        type=float,
        default=0.02,
        help="Collision distance threshold (meters)",
    )
    parser.add_argument(
        "--max_scene_points",
        type=int,
        default=8192,
        help="Downsample size for scene collision filtering",
    )
    parser.add_argument(
        "--max_object_points",
        type=int,
        default=12000,
        help="Maximum object points before outlier removal/inference (downsampled randomly)",
    )
    parser.add_argument(
        "--scene_point_size",
        type=float,
        default=0.008,
        help="Point size for scene visualization in meshcat",
    )
    parser.add_argument(
        "--object_point_size",
        type=float,
        default=0.012,
        help="Point size for object visualization in meshcat",
    )
    parser.add_argument(
        "--auto_mask_depth_delta",
        type=float,
        default=0.08,
        help="Depth window (+/- meters) around center depth for auto mask",
    )
    parser.add_argument(
        "--auto_mask_min_pixels",
        type=int,
        default=800,
        help="Minimum pixels required for auto mask",
    )
    parser.add_argument(
        "--save_capture_prefix",
        type=str,
        default="",
        help="Optional prefix path to save captured depth/rgb/mask as .npy",
    )
    parser.add_argument(
        "--keypress",
        action="store_true",
        help="Interactive mode: press Enter to capture a snapshot, or type q to quit",
    )
    return parser.parse_args()


def _extract_intrinsics(profile, frame):
    fx = fy = cx = cy = None

    for obj in [frame, profile]:
        if obj is None:
            continue
        for method_name in ["get_intrinsic", "get_camera_intrinsic", "get_intrinsics"]:
            if not hasattr(obj, method_name):
                continue
            intr = getattr(obj, method_name)()
            for attr_name in ["fx", "focal_x"]:
                if hasattr(intr, attr_name):
                    fx = float(getattr(intr, attr_name))
                    break
            for attr_name in ["fy", "focal_y"]:
                if hasattr(intr, attr_name):
                    fy = float(getattr(intr, attr_name))
                    break
            for attr_name in ["cx", "ppx", "principal_x"]:
                if hasattr(intr, attr_name):
                    cx = float(getattr(intr, attr_name))
                    break
            for attr_name in ["cy", "ppy", "principal_y"]:
                if hasattr(intr, attr_name):
                    cy = float(getattr(intr, attr_name))
                    break
            if None not in [fx, fy, cx, cy]:
                return fx, fy, cx, cy

    raise RuntimeError(
        "Could not read camera intrinsics from Orbbec SDK objects. Pass intrinsics manually by editing script if your SDK exposes different APIs."
    )


def _to_rgb_array(color_frame, ob_format):
    from io import BytesIO
    from PIL import Image

    h, w = color_frame.get_height(), color_frame.get_width()
    raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    fmt = color_frame.get_format()
    fmt_name = str(fmt).upper()

    if hasattr(ob_format, "RGB") and fmt == ob_format.RGB:
        return raw.reshape(h, w, 3)
    if hasattr(ob_format, "BGR") and fmt == ob_format.BGR:
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()

    if (hasattr(ob_format, "MJPG") and fmt == ob_format.MJPG) or "MJPG" in fmt_name:
        with Image.open(BytesIO(raw.tobytes())) as img:
            return np.array(img.convert("RGB"))

    if (hasattr(ob_format, "YUYV") and fmt == ob_format.YUYV) or "YUYV" in fmt_name:
        import cv2

        yuyv = raw.reshape(h, w, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)

    if (hasattr(ob_format, "UYVY") and fmt == ob_format.UYVY) or "UYVY" in fmt_name:
        import cv2

        uyvy = raw.reshape(h, w, 2)
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2RGB_UYVY)

    expected_rgb_size = h * w * 3
    if raw.size == expected_rgb_size:
        return raw.reshape(h, w, 3)

    if raw.size == h * w * 2:
        import cv2

        yuyv = raw.reshape(h, w, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)

    raise RuntimeError(
        f"Unsupported color frame format {fmt} with buffer size {raw.size} for resolution {w}x{h}"
    )


def capture_orbbec_frame():
    try:
        from pyorbbecsdk import (
            Config,
            OBSensorType,
            OBFormat,
            Pipeline,
        )
    except Exception as error:
        raise RuntimeError(
            "pyorbbecsdk is not installed in this environment. Install it first for Orbbec capture."
        ) from error

    pipeline = Pipeline()
    config = Config()

    profile_list_depth = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list_depth.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    profile_list_color = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list_color.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    pipeline.start(config)

    try:
        frames = None
        for _ in range(40):
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if depth_frame is not None and color_frame is not None:
                break

        if frames is None:
            raise RuntimeError("Failed to receive frames from Orbbec pipeline")

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth_frame is None or color_frame is None:
            raise RuntimeError("Could not get both depth and color frames")

        depth_h, depth_w = depth_frame.get_height(), depth_frame.get_width()

        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            depth_h, depth_w
        )
        depth_scale = float(depth_frame.get_depth_scale())
        depth_m = depth_raw.astype(np.float32) * depth_scale

        valid_depth = depth_m[(depth_m > 0.05) & np.isfinite(depth_m)]
        if valid_depth.size > 0:
            median_depth = float(np.median(valid_depth))
            if median_depth > 20.0:
                depth_m = depth_m / 1000.0
                print(
                    f"Depth values appear to be in millimeters (median={median_depth:.3f}); converting to meters"
                )

        rgb = _to_rgb_array(color_frame, OBFormat)
        if rgb.shape[0] != depth_h or rgb.shape[1] != depth_w:
            from PIL import Image

            rgb = np.array(
                Image.fromarray(rgb).resize((depth_w, depth_h), resample=Image.BILINEAR)
            )

        fx, fy, cx, cy = _extract_intrinsics(depth_profile, depth_frame)
        return depth_m, rgb, (fx, fy, cx, cy)
    finally:
        pipeline.stop()


def load_mask(mask_path):
    if mask_path.endswith(".npy"):
        return np.load(mask_path)
    from PIL import Image

    return np.array(Image.open(mask_path))


def auto_mask_from_depth(depth_m, depth_delta, min_pixels):
    h, w = depth_m.shape
    cy, cx = h // 2, w // 2

    valid = (depth_m > 0.1) & np.isfinite(depth_m)
    if not valid.any():
        raise RuntimeError("No valid depth values found for auto mask")

    center_depth = depth_m[cy, cx]
    if not np.isfinite(center_depth) or center_depth <= 0.1:
        valid_depths = depth_m[valid]
        center_depth = np.percentile(valid_depths, 35)

    lower = max(0.1, center_depth - depth_delta)
    upper = center_depth + depth_delta
    foreground = valid & (depth_m >= lower) & (depth_m <= upper)

    if foreground.sum() < min_pixels:
        valid_depths = depth_m[valid]
        lo = np.percentile(valid_depths, 10)
        hi = np.percentile(valid_depths, 45)
        foreground = valid & (depth_m >= lo) & (depth_m <= hi)

    if foreground.sum() < min_pixels:
        raise RuntimeError(
            f"Auto mask too small ({foreground.sum()} pixels). Provide --segmentation_mask_path."
        )

    mask = np.zeros_like(depth_m, dtype=np.uint8)
    mask[foreground] = 1
    return mask


def process_point_cloud(pc, grasps, grasp_conf):
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    grasps[:, 3, 3] = 1
    t_center = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, t_center)
    grasps_centered = np.array([t_center @ np.array(g) for g in grasps.tolist()])
    return pc_centered, grasps_centered, scores, t_center


def run_snapshot(args, vis):
    vis.delete()

    print("Capturing one frame from Orbbec Gemini 2...")
    cap_start = time.time()
    depth_m, rgb, (fx, fy, cx, cy) = capture_orbbec_frame()
    print(
        f"Capture complete in {time.time() - cap_start:.2f}s | depth={depth_m.shape} rgb={rgb.shape} intrinsics={(fx, fy, cx, cy)}"
    )

    if args.segmentation_mask_path:
        segmentation_mask = load_mask(args.segmentation_mask_path)
        print(f"Loaded segmentation mask from {args.segmentation_mask_path}")
    else:
        segmentation_mask = auto_mask_from_depth(
            depth_m,
            depth_delta=args.auto_mask_depth_delta,
            min_pixels=args.auto_mask_min_pixels,
        )
        print("Generated segmentation mask automatically from depth")

    if segmentation_mask.shape != depth_m.shape:
        raise ValueError(
            f"Mask shape {segmentation_mask.shape} does not match depth shape {depth_m.shape}"
        )

    if args.save_capture_prefix:
        np.save(f"{args.save_capture_prefix}_depth.npy", depth_m)
        np.save(f"{args.save_capture_prefix}_rgb.npy", rgb)
        np.save(f"{args.save_capture_prefix}_mask.npy", segmentation_mask)
        print(f"Saved capture to prefix: {args.save_capture_prefix}")

    scene_pc, object_pc, scene_colors, object_colors = depth_and_segmentation_to_point_clouds(
        depth_image=depth_m,
        segmentation_mask=segmentation_mask,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rgb_image=rgb,
        target_object_id=args.target_object_id,
        remove_object_from_scene=True,
    )

    if len(object_pc) > args.max_object_points:
        keep_idx = np.random.choice(len(object_pc), args.max_object_points, replace=False)
        object_pc = object_pc[keep_idx]
        if object_colors is not None:
            object_colors = object_colors[keep_idx]
        print(
            f"Downsampled object point cloud to {len(object_pc)} points (max_object_points={args.max_object_points})"
        )

    object_pc_torch = torch.from_numpy(object_pc)
    pc_filtered, _ = point_cloud_outlier_removal(object_pc_torch)
    pc_filtered = pc_filtered.numpy()

    if len(pc_filtered) > 0:
        t_center = tra.translation_matrix(-pc_filtered.mean(axis=0))
        pc_centered = tra.transform_points(pc_filtered, t_center)
    else:
        t_center = np.eye(4)
        pc_centered = pc_filtered

    if scene_colors is None:
        scene_colors = np.tile(np.array([[120, 120, 120]], dtype=np.uint8), (len(scene_pc), 1))

    object_vis_colors = np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (len(pc_centered), 1))

    scene_centered = tra.transform_points(scene_pc, t_center)
    visualize_pointcloud(
        vis,
        "pc_scene",
        scene_centered,
        scene_colors,
        size=args.scene_point_size,
    )
    visualize_pointcloud(
        vis,
        "pc_obj",
        pc_centered,
        object_vis_colors,
        size=args.object_point_size,
    )

    if len(scene_centered) > 0:
        scene_min = scene_centered.min(axis=0)
        scene_max = scene_centered.max(axis=0)
        print(f"Scene bounds (centered): min={scene_min}, max={scene_max}")

    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    sampler = GraspGenSampler(grasp_cfg)

    grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
        pc_filtered,
        sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
    )

    if len(grasps_inferred) == 0:
        print("No grasps found. Point clouds are shown in the browser.")
        return

    grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
    grasps_inferred = grasps_inferred.cpu().numpy()
    pc_centered, grasps_centered, scores, t_center = process_point_cloud(
        pc_filtered, grasps_inferred, grasp_conf_inferred
    )

    if args.collision_filter:
        gripper_info = get_gripper_info(gripper_name)
        collision_mesh = gripper_info.collision_mesh
        if len(scene_centered) > args.max_scene_points:
            idx = np.random.choice(len(scene_centered), args.max_scene_points, replace=False)
            scene_for_collision = scene_centered[idx]
        else:
            scene_for_collision = scene_centered

        collision_free_mask = filter_colliding_grasps(
            scene_pc=scene_for_collision,
            grasp_poses=grasps_centered,
            gripper_collision_mesh=collision_mesh,
            collision_threshold=args.collision_threshold,
        )

        free_grasps = grasps_centered[collision_free_mask]
        colliding_grasps = grasps_centered[~collision_free_mask]
        free_scores = scores[collision_free_mask]

        for j, grasp in enumerate(free_grasps):
            visualize_grasp(
                vis,
                f"grasps/free/{j:03d}/grasp",
                grasp,
                color=free_scores[j],
                gripper_name=gripper_name,
                linewidth=1.5,
            )

        for j, grasp in enumerate(colliding_grasps[:40]):
            visualize_grasp(
                vis,
                f"grasps/colliding/{j:03d}/grasp",
                grasp,
                color=[255, 0, 0],
                gripper_name=gripper_name,
                linewidth=0.4,
            )

        print(
            f"Collision filter: {collision_free_mask.sum()}/{len(collision_free_mask)} grasps are collision-free"
        )
    else:
        for j, grasp in enumerate(grasps_centered):
            visualize_grasp(
                vis,
                f"grasps/{j:03d}/grasp",
                grasp,
                color=scores[j],
                gripper_name=gripper_name,
                linewidth=1.2,
            )

    print(
        f"Done. Inferred {len(grasps_inferred)} grasps with confidence range [{grasp_conf_inferred.min():.3f}, {grasp_conf_inferred.max():.3f}]"
    )


def main():
    args = parse_args()

    if not os.path.exists(args.gripper_config):
        raise FileNotFoundError(args.gripper_config)

    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    vis = create_visualizer()
    make_frame(vis, "world", h=0.12, radius=0.004)
    print("Meshcat visualization active.")
    print("If needed, run `meshcat-server` and open http://127.0.0.1:7000 in your browser.")

    if args.keypress:
        print("Keypress mode enabled: press Enter to capture a new snapshot, or type 'q' to quit.")
        while True:
            user_in = input("[Enter=Capture, q=Quit] > ").strip().lower()
            if user_in in ["q", "quit", "exit"]:
                print("Exiting keypress mode.")
                break
            try:
                run_snapshot(args, vis)
            except Exception as error:
                print(f"Snapshot failed: {error}")
    else:
        run_snapshot(args, vis)
        print("Visualizer running. Open http://127.0.0.1:7000 and keep this process alive.")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
