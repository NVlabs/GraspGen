# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time
import logging
from typing import Optional

import numpy as np
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

logger = logging.getLogger(__name__)


class GraspGenZMQServer:
    """ZMQ server that wraps GraspGenSampler for remote grasp inference.

    Protocol (msgpack over ZMQ REP socket):
        Request:  {"action": "infer", "point_cloud": ndarray(N,3), ...params}
                  {"action": "metadata"}
                  {"action": "health"}
        Response: msgpack-encoded dict with results or error.
    """

    def __init__(
        self,
        gripper_config: str,
        host: str = "0.0.0.0",
        port: int = 5556,
    ) -> None:
        self._host = host
        self._port = port
        self._gripper_config = gripper_config

        logger.info("Loading gripper config from %s", gripper_config)
        self._cfg = load_grasp_cfg(gripper_config)
        self._gripper_name = self._cfg.data.gripper_name
        self._model_name = self._cfg.eval.model_name

        logger.info(
            "Initializing GraspGenSampler (model=%s, gripper=%s)",
            self._model_name,
            self._gripper_name,
        )
        self._sampler = GraspGenSampler(self._cfg)
        logger.info("Model loaded and ready for inference")

        self._metadata = {
            "gripper_name": self._gripper_name,
            "model_name": self._model_name,
            "gripper_config": gripper_config,
        }

    def serve_forever(self) -> None:
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        bind_addr = f"tcp://{self._host}:{self._port}"
        socket.bind(bind_addr)
        logger.info("GraspGen ZMQ server listening on %s", bind_addr)

        try:
            while True:
                raw = socket.recv()
                try:
                    request = msgpack.unpackb(raw, raw=False)
                    response = self._handle(request)
                except Exception as exc:
                    logger.exception("Error handling request")
                    response = {"error": str(exc)}
                socket.send(msgpack.packb(response, use_bin_type=True))
        except KeyboardInterrupt:
            logger.info("Shutting down server")
        finally:
            socket.close()
            ctx.term()

    def _handle(self, request: dict) -> dict:
        action = request.get("action")
        if action == "health":
            return {"status": "ok"}
        if action == "metadata":
            return self._metadata
        if action == "infer":
            return self._handle_infer(request)
        return {"error": f"Unknown action: {action}"}

    def _handle_infer(self, request: dict) -> dict:
        point_cloud = request.get("point_cloud")
        if point_cloud is None:
            return {"error": "Missing required field 'point_cloud'"}

        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            return {
                "error": f"point_cloud must be (N, 3), got {point_cloud.shape}"
            }

        params = {
            "grasp_threshold": float(request.get("grasp_threshold", -1.0)),
            "num_grasps": int(request.get("num_grasps", 200)),
            "topk_num_grasps": int(request.get("topk_num_grasps", -1)),
            "min_grasps": int(request.get("min_grasps", 40)),
            "max_tries": int(request.get("max_tries", 6)),
            "remove_outliers": bool(request.get("remove_outliers", True)),
        }

        t0 = time.monotonic()
        grasps, grasp_conf = GraspGenSampler.run_inference(
            point_cloud, self._sampler, **params
        )
        infer_ms = (time.monotonic() - t0) * 1000

        if len(grasps) == 0:
            return {
                "grasps": np.empty((0, 4, 4), dtype=np.float32),
                "confidences": np.empty((0,), dtype=np.float32),
                "num_grasps": 0,
                "timing": {"infer_ms": infer_ms},
            }

        grasps_np = grasps.cpu().numpy().astype(np.float32)
        conf_np = grasp_conf.cpu().numpy().astype(np.float32)

        logger.info(
            "Inferred %d grasps in %.1f ms (conf range %.3f - %.3f)",
            len(grasps_np),
            infer_ms,
            conf_np.min(),
            conf_np.max(),
        )

        return {
            "grasps": grasps_np,
            "confidences": conf_np,
            "num_grasps": len(grasps_np),
            "timing": {"infer_ms": infer_ms},
        }
