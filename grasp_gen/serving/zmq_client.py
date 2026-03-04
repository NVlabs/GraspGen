# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Lightweight ZMQ client for GraspGen server.

Only depends on pyzmq, msgpack, msgpack-numpy, and numpy — no torch / CUDA needed.
This makes it suitable for running on robot controllers or edge devices.

Usage:
    from grasp_gen.serving.zmq_client import GraspGenClient

    client = GraspGenClient("localhost", 5556)
    grasps, confidences = client.infer(point_cloud)
"""

import logging
import time
from typing import Optional

import numpy as np
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

logger = logging.getLogger(__name__)


class GraspGenClient:
    """Client that connects to a GraspGen ZMQ server for remote grasp inference."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        timeout_ms: int = 60_000,
        wait_for_server: bool = True,
        retry_interval_s: float = 2.0,
    ) -> None:
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context()
        self._socket: Optional[zmq.Socket] = None
        self._server_metadata: Optional[dict] = None

        if wait_for_server:
            self._wait_for_server(retry_interval_s)

    def _create_socket(self) -> zmq.Socket:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._addr)
        return sock

    def _wait_for_server(self, retry_interval_s: float) -> None:
        logger.info("Waiting for GraspGen server at %s ...", self._addr)
        while True:
            try:
                self._socket = self._create_socket()
                self._server_metadata = self._request({"action": "metadata"})
                logger.info(
                    "Connected to GraspGen server: %s", self._server_metadata
                )
                return
            except (zmq.error.Again, zmq.error.ZMQError):
                logger.info("Server not ready, retrying in %.1fs ...", retry_interval_s)
                if self._socket is not None:
                    self._socket.close()
                    self._socket = None
                time.sleep(retry_interval_s)

    def _ensure_connected(self) -> None:
        if self._socket is None:
            self._socket = self._create_socket()

    def _request(self, payload: dict) -> dict:
        self._ensure_connected()
        self._socket.send(msgpack.packb(payload, use_bin_type=True))
        raw = self._socket.recv()
        response = msgpack.unpackb(raw, raw=False)
        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    @property
    def server_metadata(self) -> Optional[dict]:
        return self._server_metadata

    def health_check(self) -> bool:
        try:
            resp = self._request({"action": "health"})
            return resp.get("status") == "ok"
        except Exception:
            return False

    def get_metadata(self) -> dict:
        return self._request({"action": "metadata"})

    def infer(
        self,
        point_cloud: np.ndarray,
        *,
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = -1,
        min_grasps: int = 40,
        max_tries: int = 6,
        remove_outliers: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Send a point cloud to the server and receive grasp predictions.

        Args:
            point_cloud: (N, 3) float32 array of object points.
            grasp_threshold: Min confidence to keep. -1.0 returns top-k instead.
            num_grasps: Number of grasps the diffusion model should sample.
            topk_num_grasps: Return only top-k grasps (-1 = use threshold).
            min_grasps: Minimum grasps before retrying.
            max_tries: Max inference retries on the server.
            remove_outliers: Whether to filter point cloud outliers.

        Returns:
            grasps: (M, 4, 4) float32 array of 6-DOF grasp poses.
            confidences: (M,) float32 array of grasp confidence scores.
        """
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError(f"point_cloud must be (N, 3), got {point_cloud.shape}")

        payload = {
            "action": "infer",
            "point_cloud": point_cloud,
            "grasp_threshold": grasp_threshold,
            "num_grasps": num_grasps,
            "topk_num_grasps": topk_num_grasps,
            "min_grasps": min_grasps,
            "max_tries": max_tries,
            "remove_outliers": remove_outliers,
        }

        response = self._request(payload)
        grasps = np.asarray(response["grasps"], dtype=np.float32)
        confidences = np.asarray(response["confidences"], dtype=np.float32)
        return grasps, confidences

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
