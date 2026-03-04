"""Tests for the GraspGen ZMQ server/client architecture.

Three test categories:
  1. Unit tests for client-side point cloud loading (no server needed).
  2. Protocol tests using a lightweight mock ZMQ server in a thread.
  3. Integration test against a live server (skipped unless --run-integration).
"""

import os
import threading
import time
import tempfile

import numpy as np
import pytest
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "objects")
EXAMPLE_PCD = os.path.join(ASSETS_DIR, "example_object.pcd")
EXAMPLE_MESH = os.path.join(ASSETS_DIR, "box.obj")


# ===================================================================
# 1. Unit tests — client-side point cloud loaders
# ===================================================================


class TestPCDReader:
    """Tests for the minimal ASCII PCD reader."""

    def test_read_example_pcd(self):
        from tools.graspgen_client import _read_pcd_ascii

        if not os.path.exists(EXAMPLE_PCD):
            pytest.skip("Example PCD not found")
        pts = _read_pcd_ascii(EXAMPLE_PCD)
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert pts.dtype == np.float32
        assert len(pts) > 100

    def test_read_synthetic_pcd(self, tmp_path):
        from tools.graspgen_client import _read_pcd_ascii

        pcd_file = tmp_path / "test.pcd"
        pcd_file.write_text(
            "# .PCD v0.7\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            "WIDTH 3\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            "POINTS 3\n"
            "DATA ascii\n"
            "1.0 2.0 3.0\n"
            "4.0 5.0 6.0\n"
            "7.0 8.0 9.0\n"
        )
        pts = _read_pcd_ascii(str(pcd_file))
        np.testing.assert_array_almost_equal(
            pts, [[1, 2, 3], [4, 5, 6], [7, 8, 9]], decimal=5
        )


class TestLoadPointCloudFromFile:
    """Tests for load_point_cloud_from_file across supported formats."""

    def test_load_pcd(self):
        from tools.graspgen_client import load_point_cloud_from_file

        if not os.path.exists(EXAMPLE_PCD):
            pytest.skip("Example PCD not found")
        pts = load_point_cloud_from_file(EXAMPLE_PCD)
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        # Centering: mean should be ~0
        np.testing.assert_array_almost_equal(pts.mean(axis=0), [0, 0, 0], decimal=4)

    def test_load_npy(self, tmp_path):
        from tools.graspgen_client import load_point_cloud_from_file

        arr = np.random.randn(500, 3).astype(np.float64)
        npy_file = tmp_path / "cloud.npy"
        np.save(str(npy_file), arr)
        pts = load_point_cloud_from_file(str(npy_file))
        assert pts.dtype == np.float32
        assert pts.shape == (500, 3)
        np.testing.assert_array_almost_equal(pts.mean(axis=0), [0, 0, 0], decimal=4)

    def test_load_xyz(self, tmp_path):
        from tools.graspgen_client import load_point_cloud_from_file

        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        xyz_file = tmp_path / "cloud.xyz"
        np.savetxt(str(xyz_file), arr)
        pts = load_point_cloud_from_file(str(xyz_file))
        assert pts.shape == (3, 3)
        np.testing.assert_array_almost_equal(pts.mean(axis=0), [0, 0, 0], decimal=4)

    def test_unsupported_format_raises(self, tmp_path):
        from tools.graspgen_client import load_point_cloud_from_file

        bad = tmp_path / "cloud.csv"
        bad.write_text("1,2,3\n")
        with pytest.raises(ValueError, match="Unsupported"):
            load_point_cloud_from_file(str(bad))

    def test_bad_shape_raises(self, tmp_path):
        from tools.graspgen_client import load_point_cloud_from_file

        arr = np.random.randn(10, 2).astype(np.float32)
        npy_file = tmp_path / "bad.npy"
        np.save(str(npy_file), arr)
        with pytest.raises(ValueError, match="Expected"):
            load_point_cloud_from_file(str(npy_file))


class TestLoadPointCloudFromMesh:
    def test_load_box_mesh(self):
        from tools.graspgen_client import load_point_cloud_from_mesh

        if not os.path.exists(EXAMPLE_MESH):
            pytest.skip("Example mesh not found")
        pts = load_point_cloud_from_mesh(EXAMPLE_MESH, scale=1.0, num_points=500)
        assert pts.shape == (500, 3)
        assert pts.dtype == np.float32
        np.testing.assert_array_almost_equal(pts.mean(axis=0), [0, 0, 0], decimal=3)

    def test_mesh_scale(self):
        from tools.graspgen_client import load_point_cloud_from_mesh

        if not os.path.exists(EXAMPLE_MESH):
            pytest.skip("Example mesh not found")
        pts1 = load_point_cloud_from_mesh(EXAMPLE_MESH, scale=1.0, num_points=1000)
        pts2 = load_point_cloud_from_mesh(EXAMPLE_MESH, scale=2.0, num_points=1000)
        extent1 = pts1.max(axis=0) - pts1.min(axis=0)
        extent2 = pts2.max(axis=0) - pts2.min(axis=0)
        # Scaled mesh should have ~2x extent
        ratio = extent2 / (extent1 + 1e-8)
        assert np.all(ratio > 1.5), f"Scale ratio unexpectedly low: {ratio}"


# ===================================================================
# 2. Mock ZMQ server tests — protocol correctness
# ===================================================================


def _make_fake_grasps(n: int):
    """Return fake (n, 4, 4) grasps and (n,) confidences."""
    grasps = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    for i in range(n):
        grasps[i, :3, 3] = np.random.randn(3).astype(np.float32) * 0.1
    confs = np.linspace(0.3, 0.95, n).astype(np.float32)
    return grasps, confs


class MockGraspGenServer:
    """Lightweight ZMQ REP server for testing the client protocol."""

    def __init__(self, port: int, num_fake_grasps: int = 10):
        self._port = port
        self._num_fake_grasps = num_fake_grasps
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REP)
        self._socket.bind(f"tcp://127.0.0.1:{port}")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            socks = dict(poller.poll(timeout=200))
            if self._socket in socks:
                raw = self._socket.recv()
                req = msgpack.unpackb(raw, raw=False)
                resp = self._handle(req)
                self._socket.send(msgpack.packb(resp, use_bin_type=True))

    def _handle(self, req: dict) -> dict:
        action = req.get("action")
        if action == "health":
            return {"status": "ok"}
        if action == "metadata":
            return {
                "gripper_name": "mock_gripper",
                "model_name": "mock_model",
                "gripper_config": "/mock/config.yml",
            }
        if action == "infer":
            pc = np.asarray(req.get("point_cloud", []), dtype=np.float32)
            if pc.ndim != 2 or pc.shape[1] != 3:
                return {"error": f"point_cloud must be (N, 3), got {pc.shape}"}
            n = min(self._num_fake_grasps, int(req.get("topk_num_grasps", self._num_fake_grasps)))
            if n <= 0:
                n = self._num_fake_grasps
            grasps, confs = _make_fake_grasps(n)
            return {
                "grasps": grasps,
                "confidences": confs,
                "num_grasps": n,
                "timing": {"infer_ms": 42.0},
            }
        return {"error": f"Unknown action: {action}"}

    def stop(self):
        self._running = False
        self._thread.join(timeout=3)
        self._socket.close()
        self._ctx.term()


@pytest.fixture(scope="module")
def mock_server():
    """Start a mock ZMQ server on a random port for the test module."""
    port = 15556 + os.getpid() % 1000
    server = MockGraspGenServer(port=port, num_fake_grasps=10)
    time.sleep(0.3)
    yield port
    server.stop()


class TestClientProtocol:
    """Test the GraspGenClient against the mock server."""

    def test_health_check(self, mock_server):
        from grasp_gen.serving.zmq_client import GraspGenClient

        client = GraspGenClient(
            host="127.0.0.1", port=mock_server,
            wait_for_server=False, timeout_ms=5000,
        )
        assert client.health_check() is True
        client.close()

    def test_get_metadata(self, mock_server):
        from grasp_gen.serving.zmq_client import GraspGenClient

        client = GraspGenClient(
            host="127.0.0.1", port=mock_server,
            wait_for_server=False, timeout_ms=5000,
        )
        meta = client.get_metadata()
        assert meta["gripper_name"] == "mock_gripper"
        assert meta["model_name"] == "mock_model"
        client.close()

    def test_infer_returns_correct_shapes(self, mock_server):
        from grasp_gen.serving.zmq_client import GraspGenClient

        client = GraspGenClient(
            host="127.0.0.1", port=mock_server,
            wait_for_server=False, timeout_ms=5000,
        )
        pc = np.random.randn(500, 3).astype(np.float32)
        grasps, confs = client.infer(pc, num_grasps=50, topk_num_grasps=10)
        assert grasps.ndim == 3
        assert grasps.shape[1:] == (4, 4)
        assert grasps.dtype == np.float32
        assert confs.ndim == 1
        assert len(confs) == len(grasps)
        assert confs.dtype == np.float32
        client.close()

    def test_infer_validates_bad_input(self, mock_server):
        from grasp_gen.serving.zmq_client import GraspGenClient

        client = GraspGenClient(
            host="127.0.0.1", port=mock_server,
            wait_for_server=False, timeout_ms=5000,
        )
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            client.infer(np.random.randn(10, 2).astype(np.float32))
        client.close()

    def test_context_manager(self, mock_server):
        from grasp_gen.serving.zmq_client import GraspGenClient

        with GraspGenClient(
            host="127.0.0.1", port=mock_server,
            wait_for_server=False, timeout_ms=5000,
        ) as client:
            assert client.health_check() is True

    def test_wait_for_server_connects(self, mock_server):
        from grasp_gen.serving.zmq_client import GraspGenClient

        client = GraspGenClient(
            host="127.0.0.1", port=mock_server,
            wait_for_server=True, timeout_ms=5000,
        )
        assert client.server_metadata is not None
        assert client.server_metadata["gripper_name"] == "mock_gripper"
        client.close()

    def test_multiple_infer_calls(self, mock_server):
        """Verify the REQ/REP pattern works for sequential requests."""
        from grasp_gen.serving.zmq_client import GraspGenClient

        with GraspGenClient(
            host="127.0.0.1", port=mock_server,
            wait_for_server=False, timeout_ms=5000,
        ) as client:
            for _ in range(5):
                pc = np.random.randn(200, 3).astype(np.float32)
                grasps, confs = client.infer(pc, topk_num_grasps=5)
                assert len(grasps) == 5
                assert len(confs) == 5


# ===================================================================
# 3. Integration test — live server
# ===================================================================


def _live_server_available(host: str = "localhost", port: int = 5556) -> bool:
    """Quick probe: can we connect and get a health response?"""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.setsockopt(zmq.SNDTIMEO, 2000)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(f"tcp://{host}:{port}")
        sock.send(msgpack.packb({"action": "health"}, use_bin_type=True))
        resp = msgpack.unpackb(sock.recv(), raw=False)
        return resp.get("status") == "ok"
    except Exception:
        return False
    finally:
        sock.close()
        ctx.term()


@pytest.mark.integration
class TestLiveServer:
    """Tests that require a running GraspGen server on localhost:5556.

    Run with: pytest tests/test_serving.py -m integration
    These are skipped by default unless the server is reachable.
    """

    @pytest.fixture(autouse=True)
    def _require_live_server(self):
        if not _live_server_available():
            pytest.skip("Live GraspGen server not available on localhost:5556")

    def test_live_health(self):
        from grasp_gen.serving.zmq_client import GraspGenClient

        with GraspGenClient(host="localhost", port=5556, wait_for_server=False, timeout_ms=10000) as client:
            assert client.health_check() is True

    def test_live_metadata(self):
        from grasp_gen.serving.zmq_client import GraspGenClient

        with GraspGenClient(host="localhost", port=5556, wait_for_server=True, timeout_ms=10000) as client:
            meta = client.server_metadata
            assert "gripper_name" in meta
            assert "model_name" in meta

    def test_live_infer_random_cloud(self):
        from grasp_gen.serving.zmq_client import GraspGenClient

        pc = np.random.randn(2000, 3).astype(np.float32) * 0.05
        with GraspGenClient(host="localhost", port=5556, wait_for_server=True, timeout_ms=60000) as client:
            grasps, confs = client.infer(
                pc, num_grasps=50, topk_num_grasps=10,
            )
            assert grasps.ndim == 3
            assert grasps.shape[1:] == (4, 4)
            assert len(confs) == len(grasps)

    def test_live_infer_example_pcd(self):
        from grasp_gen.serving.zmq_client import GraspGenClient
        from tools.graspgen_client import load_point_cloud_from_file

        if not os.path.exists(EXAMPLE_PCD):
            pytest.skip("Example PCD not found")

        pc = load_point_cloud_from_file(EXAMPLE_PCD)
        with GraspGenClient(host="localhost", port=5556, wait_for_server=True, timeout_ms=60000) as client:
            grasps, confs = client.infer(
                pc, num_grasps=100, topk_num_grasps=20,
            )
            assert grasps.shape[1:] == (4, 4)
            assert len(confs) == len(grasps)
            if len(confs) > 0:
                assert confs.min() >= 0.0
                assert confs.max() <= 1.0

    def test_live_infer_example_mesh(self):
        from grasp_gen.serving.zmq_client import GraspGenClient
        from tools.graspgen_client import load_point_cloud_from_mesh

        if not os.path.exists(EXAMPLE_MESH):
            pytest.skip("Example mesh not found")

        pc = load_point_cloud_from_mesh(EXAMPLE_MESH, scale=1.0, num_points=2000)
        with GraspGenClient(host="localhost", port=5556, wait_for_server=True, timeout_ms=60000) as client:
            grasps, confs = client.infer(
                pc, num_grasps=100, topk_num_grasps=20,
            )
            assert grasps.shape[1:] == (4, 4)
            assert len(confs) == len(grasps)
