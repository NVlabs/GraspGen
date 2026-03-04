# GraspGen Standalone Server

GraspGen can be run as a standalone ZMQ server so that any application — on the same machine or across the network — can request 6-DOF grasp predictions without importing the model code or needing a GPU.

```
┌──────────────────────┐         ZMQ (tcp)         ┌──────────────────────┐
│   Client (any lang)  │  ──── point cloud / mesh ──────▶  │  GraspGen Server     │
│   - Python / C++ / … │  ◀── grasps + scores ───  │  - GPU, model loaded  │
│   - No CUDA needed   │                           │  - Runs in Docker     │
└──────────────────────┘                           └──────────────────────┘
```

The server loads a gripper model (Franka Panda, Robotiq 2F-140, or Single Suction Cup 30mm) and listens on a ZMQ REP socket. Clients send point clouds (as numpy arrays serialized with msgpack) and receive back 6-DOF grasp poses and confidence scores.

## With Docker (recommended)

Terminal window 1 — **start the server**:

```bash
# Build the base Docker image (one-time):
bash docker/build.sh

# Start the server (default: Robotiq 2F-140 on port 5556):
MODELS_DIR=/path/to/GraspGenModels docker compose -f docker/compose.serve.yml up --build

# Or with a custom gripper and port:
MODELS_DIR=/path/to/GraspGenModels \
SERVER_ARGS="--gripper_config /models/checkpoints/graspgen_franka_panda.yml --port 5557" \
  docker compose -f docker/compose.serve.yml up --build
```

You can customize the loaded gripper by providing `SERVER_ARGS` (see `client-server/graspgen_server.py --help`). Available gripper configs in the checkpoints directory:
- `graspgen_robotiq_2f_140.yml` (default)
- `graspgen_franka_panda.yml`
- `graspgen_single_suction_cup_30mm.yml`

Terminal window 2 — **run the client** (lightweight uv environment, no CUDA needed):

```bash
# Create a client environment (one-time):
uv venv --python 3.10 client-server/.venv
source client-server/.venv/bin/activate
uv pip install pyzmq msgpack msgpack-numpy numpy trimesh
uv pip install -e . --no-deps

# Run the client with a mesh file:
python client-server/graspgen_client.py \
    --mesh_file /path/to/GraspGenModels/sample_data/meshes/box.obj \
    --mesh_scale 1.0 \
    --host localhost --port 5556

# Or with a point cloud file (.pcd / .ply / .xyz / .npy):
python client-server/graspgen_client.py \
    --pcd_file assets/objects/example_object.pcd \
    --host localhost --port 5556
```

## Without Docker

Terminal window 1 — **start the server**:

```bash
# Activate your GraspGen environment (must have CUDA + all GraspGen dependencies):
conda activate GraspGen   # or source .venv/bin/activate

# Install serving dependencies:
pip install pyzmq msgpack msgpack-numpy

# Start the server:
python client-server/graspgen_server.py \
    --gripper_config /path/to/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml \
    --port 5556
```

Terminal window 2 — **run the client**:

```bash
# Create a client environment (one-time):
uv venv --python 3.10 client-server/.venv
source client-server/.venv/bin/activate
uv pip install pyzmq msgpack msgpack-numpy numpy trimesh
uv pip install -e . --no-deps

# Run the client with a mesh file:
python client-server/graspgen_client.py \
    --mesh_file /path/to/GraspGenModels/sample_data/meshes/box.obj \
    --mesh_scale 1.0 \
    --host localhost --port 5556

# Or with a point cloud file:
python client-server/graspgen_client.py \
    --pcd_file assets/objects/example_object.pcd \
    --host localhost --port 5556
```

## Python Client API

The client only requires `pyzmq`, `msgpack`, `msgpack-numpy`, and `numpy` — no PyTorch or CUDA.

```python
from grasp_gen.serving.zmq_client import GraspGenClient

client = GraspGenClient(host="localhost", port=5556)

# Get server info
print(client.server_metadata)
# {'gripper_name': 'robotiq_2f_140', 'model_name': 'diffusion-discriminator', ...}

# Run inference
grasps, confidences = client.infer(
    point_cloud,          # (N, 3) numpy float32 array
    num_grasps=200,       # diffusion samples
    topk_num_grasps=100,  # return top-k by confidence
)
# grasps:       (M, 4, 4) float32 — 6-DOF grasp poses
# confidences:  (M,)      float32 — grasp quality scores [0, 1]

client.close()
```

## Protocol Reference

The server uses **msgpack** serialization over a **ZMQ REP** socket.

| Request | Fields | Response |
|---------|--------|----------|
| `{"action": "health"}` | — | `{"status": "ok"}` |
| `{"action": "metadata"}` | — | `{"gripper_name": ..., "model_name": ..., ...}` |
| `{"action": "infer", "point_cloud": ndarray, ...}` | `grasp_threshold`, `num_grasps`, `topk_num_grasps`, `min_grasps`, `max_tries`, `remove_outliers` | `{"grasps": ndarray, "confidences": ndarray, "num_grasps": int, "timing": {...}}` |

This makes it straightforward to write clients in any language with ZMQ and msgpack bindings (C++, Rust, etc.).
