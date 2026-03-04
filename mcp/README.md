# GraspGen MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that enables LLMs (Claude, Cursor, etc.) to generate 6-DOF robotic grasp poses using [GraspGen](https://github.com/NVlabs/GraspGen).

```
┌─────────────────────┐        MCP (stdio)        ┌─────────────────────┐       ZMQ (tcp)       ┌─────────────────────┐
│  LLM / AI Agent     │  ◀─── tool calls ───────▶  │  MCP Server         │  ── point cloud ───▶  │  GraspGen Server    │
│  (Cursor, Claude…)  │                            │  (this package)     │  ◀── grasps ────────  │  (GPU, model loaded)│
└─────────────────────┘                            └─────────────────────┘                       └─────────────────────┘
```

The MCP server is a lightweight bridge — it requires no CUDA or model weights. It connects to a running GraspGen ZMQ inference server and exposes its capabilities as MCP tools that any LLM agent can call.

## Available Tools

| Tool | Description |
|------|-------------|
| `generate_grasps_from_mesh` | Generate 6-DOF grasp poses from a 3D mesh file (.obj, .stl, .ply, .glb). Samples a point cloud from the mesh surface and runs GraspGen inference. |
| `generate_grasps_from_point_cloud` | Generate 6-DOF grasp poses from a point cloud file (.npy, .npz, .ply, .pcd). |
| `visualize_grasps` | Generate grasps and visualize them interactively in a 3D [viser](https://viser.studio/) web viewer. Accepts a mesh or point cloud file. Grasps are color-coded by confidence (green=high, red=low). |
| `graspgen_health_check` | Check if the GraspGen inference server is running and responsive. |
| `graspgen_server_info` | Get metadata about the server: loaded gripper name, model config, etc. |

## Prerequisites

The GraspGen ZMQ server must be running. See the [client-server/README.md](../client-server/README.md) for setup instructions.

**Quick start (Docker):**

```bash
# From the GraspGen repo root:
MODELS_DIR=/path/to/GraspGenModels docker compose -f docker/compose.serve.yml up --build
```

**Quick start (local):**

```bash
python client-server/graspgen_server.py --gripper_config /path/to/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml
```

## Installation

### Using uv (recommended)

```bash
cd GraspGen/mcp
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e .
```

### Using pip

```bash
cd GraspGen/mcp
pip install -e .
```

## Configuration

### Configure for Cursor

Add the following to `.cursor/mcp.json` in your workspace (or to your global Cursor settings). Make sure to edit the `--directory` entry.

```json
{
  "mcpServers": {
    "graspgen": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/GraspGen/mcp",
        "mcp-server-graspgen"
      ],
      "env": {
        "GRASPGEN_HOST": "localhost",
        "GRASPGEN_PORT": "5556"
      }
    }
  }
}
```

Or if you installed with pip:

```json
{
  "mcpServers": {
    "graspgen": {
      "command": "python",
      "args": ["-m", "mcp_server_graspgen"],
      "env": {
        "GRASPGEN_HOST": "localhost",
        "GRASPGEN_PORT": "5556"
      }
    }
  }
}
```

### Configure for Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "graspgen": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/GraspGen/mcp",
        "mcp-server-graspgen"
      ],
      "env": {
        "GRASPGEN_HOST": "localhost",
        "GRASPGEN_PORT": "5556"
      }
    }
  }
}
```

### Configure for VS Code

Add to `.vscode/mcp.json` in your workspace:

```json
{
  "mcp": {
    "servers": {
      "graspgen": {
        "command": "uv",
        "args": [
          "run",
          "--directory", "/absolute/path/to/GraspGen/mcp",
          "mcp-server-graspgen"
        ],
        "env": {
          "GRASPGEN_HOST": "localhost",
          "GRASPGEN_PORT": "5556"
        }
      }
    }
  }
}
```

### Custom Server Address

If the GraspGen ZMQ server is on a different host or port, set the environment variables:

- `GRASPGEN_HOST` — default: `localhost`
- `GRASPGEN_PORT` — default: `5556`

Or pass them as CLI arguments:

```bash
mcp-server-graspgen --host 192.168.1.100 --port 5557
```

## Example LLM Interactions

Once configured, an LLM can naturally call GraspGen:

> **User:** "Generate grasps for the box mesh at `/models/sample_data/meshes/box.obj`"
>
> **LLM → `generate_grasps_from_mesh`:** `{"mesh_file": "/models/sample_data/meshes/box.obj", "mesh_scale": 1.0}`
>
> **Response:** "Generated 100 grasps. Confidence range: 0.7234 – 0.9812. Top grasp at position (0.012, -0.003, 0.045) with confidence 0.9812..."

> **User:** "Is the grasp server running?"
>
> **LLM → `graspgen_health_check`**
>
> **Response:** "GraspGen server status: ok"

## Debugging

Use the MCP inspector to test the server:

```bash
cd GraspGen/mcp
npx @modelcontextprotocol/inspector uv run mcp-server-graspgen
```
