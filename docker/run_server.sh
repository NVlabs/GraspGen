#!/bin/bash

# Start the GraspGen ZMQ inference server in Docker.
#
# Usage:
#   bash docker/run_server.sh <graspgen_code_dir> --models <models_dir> [--gripper_config CONFIG] [--port PORT]
#
# Examples:
#   # Default (Robotiq 2F-140 on port 5556):
#   bash docker/run_server.sh $(pwd) --models /path/to/GraspGenModels
#
#   # Franka Panda on port 5557:
#   bash docker/run_server.sh $(pwd) --models /path/to/GraspGenModels \
#       --gripper_config /models/checkpoints/graspgen_franka_panda.yml --port 5557

set -e

show_usage() {
    echo "Usage: $0 <graspgen_code_dir> --models <models_dir> [--gripper_config CONFIG] [--port PORT]"
    echo ""
    echo "Arguments:"
    echo "  graspgen_code_dir        Path to the GraspGen code repository"
    echo "  --models <dir>           Path to GraspGenModels directory"
    echo "  --gripper_config <path>  Gripper config inside the container (default: /models/checkpoints/graspgen_robotiq_2f_140.yml)"
    echo "  --port <port>            ZMQ port (default: 5556)"
    exit 1
}

make_absolute_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then echo "$path"; else echo "$(realpath "$path")"; fi
}

if [ $# -lt 3 ]; then show_usage; fi

GRASPGEN_CODE_DIR="$(make_absolute_path "$1")"
shift

MODELS_DIR=""
GRIPPER_CONFIG="/models/checkpoints/graspgen_robotiq_2f_140.yml"
PORT=5556

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)         MODELS_DIR="$(make_absolute_path "$2")"; shift 2 ;;
        --gripper_config) GRIPPER_CONFIG="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        -h|--help)        show_usage ;;
        *)                echo "Unknown option: $1"; show_usage ;;
    esac
done

if [ -z "$MODELS_DIR" ]; then echo "Error: --models is required"; show_usage; fi
if [ ! -d "$GRASPGEN_CODE_DIR" ]; then echo "Error: $GRASPGEN_CODE_DIR not found"; exit 1; fi
if [ ! -d "$MODELS_DIR" ]; then echo "Error: $MODELS_DIR not found"; exit 1; fi

echo "Starting GraspGen ZMQ server in Docker ..."
echo "  Code   : $GRASPGEN_CODE_DIR -> /code"
echo "  Models : $MODELS_DIR -> /models"
echo "  Config : $GRIPPER_CONFIG"
echo "  Port   : $PORT"

docker run \
    --rm \
    --gpus all \
    --net host \
    --shm-size 8G \
    -e NVIDIA_DISABLE_REQUIRE=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v "${GRASPGEN_CODE_DIR}:/code" \
    -v "${MODELS_DIR}:/models" \
    graspgen:latest \
    /bin/bash -c "cd /code && pip install -q -e . && pip install -q pyzmq msgpack msgpack-numpy && python tools/graspgen_server.py --gripper_config ${GRIPPER_CONFIG} --port ${PORT}"
