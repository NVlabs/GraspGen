#!/bin/bash

# Function to display usage information
show_usage() {
    echo "Usage: $0 <grasp_gen_code_dir> [--models <models_dir>] [--grasp_dataset <grasp_dataset_dir> --object_dataset <object_dataset_dir> --results <results_dir>]"
    echo ""
    echo "Arguments:"
    echo "  grasp_gen_code_dir  (required) Path to the GraspGen code repository (relative or absolute)"
    echo "  --models <dir>      (optional) Path to the models directory (for inference, relative or absolute)"
    echo "  --grasp_dataset <dir> (optional) Path to the grasp dataset directory (for training, relative or absolute)"
    echo "  --object_dataset <dir> (optional) Path to the object dataset directory (for training, relative or absolute)"
    echo "  --results <dir>     (optional) Path to the results directory (for training outputs, relative or absolute)"
    echo ""
    echo "Examples:"
    echo "  # For inference demo (with models):"
    echo "  $0 /path/to/graspgen/code --models /path/to/models"
    echo "  $0 ./graspgen --models ./models"
    echo ""
    echo "  # For training (with grasp dataset, object dataset, and results):"
    echo "  $0 /path/to/graspgen/code --grasp_dataset /path/to/grasp_dataset --object_dataset /path/to/object_dataset --results /path/to/results"
    echo "  $0 ./graspgen --grasp_dataset ./datasets/grasp --object_dataset ./datasets/objects --results ./output"
    echo ""
    echo "Note: grasp_gen_code_dir is always mounted at /code"
    echo "      models_dir (if provided) is mounted at /models"
    echo "      grasp_dataset_dir (if provided) is mounted at /grasp_dataset"
    echo "      object_dataset_dir (if provided) is mounted at /object_dataset"
    echo "      results_dir (if provided) is mounted at /results"
    echo "      All paths are converted to absolute paths for Docker volume mounting"
    exit 1
}

# Function to convert path to absolute path
make_absolute_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        # Already absolute path
        echo "$path"
    else
        # Relative path, convert to absolute
        echo "$(realpath "$path")"
    fi
}

# Check if at least the required argument is provided
if [ $# -lt 1 ]; then
    show_usage
fi

GRASP_GEN_CODE_DIR="$(make_absolute_path "$1")"
MODELS_DIR=""
GRASP_DATASET_DIR=""
OBJECT_DATASET_DIR=""
RESULTS_DIR=""

# Parse optional arguments
shift  # Remove the first argument (grasp_gen_code_dir)
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS_DIR="$(make_absolute_path "$2")"
            shift 2
            ;;
        --grasp_dataset)
            GRASP_DATASET_DIR="$(make_absolute_path "$2")"
            shift 2
            ;;
        --object_dataset)
            OBJECT_DATASET_DIR="$(make_absolute_path "$2")"
            shift 2
            ;;
        --results)
            RESULTS_DIR="$(make_absolute_path "$2")"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Validate that the required directory exists
if [ ! -d "$GRASP_GEN_CODE_DIR" ]; then
    echo "Error: grasp_gen_code_dir '$GRASP_GEN_CODE_DIR' does not exist or is not a directory"
    exit 1
fi

# Build volume mount arguments
VOLUME_MOUNTS="-v ${GRASP_GEN_CODE_DIR}:/code"

# Add models directory if provided
if [ -n "$MODELS_DIR" ]; then
    if [ ! -d "$MODELS_DIR" ]; then
        echo "Error: models_dir '$MODELS_DIR' does not exist or is not a directory"
        exit 1
    fi
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v ${MODELS_DIR}:/models"
fi

# Add grasp dataset directory if provided
if [ -n "$GRASP_DATASET_DIR" ]; then
    if [ ! -d "$GRASP_DATASET_DIR" ]; then
        echo "Error: grasp_dataset_dir '$GRASP_DATASET_DIR' does not exist or is not a directory"
        exit 1
    fi
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v ${GRASP_DATASET_DIR}:/grasp_dataset"
fi

# Add object dataset directory if provided
if [ -n "$OBJECT_DATASET_DIR" ]; then
    if [ ! -d "$OBJECT_DATASET_DIR" ]; then
        echo "Error: object_dataset_dir '$OBJECT_DATASET_DIR' does not exist or is not a directory"
        exit 1
    fi
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v ${OBJECT_DATASET_DIR}:/object_dataset"
fi

# Add results directory if provided
if [ -n "$RESULTS_DIR" ]; then
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "Creating results directory: $RESULTS_DIR"
        mkdir -p "$RESULTS_DIR"
    fi
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v ${RESULTS_DIR}:/results"
fi

echo "Starting Docker container with:"
echo "  Code directory: $GRASP_GEN_CODE_DIR -> /code"
if [ -n "$MODELS_DIR" ]; then
    echo "  Models directory: $MODELS_DIR -> /models (for inference)"
fi
if [ -n "$GRASP_DATASET_DIR" ]; then
    echo "  Grasp dataset directory: $GRASP_DATASET_DIR -> /grasp_dataset (for training)"
fi
if [ -n "$OBJECT_DATASET_DIR" ]; then
    echo "  Object dataset directory: $OBJECT_DATASET_DIR -> /object_dataset (for training)"
fi
if [ -n "$RESULTS_DIR" ]; then
    echo "  Results directory: $RESULTS_DIR -> /results (for training outputs)"
fi
echo ""

xhost +local:root
docker run \
  --privileged \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --device /dev/dri \
    --device /dev/bus/usb:/dev/bus/usb \
    --device-cgroup-rule='c 189:* rmw' \
    -v /run/udev:/run/udev:ro \
  -it \
  -e DISPLAY \
  $VOLUME_MOUNTS \
  --gpus all \
  --net host \
  --shm-size 40G \
  graspgen:latest \
  /bin/bash \
    -c "cd /code/ && pip install -e . && bash"
xhost -local:root
