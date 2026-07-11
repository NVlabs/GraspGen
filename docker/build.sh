#!/bin/bash
set -euo pipefail

VER=1.0

BUILD_ARGS=()
if [[ -n "${PYORBBECSDK2_WHL:-}" ]]; then
	BUILD_ARGS+=(--build-arg "PYORBBECSDK2_WHL=${PYORBBECSDK2_WHL}")
	echo "Using PYORBBECSDK2_WHL=${PYORBBECSDK2_WHL}"
else
	echo "PYORBBECSDK2_WHL not set; building without pyorbbecsdk2 preinstall"
fi

docker build -f docker/graspgen_cuda121.dockerfile --progress=plain . --network=host "${BUILD_ARGS[@]}" -t graspgen:$VER -t graspgen:latest
