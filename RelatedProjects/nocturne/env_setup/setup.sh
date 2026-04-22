#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="nocturne:latest"
CONTAINER_NAME="nocturne-container"

docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -it \
    --name "$CONTAINER_NAME" \
    --gpus all \
    -v "$SCRIPT_DIR:/workspace" \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    "$IMAGE_NAME" \
    bash -c "cd /app/nocturne && python examples/create_env.py"