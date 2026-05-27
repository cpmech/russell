#!/bin/bash

set -euo pipefail

# image name
NAME="cpmech/russell_ubuntu"
DOCKERFILE="zdocker/Dockerfile.Ubuntu"

# build Docker image
docker build -f "$DOCKERFILE" -t "$NAME" .

echo
echo "... SUCCESS: image ${NAME} created ..."
echo
