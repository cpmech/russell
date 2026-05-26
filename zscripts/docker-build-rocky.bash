#!/bin/bash

set -euo pipefail

# image name
NAME="cpmech/russell_rocky"
DOCKERFILE="zdocker/Dockerfile.Rocky"

# build Docker image
docker build -f "$DOCKERFILE" -t "$NAME" .

echo
echo "... SUCCESS: image ${NAME} created ..."
echo
