#!/bin/bash

set -euo pipefail

# image name
NAME="cpmech/russell_ubuntu"
DKFILE="zdocker/Dockerfile.Ubuntu"

# build Docker image
docker build -f "$DKFILE" -t "$NAME" .

echo
echo "... SUCCESS: image ${NAME} created ..."
echo
