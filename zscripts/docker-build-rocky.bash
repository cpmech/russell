#!/bin/bash

set -e

# image name
NAME="cpmech/russell_rocky"
DKFILE="zdocker/Dockerfile.Rocky"

# build Docker image
docker build -f "$DKFILE" -t "$NAME" .

echo
echo "... SUCCESS: image ${NAME} created ..."
echo
