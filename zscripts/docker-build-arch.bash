#!/bin/bash

set -e

# the first argument is "1" to enable MKL
# the second argument is "1" to enable MUMPS
INTEL_MKL="${1:-0}"
WITH_MUMPS="${2:-0}"

# image name
NAME="cpmech/russell_arch"
[ "${INTEL_MKL}" = "1" ] && NAME="${NAME}_mkl"
[ "${WITH_MUMPS}" = "1" ] && NAME="${NAME}_mumps"

# build Docker image
docker build \
    --build-arg INTEL_MKL="${INTEL_MKL}" \
    --build-arg WITH_MUMPS="${WITH_MUMPS}" \
    -f "zdocker/Dockerfile.Arch" \
    -t "$NAME" \
    .

echo
echo "... SUCCESS: image ${NAME} created ..."
echo
