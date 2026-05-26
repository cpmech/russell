#!/bin/bash

set -euo pipefail

# the first argument is "1" to enable MKL
# the second argument is "1" to enable MUMPS
INTEL_MKL="${1:-0}"
WITH_MUMPS="${2:-0}"

# image name and Dockerfile
NAME="cpmech/russell_arch"
DOCKERFILE="zdocker/Dockerfile.Arch"
if [ "${INTEL_MKL}" = "1" ] && [ "${WITH_MUMPS}" = "1" ]; then
    NAME="${NAME}_mkl_mumps"
    DOCKERFILE="zdocker/Dockerfile.Arch.Mkl.Mumps"
elif [ "${INTEL_MKL}" = "1" ]; then
    NAME="${NAME}_mkl"
    DOCKERFILE="zdocker/Dockerfile.Arch.Mkl"
elif [ "${WITH_MUMPS}" = "1" ]; then
    NAME="${NAME}_mumps"
    DOCKERFILE="zdocker/Dockerfile.Arch.Mumps"
fi

# build Docker image
docker build \
    -f "${DOCKERFILE}" \
    -t "${NAME}" \
    .

echo
echo "... SUCCESS: image ${NAME} created ..."
echo
