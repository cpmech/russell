#!/bin/bash

set -euo pipefail

# the first argument is "1" to enable MKL
# the second argument is "1" to enable local sparse libs
INTEL_MKL="${1:-0}"
LOCAL_SPARSE="${2:-0}"

# image name and Dockerfile
if [ "${INTEL_MKL}" = "1" ]; then
    DOCKERFILE="zdocker/Dockerfile.Arch.Mkl.Local"
    NAME="cpmech/russell_arch_mkl_local"
elif [ "${LOCAL_SPARSE}" = "1" ]; then
    DOCKERFILE="zdocker/Dockerfile.Arch.Local"
    NAME="cpmech/russell_arch_local"
else
    DOCKERFILE="zdocker/Dockerfile.Arch"
    NAME="cpmech/russell_arch"
fi

# build Docker image
docker build \
    -f "${DOCKERFILE}" \
    -t "${NAME}" \
    .

echo
echo "... SUCCESS: image ${NAME} created ..."
echo
