#!/bin/bash

set -euo pipefail

# first argument:  distro = "" (ubuntu) | "arch" | "rocky"
# second argument: "1" to enable Intel MKL (arch only)
# third argument:  "1" to enable MUMPS    (arch only)
DISTRO="${1:-}"
INTEL_MKL="${2:-0}"
LOCAL_SPARSE="${3:-0}"

# image name
if [ "${DISTRO}" = "arch" ]; then
    NAME="cpmech/russell_arch"
    if [ "${INTEL_MKL}" = "1" ]; then
        NAME="${NAME}_mkl_local"
    elif [ "${LOCAL_SPARSE}" = "1" ]; then
        NAME="${NAME}_local"
    fi
elif [ "${DISTRO}" = "rocky" ]; then
    NAME="cpmech/russell_rocky"
else
    NAME="cpmech/russell_ubuntu"
fi

docker run --rm -it "${NAME}:latest" /bin/bash
