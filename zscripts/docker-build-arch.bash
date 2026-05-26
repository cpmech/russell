#!/bin/bash

set -e

# the first argument is "1" to enable MKL
# the second argument is "1" to enable MUMPS
INTEL_MKL=${1:-""}
WITH_MUMPS=${2:-""}

# image name
NAME="cpmech/russell_arch"
DKFILE="zdocker/Dockerfile.Arch"
if [ "${INTEL_MKL}" = "1" ]; then
    NAME="${NAME}_mkl"
    DKFILE="${DKFILE}.Mkl"
fi
if [ "${WITH_MUMPS}" = "1" ]; then
    NAME="${NAME}_mumps"
    DKFILE="${DKFILE}.Mumps"
fi

# build Docker image
docker build -f "$DKFILE" -t "$NAME" .

echo
echo "... SUCCESS: image ${NAME} created ..."
echo
