#!/bin/bash

# the first argument is the distro: "arch" or "rocky"
# the second argument is "1" to enable MUMPS
DISTRO=${1:-""}
WITH_MUMPS=${2:-""}

# image name
NAME="cpmech/russell_ubuntu"
DKFILE="Dockerfile.Ubuntu"
if [ "${DISTRO}" = "arch" ]; then
    NAME="cpmech/russell_arch"
    DKFILE="Dockerfile.Arch"
fi
if [ "${DISTRO}" = "rocky" ]; then
    NAME="cpmech/russell_rocky"
    DKFILE="Dockerfile.Rocky"
fi
if [ "${WITH_MUMPS}" = "1" ]; then
    NAME="${NAME}_mumps"
    DKFILE="${DKFILE}.Mumps"
fi

# build Docker image
docker build -f $DKFILE -t $NAME .

echo
echo
echo
echo "... SUCCESS ..."
echo
echo "... image ${NAME} created ..."
echo
