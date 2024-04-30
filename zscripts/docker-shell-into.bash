#!/bin/bash

# the first argument is the distro: "arch" or "rocky"
# the second argument is "1" to enable MUMPS
DISTRO=${1:-""}
WITH_MUMPS=${2:-""}

# image name
NAME="cpmech/russell_ubuntu"
if [ "${DISTRO}" = "arch" ]; then
    NAME="cpmech/russell_arch"
fi
if [ "${DISTRO}" = "rocky" ]; then
    NAME="cpmech/russell_rocky"
fi
if [ "${WITH_MUMPS}" = "1" ]; then
    NAME="${NAME}_mumps"
fi

VERSION="latest"

docker run --rm -it $NAME:$VERSION /bin/bash
