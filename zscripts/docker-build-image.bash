#!/bin/bash

# the first argument is "rocky" to build Rocky linux instead of Ubuntu
ROCKY=${1:-""}

# image name
NAME="cpmech/russell"
DKFILE="Dockerfile.Ubuntu"
if [ "${ROCKY}" = "rocky" ]; then
    NAME="cpmech/russell_rocky"
    DKFILE="Dockerfile.Rocky"
fi

# build Docker image
docker build --no-cache -f $DKFILE -t $NAME .

echo
echo
echo
echo "... SUCCESS ..."
echo
echo "... image ${NAME} created ..."
echo
