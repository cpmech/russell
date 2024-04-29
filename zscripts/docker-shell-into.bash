#!/bin/bash

# the first argument is "rocky" to use Rocky linux instead of Ubuntu
ROCKY=${1:-""}

# image name
NAME="cpmech/russell"
if [ "${ROCKY}" = "rocky" ]; then
    NAME="cpmech/russell_rocky"
fi

VERSION="latest"

docker run --rm -it $NAME:$VERSION /bin/bash
