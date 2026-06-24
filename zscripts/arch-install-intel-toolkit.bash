#!/bin/bash

# Exit on error (-e), treat unset variables as errors (-u), and propagate
# pipeline failures (-o pipefail) so any silent failure is caught early
set -euo pipefail

# When running as root (e.g. inside Docker), act as a no-op wrapper;
# otherwise delegate to the real sudo
sudo () {
  [[ $EUID = 0 ]] || set -- command sudo "$@"
  "$@"
}

# Install the Intel oneAPI toolkit package
sudo pacman -S --noconfirm intel-oneapi-toolkit

# Source the oneAPI environment to make MKLROOT and compiler paths available;
# suppress -u temporarily because setvars.sh references variables before setting them
set +u
source /opt/intel/oneapi/setvars.sh  # sets CC, FC, MKLROOT, and PATH for the ifx compiler
set -u

# Register the MKL library path with the dynamic linker
echo "$MKLROOT/lib/intel64" | sudo tee /etc/ld.so.conf.d/intel-mkl.conf >/dev/null
sudo ldconfig 2> >(grep -v 'is not an ELF file\|is not a symbolic link' >&2)
