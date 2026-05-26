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

# Add the Intel oneAPI APT repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor \
    | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | sudo tee /etc/apt/sources.list.d/oneAPI.list

# Install the Intel oneAPI packages
# note: GCC is still needed as a backend for the Intel IFX (Fortran) compiler
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    gcc \
    intel-oneapi-compiler-fortran \
    intel-oneapi-mkl \
    intel-oneapi-mkl-devel

# Source the oneAPI environment to make MKLROOT and compiler paths available;
# suppress -u temporarily because setvars.sh references variables before setting them
set +u
source /opt/intel/oneapi/setvars.sh  # sets CC, FC, MKLROOT, and PATH for the ifx compiler
set -u

# Remove the broken non-ELF file that ships with MKL and confuses ldconfig
WEIRD_FILE="$MKLROOT/lib/intel64/libmkl_sycl.so"
if [ -f "$WEIRD_FILE" ]; then
    sudo mv "$WEIRD_FILE" "$MKLROOT/lib/intel64/libmkl_sick.txt"
fi

# Register the MKL and compiler library paths with the dynamic linker
echo "$MKLROOT/lib/intel64" | sudo tee /etc/ld.so.conf.d/intel-mkl.conf >/dev/null
echo "$CMPLR_ROOT/lib" | sudo tee -a /etc/ld.so.conf.d/intel-compiler.conf >/dev/null
sudo ldconfig
