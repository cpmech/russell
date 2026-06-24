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

# Versions and installation paths
CUDSS_VERSION="0.8.0.10"
CUDA_VERSION="13"
CUDSS_TAR="libcudss-linux-x86_64-${CUDSS_VERSION}_cuda${CUDA_VERSION}-archive.tar.xz"
CUDSS_URL="https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/${CUDSS_TAR}"
INSTALL_DIR="/opt/libcudss"

# Install prerequisites for Ubuntu 24.04
echo "... installing system pre-requisites"
sudo apt-get update
sudo apt-get install -y wget curl build-essential xz-utils

# Set up official NVIDIA Ubuntu 24.04 Repository and install CUDA Toolkit
echo "... setting up official NVIDIA repository and installing CUDA"
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit

# Download the cuDSS redistributable tarball (reuse it if already present)
if [ -f "$CUDSS_TAR" ]; then
    echo "... using existing $CUDSS_TAR file"
else
    echo "... downloading cuDSS"
    curl -fL "$CUDSS_URL" -o "$CUDSS_TAR"
fi

# Extract
echo "... extracting $CUDSS_TAR"
tar -xvJf "$CUDSS_TAR"

# The archive extracts to a directory named libcudss-linux-x86_64-${VERSION}_cuda${CUDA_VERSION}-archive
EXTRACTED_DIR="libcudss-linux-x86_64-${CUDSS_VERSION}_cuda${CUDA_VERSION}-archive"

# Install to /opt/libcudss
if [ -d "$INSTALL_DIR" ]; then
    echo "... removing previous $INSTALL_DIR"
    sudo rm -rf "$INSTALL_DIR"
fi
sudo mv "$EXTRACTED_DIR" "$INSTALL_DIR"

# Register the library path with the dynamic linker
echo "${INSTALL_DIR}/lib" | sudo tee /etc/ld.so.conf.d/libcudss.conf >/dev/null
sudo ldconfig 2> >(grep -v 'is not an ELF file\|is not a symbolic link' >&2)

# Print environment variable instructions (Updated paths for Ubuntu)
echo ""
echo "======================================================================="
echo " cuDSS ${CUDSS_VERSION} installed to ${INSTALL_DIR}"
echo "======================================================================="
echo ""
echo "Add the following to your shell profile (~/.bashrc or equivalent):"
echo ""
echo "  export PATH=/usr/local/cuda/bin:\$PATH"
echo "  export CTK_DIR=/usr/local/cuda"
echo "  export CUDSS_DIR=${INSTALL_DIR}"
echo "  export LD_LIBRARY_PATH=\${CUDSS_DIR}/lib:\${CTK_DIR}/lib64:\${LD_LIBRARY_PATH:-}"
echo ""
echo "======================================================================="
