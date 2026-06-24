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

# Install CUDA toolkit (nvcc and runtime)
sudo pacman -S --noconfirm cuda

# Download the cuDSS redistributable tarball (reuse it if already present)
cd /tmp
if [ -f "$CUDSS_TAR" ]; then
    echo "... using existing $CUDSS_TAR file"
else
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

# Print environment variable instructions
echo ""
echo "======================================================================="
echo " cuDSS ${CUDSS_VERSION} installed to ${INSTALL_DIR}"
echo "======================================================================="
echo ""
echo "Add the following to your shell profile (~/.bashrc or equivalent):"
echo ""
echo "  export PATH=/opt/cuda/bin:\$PATH"
echo "  export CTK_DIR=/opt/cuda"
echo "  export CUDSS_DIR=${INSTALL_DIR}"
echo "  export LD_LIBRARY_PATH=\${CUDSS_DIR}/lib:\${CTK_DIR}/lib64:\${LD_LIBRARY_PATH}"
echo ""
echo "======================================================================="
