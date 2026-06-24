#!/bin/bash

# Exit on error (-e), treat unset variables as errors (-u), and propagate
# pipeline failures (-o pipefail) so any silent failure is caught early
set -euo pipefail

# Versions and installation paths
CUDSS_VERSION="0.8.0.10"
CUDA_VERSION="13"
CUDSS_ZIP="libcudss-windows-x86_64-${CUDSS_VERSION}_cuda${CUDA_VERSION}-archive.zip"
CUDSS_URL="https://developer.download.nvidia.com/compute/cudss/redist/libcudss/windows-x86_64/${CUDSS_ZIP}"
PREFIX="/ucrt64"
INSTALL_DIR="${PREFIX}/libcudss"

# Note: CUDA toolkit must be installed separately via the NVIDIA Windows installer
# (https://developer.nvidia.com/cuda-downloads).
# This script assumes CUDA is already present at the standard location
# (typically C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}.x).

# Download the cuDSS redistributable archive (reuse it if already present)
cd /tmp
if [ -f "$CUDSS_ZIP" ]; then
    echo "... using existing $CUDSS_ZIP file"
else
    curl -fL "$CUDSS_URL" -o "$CUDSS_ZIP"
fi

# Extract
echo "... extracting $CUDSS_ZIP"
unzip -qo "$CUDSS_ZIP"

# The archive extracts to a directory named libcudss-windows-x86_64-${VERSION}_cuda${CUDA_VERSION}-archive
EXTRACTED_DIR="libcudss-windows-x86_64-${CUDSS_VERSION}_cuda${CUDA_VERSION}-archive"

# Install to UCRT64 prefix
if [ -d "$INSTALL_DIR" ]; then
    echo "... removing previous $INSTALL_DIR"
    rm -rf "$INSTALL_DIR"
fi
mv "$EXTRACTED_DIR" "$INSTALL_DIR"

# Copy libraries and headers into the UCRT64 tree so the build system finds them
mkdir -p "${PREFIX}/include/libcudss" "${PREFIX}/lib" "${PREFIX}/bin"
cp -av "${INSTALL_DIR}/include/"* "${PREFIX}/include/libcudss/"
cp -av "${INSTALL_DIR}/lib/"*.lib "${PREFIX}/lib/"  2>/dev/null || true
cp -av "${INSTALL_DIR}/lib/"*.dll "${PREFIX}/bin/"  2>/dev/null || true
cp -av "${INSTALL_DIR}/bin/"*.dll "${PREFIX}/bin/"  2>/dev/null || true

# Print environment variable instructions
echo ""
echo "======================================================================="
echo " cuDSS ${CUDSS_VERSION} installed to ${INSTALL_DIR}"
echo "======================================================================="
echo ""
echo "Add the following to your MSYS2 UCRT64 shell profile (~/.bashrc):"
echo ""
echo "  export CTK_DIR=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v${CUDA_VERSION}.0"
echo "  export CUDSS_DIR=${INSTALL_DIR}"
echo "  export PATH=\${CTK_DIR}/bin:\${PATH}"
echo ""
echo "======================================================================="
