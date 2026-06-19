# NVIDIA CUDA and cuDSS (Direct Sparse Solver)

## Install CUDA

**1. Install the CUDA compiler (nvcc)**

```bash
sudo pacman -S cuda
```

**2. Download the examples**

```bash
git clone https://github.com/NVIDIA/cuda-samples.git /tmp/cuda-samples
```

**3. Compile and run the examples**

Make sure to install GCC 15 because CUDA won't work with newer GCC.

```bash
cd /tmp/cuda-samples
mkdir build && cd build
cmake .. -DCMAKE_CUDA_HOST_COMPILER=gcc-15 -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_CXX_COMPILER=g++-15
cd cpp/1_Utilities/deviceQuery
make
./deviceQuery 
```

## Install cuDSS

Either [download cuDSS from here](https://developer.nvidia.com/cudss-downloads?target_os=Linux&target_arch=x86_64&Distribution=Agnostic&cuda_version=13) or use the following command (you may choose a newer version):

```bash
cd ~/Downloads/
wget https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.8.0.10_cuda13-archive.tar.xz
tar -xvJf libcudss-linux-x86_64-0.8.0.10_cuda13-archive.tar.xz
sudo mv libcudss-linux-x86_64-0.8.0.10_cuda13-archive /opt/libcudss
```

Define the following environment variables in your system (e.g., bashrc):

```bashrc
export PATH=/opt/cuda/bin:$PATH
export CTK_DIR=/opt/cuda
export CUDSS_DIR=/opt/libcudss
export LD_LIBRARY_PATH=${CUDSS_DIR}/lib:${CTK_DIR}/lib64:${LD_LIBRARY_PATH}
```
    
## Build-time environment variables

The build script (`build.rs`) supports the following environment variables:

| Variable          | Default  | Description                                                                                                                                                              |
| ----------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `CXX`             | `g++-15` | CUDA host compiler (GCC). Panics if the version detected is > 15, because CUDA's `nvcc` is incompatible with GCC ≥ 16.                                                   |
| `CUDSS_CUDA_ARCH` | auto-detected | CUDA compute architecture passed to `nvcc -arch`. If not set, `build.rs` queries `nvidia-smi` to auto-detect the installed GPU (maps e.g. "8.9" → "sm_89"). Falls back to `sm_89` if detection fails. Set explicitly for cross-compilation: `sm_90` for H100, `sm_80` for A100, `sm_86` for RTX 30-series, etc. |

Example — auto-detection or explicit override:

```bash
# Auto-detect (uses nvidia-smi, falls back to sm_89)
cargo build --features cudss

# Force a specific architecture
CUDSS_CUDA_ARCH=sm_90 cargo build --features cudss
```

## Test the code

```bash
cd russell_sparse
cargo test --features cudss
```
