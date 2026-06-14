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

```bash
cd /tmp/cuda-samples
mkdir build && cd build
cmake .. -DCMAKE_CUDA_HOST_COMPILER=gcc-15 -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_CXX_COMPILER=g++-15
cd cpp/1_Utilities/deviceQuery
make
./deviceQuery 
```

## Install cuDSS

[Download cuDSS](https://developer.nvidia.com/cudss-downloads?target_os=Linux&target_arch=x86_64&Distribution=Agnostic&cuda_version=13)

```bash
wget https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.8.0.10_cuda13-archive.tar.xz
```
