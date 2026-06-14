// kernel.cu
#include <cuda_runtime.h>
#include <stdio.h>

// 1. The GPU Kernel
// Executes on the device. Each thread prints a message.
__global__ void hello_kernel() {
    printf("Hello World from GPU! Block: %d, Thread: %d\n", blockIdx.x, threadIdx.x);
}

// 2. The Host Wrapper
// Executes on the CPU. Launches the kernel and synchronizes.
// Must be 'extern "C"' to prevent C++ name mangling for Rust FFI.
extern "C" void run_hello_world() {
    // Launch 1 block of 1 thread
    hello_kernel<<<1, 1>>>();
    
    // Synchronize to ensure the kernel finishes and printf output is flushed
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }
}   