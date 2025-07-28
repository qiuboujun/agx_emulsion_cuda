#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__ void invert_kernel(float* img, int width, int height, float strength){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;
    
    int idx = (y * width + x) * 4;
    // Only process RGB, leave alpha unchanged
    for(int c = 0; c < 3; ++c){
        float v = img[idx + c];
        img[idx + c] = v * (1.0f - strength) + (1.0f - v) * strength;
    }
}

void LaunchInvertCUDA(float* img, int width, int height, float strength){
    // Safety checks
    if (!img || width <= 0 || height <= 0) {
        return;
    }
    
    // Check if we have a valid CUDA context
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before launch: %s\n", cudaGetErrorString(err));
        return;
    }
    
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    invert_kernel<<<grid, block>>>(img, width, height, strength);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
    }
} 