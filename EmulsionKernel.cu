#include "EmulsionKernel.cuh"
#include <cuda_runtime.h>

__global__ void EmulsionKernel(float* img, int width, int height, float gamma)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if(idx>=total) return;
    float4* pix = reinterpret_cast<float4*>(img);
    float4 p = pix[idx];
    // simple placeholder: apply gamma to RGB invert style
    p.x = powf(p.x, gamma);
    p.y = powf(p.y, gamma);
    p.z = powf(p.z, gamma);
    pix[idx] = p;
}

extern "C" void LaunchEmulsionCUDA(float* img, int width, int height, float gamma)
{
    int total = width * height;
    int block = 256;
    int grid = (total + block - 1) / block;
    EmulsionKernel<<<grid, block>>>(img, width, height, gamma);
    cudaDeviceSynchronize();
} 