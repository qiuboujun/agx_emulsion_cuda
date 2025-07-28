#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "GrainKernel.cuh"

__global__ void GrainKernel(float* img, int width, int height, float strength, unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=width || y>=height) return;

    int idx = (y*width + x)*4;

    // init cuRAND per thread
    curandState state;
    curand_init(seed, y*width + x, 0, &state);

    float n = (curand_uniform(&state)-0.5f)*2.f; // [-1,1]
    n *= strength;

    // apply to all channels
    img[idx]   = fminf(fmaxf(img[idx]   + n,0.f),1.f);
    img[idx+1] = fminf(fmaxf(img[idx+1] + n,0.f),1.f);
    img[idx+2] = fminf(fmaxf(img[idx+2] + n,0.f),1.f);
}

extern "C" void LaunchGrainCUDA(float* img, int width, int height, float strength, unsigned int seed)
{
    if(strength < 1e-5f) return;
    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
    GrainKernel<<<grid,block>>>(img,width,height,strength,seed);
    cudaDeviceSynchronize();
} 