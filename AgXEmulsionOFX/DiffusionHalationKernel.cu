// GPU diffusion & halation (very naive reference implementation)
#include <cuda_runtime.h>
#include "DiffusionHalationKernel.cuh"

#define MAX_RADIUS 25

__device__ __forceinline__ int clampi(int v, int lo, int hi) {return v < lo ? lo : (v > hi ? hi : v);} 

__global__ void DiffusionHalationKernel(float* img, int width, int height, int stride, int rad, float halStrength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    int index = (y * width + x) * 4; // RGBA float

    // Gaussian blur on red channel
    float sigma = rad * 0.5f + 1e-3f;
    float twoSigma2 = 2.0f * sigma * sigma;
    float sumR = 0.f;
    float wsum = 0.f;
    for(int dy=-rad; dy<=rad; ++dy){
        int yy = clampi(y + dy, 0, height-1);
        for(int dx=-rad; dx<=rad; ++dx){
            int xx = clampi(x + dx, 0, width-1);
            float w = __expf(-(dx*dx + dy*dy)/twoSigma2);
            int idx = (yy * width + xx) * 4;
            sumR += w * img[idx];
            wsum += w;
        }
    }
    float blurR = sumR / wsum;

    // original red
    float origR = img[index];
    float newR = origR + halStrength * (blurR - origR);
    img[index] = fminf(fmaxf(newR, 0.f), 1.f);
    // diffusion: simple overall blur of all channels (optional small)
    // Using same box blur scaled by radius*0.1 for now - placeholder
    // Could do proper Gaussian later
}

extern "C" void LaunchDiffusionHalationCUDA(float* img, int width, int height, float radius, float halStrength)
{
    if(radius < 0.5f && halStrength < 1e-5f) return; // skip if negligible
    int rad = (int)(radius + 0.5f);
    if(rad < 1) rad = 1;
    if(rad > MAX_RADIUS) rad = MAX_RADIUS;

    dim3 block(16,16);
    dim3 grid((width + block.x -1)/block.x, (height + block.y -1)/block.y);
    DiffusionHalationKernel<<<grid, block>>>(img, width, height, width*4, rad, halStrength);
    cudaDeviceSynchronize();
} 