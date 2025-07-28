#include "EmulsionKernel.cuh"
#include <cuda_runtime.h>

__constant__ float c_logE[601];
__constant__ float c_curveR[601];
__constant__ float c_curveG[601];
__constant__ float c_curveB[601];
__constant__ float c_gamma;

__device__ __forceinline__ float lerp(float a,float b,float t){return a+(b-a)*t;}

__device__ float lookupDensity(const float* curve,const float* logE,float val){
    if(val<=logE[0]) return curve[0];
    if(val>=logE[600]) return curve[600];
    // linear search (601 small)
    int idx=0;
    for(int i=1;i<601;i++){if(val<logE[i]){idx=i-1;break;}}
    float t=(val-logE[idx])/(logE[idx+1]-logE[idx]);
    return lerp(curve[idx],curve[idx+1],t);
}

__global__ void EmulsionKernel(float* img, int width, int height)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=width*height;
    if(idx>=total) return;
    float4* pix = reinterpret_cast<float4*>(img);
    float4 p = pix[idx];
    float maxRGB = fmaxf(p.x,fmaxf(p.y,p.z));
    maxRGB = fmaxf(maxRGB,1e-6f);
    float logE = log10f(maxRGB);
    float dR = lookupDensity(c_curveR,c_logE, logE/c_gamma);
    float dG = lookupDensity(c_curveG,c_logE, logE/c_gamma);
    float dB = lookupDensity(c_curveB,c_logE, logE/c_gamma);
    // map density to display simple
    p.x = 1.0f - exp2f(-dR);
    p.y = 1.0f - exp2f(-dG);
    p.z = 1.0f - exp2f(-dB);
    pix[idx]=p;
}

extern "C" void LaunchEmulsionCUDA(float* img, int width, int height, float gamma)
{
    int total = width * height;
    int block = 256;
    int grid = (total + block - 1) / block;
    cudaMemcpyToSymbol(c_gamma,&gamma,sizeof(float));
    EmulsionKernel<<<grid, block>>>(img, width, height);
    cudaDeviceSynchronize();
}

extern "C" void UploadLUTCUDA(const float* logE,const float* r,const float* g,const float* b){
    cudaMemcpyToSymbol(c_logE,logE,601*sizeof(float));
    cudaMemcpyToSymbol(c_curveR,r,601*sizeof(float));
    cudaMemcpyToSymbol(c_curveG,g,601*sizeof(float));
    cudaMemcpyToSymbol(c_curveB,b,601*sizeof(float));
} 