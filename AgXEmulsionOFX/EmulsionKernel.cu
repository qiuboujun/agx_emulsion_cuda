#include "EmulsionKernel.cuh"
#include <cuda_runtime.h>

__constant__ float c_logE[601];
__constant__ float c_curveR[601];
__constant__ float c_curveG[601];
__constant__ float c_curveB[601];
__constant__ float c_gamma;
__constant__ float c_exposureEV;

__device__ __forceinline__ float lerp(float a,float b,float t){return a+(b-a)*t;}

__device__ float lookupDensity(const float* curve,const float* logE,float val){
    // clamp to table range
    if(val <= logE[0]) {
        return curve[0];
    }
    if(val >= logE[600]) {
        return curve[600];
    }
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
    float logER = log10f(fmaxf(p.x,1e-6f)) + c_exposureEV*0.30103f;
    float logEG = log10f(fmaxf(p.y,1e-6f)) + c_exposureEV*0.30103f;
    float logEB = log10f(fmaxf(p.z,1e-6f)) + c_exposureEV*0.30103f;
    // Convert log-exposure to CMY density
    p.x = lookupDensity(c_curveR, c_logE, logER / c_gamma);
    p.y = lookupDensity(c_curveG, c_logE, logEG / c_gamma);
    p.z = lookupDensity(c_curveB, c_logE, logEB / c_gamma);
    pix[idx]=p;
}

extern "C" void LaunchEmulsionCUDA(float* img, int width, int height, float gamma, float exposureEV)
{
    int total = width * height;
    int block = 256;
    int grid = (total + block - 1) / block;
    cudaMemcpyToSymbol(c_gamma,&gamma,sizeof(float));
    cudaMemcpyToSymbol(c_exposureEV,&exposureEV,sizeof(float));
    EmulsionKernel<<<grid, block>>>(img, width, height);
    cudaDeviceSynchronize();
}

extern "C" void UploadLUTCUDA(const float* logE,const float* r,const float* g,const float* b){
    cudaMemcpyToSymbol(c_logE,logE,601*sizeof(float));
    cudaMemcpyToSymbol(c_curveR,r,601*sizeof(float));
    cudaMemcpyToSymbol(c_curveG,g,601*sizeof(float));
    cudaMemcpyToSymbol(c_curveB,b,601*sizeof(float));
} 