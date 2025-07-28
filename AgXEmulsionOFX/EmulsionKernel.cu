#include "EmulsionKernel.cuh"
#include "DirCouplerKernel.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// LUT data in constant memory (definitions for linker)
__constant__ float c_logE[601];
__constant__ float c_curveR[601];
__constant__ float c_curveG[601];
__constant__ float c_curveB[601];
__constant__ float c_gamma;
__constant__ float c_exposureEV;

// Local device functions
__device__ __forceinline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ __forceinline__ float lookupDensity(float logE, const float* curve) {
    if(logE<=c_logE[0]){
        float t=(logE-c_logE[0])/(c_logE[1]-c_logE[0]);
        return lerp(curve[0],curve[1],t); // extrapolate below
    }
    if(logE>=c_logE[600]){
        float t=(logE-c_logE[599])/(c_logE[600]-c_logE[599]);
        return lerp(curve[599],curve[600],t); // extrapolate above
    }
    // linear search (601 small)
    int idx=0;
    for(int i=1;i<601;i++){if(logE<c_logE[i]){idx=i-1;break;}}
    float t=(logE-c_logE[idx])/(c_logE[idx+1]-c_logE[idx]);
    return lerp(curve[idx],curve[idx+1],t);
}

__global__ void EmulsionKernel(float4* pixels, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    float4 p = pixels[idx];
    
    // Convert RGB to log exposure
    float logER = log10f(fmaxf(p.x,1e-6f)) + c_exposureEV*0.30103f;
    float logEG = log10f(fmaxf(p.y,1e-6f)) + c_exposureEV*0.30103f;
    float logEB = log10f(fmaxf(p.z,1e-6f)) + c_exposureEV*0.30103f;
    
    // Prepare log exposure array for DIR-coupler pipeline
    // Note: We'll need to call DIR-coupler separately before this kernel
    // For now, do simple density lookup as fallback
    float dR = lookupDensity(logER/c_gamma, c_curveR);
    float dG = lookupDensity(logEG/c_gamma, c_curveG);
    float dB = lookupDensity(logEB/c_gamma, c_curveB);
    
    // Convert density to light transmission: T = 10^(-density)
    p.x = __powf(10.0f, -dR);
    p.y = __powf(10.0f, -dG);
    p.z = __powf(10.0f, -dB);
    
    pixels[idx] = p;
}

__global__ void LogExposureKernel(float4* pixels, float* logE, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    float4 p = pixels[idx];
    
    // Convert RGB to log exposure and store in separate array
    logE[idx*3+0] = log10f(fmaxf(p.x,1e-6f)) + c_exposureEV*0.30103f; // Red
    logE[idx*3+1] = log10f(fmaxf(p.y,1e-6f)) + c_exposureEV*0.30103f; // Green  
    logE[idx*3+2] = log10f(fmaxf(p.z,1e-6f)) + c_exposureEV*0.30103f; // Blue
}

__global__ void DensityToLightKernel(const float* density, float4* pixels, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    // Get CMY densities from DIR-coupler output
    float cDensity = density[idx*3+0]; // Cyan (affects red)
    float mDensity = density[idx*3+1]; // Magenta (affects green)
    float yDensity = density[idx*3+2]; // Yellow (affects blue)
    
    // Convert CMY densities to RGB light transmission: T = 10^(-density)
    float4 p;
    p.x = __powf(10.0f, -cDensity);  // Red transmission
    p.y = __powf(10.0f, -mDensity);  // Green transmission  
    p.z = __powf(10.0f, -yDensity);  // Blue transmission
    p.w = pixels[idx].w; // Preserve alpha
    
    pixels[idx] = p;
}

extern "C" void LaunchEmulsionCUDA(float4* d_pixels, int width, int height, float exposureEV) {
    printf("DEBUG: LaunchEmulsionCUDA starting with DIR-coupler pipeline\n");
    
    // Allocate temporary array for log exposure data
    size_t logE_bytes = width * height * 3 * sizeof(float);
    float* d_logE;
    cudaMalloc(&d_logE, logE_bytes);
    
    dim3 blockSize(256);
    dim3 gridSize((width * height + blockSize.x - 1) / blockSize.x);
    
    // Step 1: Convert RGB pixels to log exposure
    LogExposureKernel<<<gridSize, blockSize>>>(d_pixels, d_logE, width, height);
    cudaDeviceSynchronize();
    
    // Step 2: Apply DIR-coupler pipeline (modifies d_logE in-place to contain CMY densities)
    LaunchDirCouplerCUDA(d_logE, width, height);
    
    // Step 3: Convert final CMY densities back to RGB light
    DensityToLightKernel<<<gridSize, blockSize>>>(d_logE, d_pixels, width, height);
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(d_logE);
    
    printf("DEBUG: LaunchEmulsionCUDA completed\n");
}

extern "C" void UploadLUTCUDA(const float* logE, const float* r, const float* g, const float* b, float gamma){
    cudaMemcpyToSymbol(c_logE,logE,601*sizeof(float));
    cudaMemcpyToSymbol(c_curveR,r,601*sizeof(float));
    cudaMemcpyToSymbol(c_curveG,g,601*sizeof(float));
    cudaMemcpyToSymbol(c_curveB,b,601*sizeof(float));
    cudaMemcpyToSymbol(c_gamma,&gamma,sizeof(float));
} 