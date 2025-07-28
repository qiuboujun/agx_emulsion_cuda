#include "DirCouplerKernel.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__constant__ float c_dirM[9]; // 3x3 matrix row-major
__constant__ float c_dMax[3];
__constant__ float c_highShift;
__constant__ float c_sigmaPx;

// helper to build 1D Gaussian kernel up to radius 12 on device constant mem
__constant__ float c_gaussK[25]; // radius<=12
__constant__ int c_radius;

// Local references to LUT constants from EmulsionKernel (declared there)
extern __constant__ float c_logE[601];
extern __constant__ float c_curveR[601];
extern __constant__ float c_curveG[601];
extern __constant__ float c_curveB[601];

// Local inline device functions (copied from EmulsionKernel)
__device__ __forceinline__ float lerp_local(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ __forceinline__ float lookupDensity_local(float logE, const float* curve) {
    if(logE<=c_logE[0]){
        float t=(logE-c_logE[0])/(c_logE[1]-c_logE[0]);
        return lerp_local(curve[0],curve[1],t); // extrapolate below
    }
    if(logE>=c_logE[600]){
        float t=(logE-c_logE[599])/(c_logE[600]-c_logE[599]);
        return lerp_local(curve[599],curve[600],t); // extrapolate above
    }
    // linear search (601 small)
    int idx=0;
    for(int i=1;i<601;i++){if(logE<c_logE[i]){idx=i-1;break;}}
    float t=(logE-c_logE[idx])/(c_logE[idx+1]-c_logE[idx]);
    return lerp_local(curve[idx],curve[idx+1],t);
}

__global__ void BuildGaussianKernel(float sigma){
    int r = (int)roundf(3.0f*sigma);
    if(r<1) r=1; if(r>12) r=12;
    if(threadIdx.x==0){
        float sum=0.0f;
        for(int i=-r;i<=r;i++){
            float v=expf(-0.5f*i*i/(sigma*sigma));
            ((float*)c_gaussK)[i+r]=v;
            sum+=v;
        }
        for(int i=0;i<=2*r;i++) ((float*)c_gaussK)[i]/=sum;
        *((int*)&c_radius)=r;
        printf("DEBUG DIR: Built Gaussian kernel, sigma=%.3f, radius=%d\n", sigma, r);
    }
}

__device__ __forceinline__ int reflect(int idx,int len){return (idx<0)?-idx-1: (idx>=len)?2*len-idx-1: idx;}

__global__ void HorizontalBlur(const float* in,float* out,int W,int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    int r=c_radius;
    for(int c=0;c<3;c++){
        float sum=0.0f;
        for(int t=-r;t<=r;t++){
            int xi=reflect(x+t,W);
            sum+=in[(y*W+xi)*3+c]*c_gaussK[t+r];
        }
        out[(y*W+x)*3+c]=sum;
    }
}

__global__ void VerticalBlur(const float* in,float* out,int W,int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    int r=c_radius;
    for(int c=0;c<3;c++){
        float sum=0.0f;
        for(int t=-r;t<=r;t++){
            int yi=reflect(y+t,H);
            sum+=in[(yi*W+x)*3+c]*c_gaussK[t+r];
        }
        out[(y*W+x)*3+c]=sum;
    }
}

__global__ void BuildCorrectionKernel(const float* logE, float* correction, int W, int H){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=W*H;
    if(idx>=total) return;
    
    // Get log exposure for this pixel
    float3 logExp;
    logExp.x = logE[idx*3+0];
    logExp.y = logE[idx*3+1]; 
    logExp.z = logE[idx*3+2];
    
    // Interpolate density using LUT (same as EmulsionKernel)
    float3 density;
    density.x = lookupDensity_local(logExp.x, c_curveR);
    density.y = lookupDensity_local(logExp.y, c_curveG);
    density.z = lookupDensity_local(logExp.z, c_curveB);
    
    // Normalize by density max
    float3 normD;
    normD.x = density.x / c_dMax[0];
    normD.y = density.y / c_dMax[1];
    normD.z = density.z / c_dMax[2];
    
    // Add high exposure shift (quadratic term)
    normD.x += c_highShift * normD.x * normD.x;
    normD.y += c_highShift * normD.y * normD.y;
    normD.z += c_highShift * normD.z * normD.z;
    
    // Apply DIR matrix
    float3 corr;
    corr.x = normD.x*c_dirM[0] + normD.y*c_dirM[1] + normD.z*c_dirM[2];
    corr.y = normD.x*c_dirM[3] + normD.y*c_dirM[4] + normD.z*c_dirM[5];
    corr.z = normD.x*c_dirM[6] + normD.y*c_dirM[7] + normD.z*c_dirM[8];
    
    correction[idx*3+0] = corr.x;
    correction[idx*3+1] = corr.y;
    correction[idx*3+2] = corr.z;
    
    // Debug first pixel
    if(idx == 0) {
        printf("DEBUG DIR: First pixel correction before blur: R=%.6f G=%.6f B=%.6f\n", 
               corr.x, corr.y, corr.z);
    }
}

__global__ void ApplyCorrectionKernel(float* logE, const float* blurredCorr, int W, int H){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=W*H;
    if(idx>=total) return;
    
    // Subtract blurred correction from log exposure
    float3 newLogE;
    newLogE.x = logE[idx*3+0] - blurredCorr[idx*3+0];
    newLogE.y = logE[idx*3+1] - blurredCorr[idx*3+1];
    newLogE.z = logE[idx*3+2] - blurredCorr[idx*3+2];
    
    // Re-interpolate density with corrected log exposure
    float3 finalDensity;
    finalDensity.x = lookupDensity_local(newLogE.x, c_curveR);
    finalDensity.y = lookupDensity_local(newLogE.y, c_curveG);
    finalDensity.z = lookupDensity_local(newLogE.z, c_curveB);
    
    // Store final CMY densities back to logE array (reusing same memory)
    logE[idx*3+0] = finalDensity.x; // C (red)
    logE[idx*3+1] = finalDensity.y; // M (green)
    logE[idx*3+2] = finalDensity.z; // Y (blue)
    
    // Debug first pixel
    if(idx == 0) {
        printf("DEBUG DIR: First pixel final CMY density: C=%.6f M=%.6f Y=%.6f\n", 
               finalDensity.x, finalDensity.y, finalDensity.z);
    }
}

extern "C" void UploadDirMatrixCUDA(const float* M){
    cudaMemcpyToSymbol(c_dirM,M,9*sizeof(float));
    printf("DEBUG DIR: Uploaded DIR matrix [%.3f %.3f %.3f; %.3f %.3f %.3f; %.3f %.3f %.3f]\n",
           M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8]);
}

extern "C" void UploadDirParamsCUDA(const float* dmax,float highShift,float sigmaPx){
    cudaMemcpyToSymbol(c_dMax,dmax,3*sizeof(float));
    cudaMemcpyToSymbol(c_highShift,&highShift,sizeof(float));
    cudaMemcpyToSymbol(c_sigmaPx,&sigmaPx,sizeof(float));
    printf("DEBUG DIR: Uploaded params - dMax=[%.3f,%.3f,%.3f], highShift=%.3f, sigmaPx=%.3f\n",
           dmax[0],dmax[1],dmax[2],highShift,sigmaPx);
    
    // Build Gaussian kernel if sigma is meaningful
    if(sigmaPx > 0.5f) {
        BuildGaussianKernel<<<1,1>>>(sigmaPx);
        cudaDeviceSynchronize();
    }
}

extern "C" void LaunchDirCouplerCUDA(float* logE, int W, int H){
    printf("DEBUG DIR: Starting DIR-coupler pipeline %dx%d\n", W, H);
    
    size_t bytes = W * H * 3 * sizeof(float);
    float* correction;
    float* temp;
    
    // Allocate temporary buffers
    cudaMalloc(&correction, bytes);
    cudaMalloc(&temp, bytes);
    
    dim3 blockSize(256);
    dim3 gridSize((W*H + blockSize.x - 1) / blockSize.x);
    dim3 blockSize2D(16, 16);
    dim3 gridSize2D((W + blockSize2D.x - 1) / blockSize2D.x, 
                    (H + blockSize2D.y - 1) / blockSize2D.y);
    
    // Step 1: Build correction based on normalized density and DIR matrix
    BuildCorrectionKernel<<<gridSize, blockSize>>>(logE, correction, W, H);
    cudaDeviceSynchronize();
    
    // Step 2: Blur correction if sigma > 0.5
    if(c_sigmaPx > 0.5f) {
        // Horizontal blur: correction -> temp
        HorizontalBlur<<<gridSize2D, blockSize2D>>>(correction, temp, W, H);
        cudaDeviceSynchronize();
        
        // Vertical blur: temp -> correction
        VerticalBlur<<<gridSize2D, blockSize2D>>>(temp, correction, W, H);
        cudaDeviceSynchronize();
        
        printf("DEBUG DIR: Applied Gaussian blur with sigma=%.3f\n", c_sigmaPx);
    } else {
        printf("DEBUG DIR: Skipping blur (sigma=%.3f <= 0.5)\n", c_sigmaPx);
    }
    
    // Step 3: Apply correction and re-interpolate density
    ApplyCorrectionKernel<<<gridSize, blockSize>>>(logE, correction, W, H);
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(correction);
    cudaFree(temp);
    
    printf("DEBUG DIR: Completed DIR-coupler pipeline\n");
} 