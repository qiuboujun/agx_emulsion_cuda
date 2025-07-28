#include <cuda_runtime.h>
#include "PaperKernel.cuh"

__constant__ float p_logE[601];
__constant__ float p_curveR[601];
__constant__ float p_curveG[601];
__constant__ float p_curveB[601];

__device__ __forceinline__ float lerpP(float a,float b,float t){return a+(b-a)*t;}

__device__ float interpCurve(const float* curve,const float* logE,float v){
    if(v<=logE[0]){
        float t=(v-logE[0])/(logE[1]-logE[0]);
        return lerpP(curve[0],curve[1],t);
    }
    if(v>=logE[600]){
        float t=(v-logE[599])/(logE[600]-logE[599]);
        return lerpP(curve[599],curve[600],t);
    }
    int idx=0; for(int i=1;i<601;i++){ if(v<logE[i]){idx=i-1;break;} }
    float t=(v-logE[idx])/(logE[idx+1]-logE[idx]);
    return lerpP(curve[idx],curve[idx+1],t);
}

__global__ void PaperKernel(float* img,int width,int height){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int tot = width*height; if(idx>=tot) return;
    float4* pix = reinterpret_cast<float4*>(img);
    float4 p = pix[idx];
    // convert light -> density
    float dR = -log10f(fminf(fmaxf(p.x,1e-6f),1.0f));
    float dG = -log10f(fminf(fmaxf(p.y,1e-6f),1.0f));
    float dB = -log10f(fminf(fmaxf(p.z,1e-6f),1.0f));
    dR = fminf(dR,3.0f);
    dG = fminf(dG,3.0f);
    dB = fminf(dB,3.0f);
    // map through paper curves (gamma assumed 1)
    float ndR = interpCurve(p_curveR,p_logE,dR);
    float ndG = interpCurve(p_curveG,p_logE,dG);
    float ndB = interpCurve(p_curveB,p_logE,dB);
    // back to light
    p.x = __powf(10.0f,-ndR);
    p.y = __powf(10.0f,-ndG);
    p.z = __powf(10.0f,-ndB);
    pix[idx]=p;
}

extern "C" void UploadPaperLUTCUDA(const float* logE,const float* r,const float* g,const float* b){
    cudaMemcpyToSymbol(p_logE,logE,601*sizeof(float));
    cudaMemcpyToSymbol(p_curveR,r,601*sizeof(float));
    cudaMemcpyToSymbol(p_curveG,g,601*sizeof(float));
    cudaMemcpyToSymbol(p_curveB,b,601*sizeof(float));
}

extern "C" void LaunchPaperCUDA(float* img,int width,int height){
    int total = width*height;
    dim3 block(256);
    dim3 grid((total+block.x-1)/block.x);
    PaperKernel<<<grid,block>>>(img,width,height);
    cudaDeviceSynchronize();
} 