#include "DirCouplerKernel.cuh"
#include <cuda_runtime.h>
#include <math.h>

__constant__ float c_dirM[9]; // 3x3 matrix row-major
__constant__ float c_dMax[3];
__constant__ float c_highShift;
__constant__ float c_sigmaPx;

// helper to build 1D Gaussian kernel up to radius 12 on device constant mem
__constant__ float c_gaussK[25]; // radius<=12
__constant__ int c_radius;

// Precompute Gaussian kernel on host and upload to constant memory

static void BuildGaussianKernelHost(float sigma){
    int r = (int)roundf(3.0f*sigma);
    if(r<1) r=1; if(r>12) r=12;
    float hostK[25]={0};
    float sum=0.0f;
    for(int i=-r;i<=r;i++){
        float v=expf(-0.5f*i*i/(sigma*sigma));
        hostK[i+r]=v;
        sum+=v;
    }
    for(int i=0;i<=2*r;i++) hostK[i]/=sum;
    // copy kernel and radius to device constant memory
    cudaMemcpyToSymbol(c_gaussK,hostK,(2*r+1)*sizeof(float));
    cudaMemcpyToSymbol(c_radius,&r,sizeof(int));
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

/* forward declarations so BuildCorrectionKernel can call density lookup */
extern __constant__ float c_logE[601];
extern __constant__ float c_curveR[601];
extern __constant__ float c_curveG[601];
extern __constant__ float c_curveB[601];
__device__ float lookupDensityCUDA(const float* curve,const float* logE,float val);

__global__ void BuildCorrectionKernel(const float* image,float* corr,int W,int H){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = W*H;
    if(idx>=total) return;
    int base = idx*4; // RGBA stride
    float3 d;
    // Image holds CMY density
    float densR = image[base+0];
    float densG = image[base+1];
    float densB = image[base+2];
    d.x = densR / c_dMax[0];
    d.y = densG / c_dMax[1];
    d.z = densB / c_dMax[2];
    d.x += c_highShift * d.x * d.x;
    d.y += c_highShift * d.y * d.y;
    d.z += c_highShift * d.z * d.z;
    float3 out;
    out.x=d.x*c_dirM[0]+d.y*c_dirM[1]+d.z*c_dirM[2];
    out.y=d.x*c_dirM[3]+d.y*c_dirM[4]+d.z*c_dirM[5];
    out.z=d.x*c_dirM[6]+d.y*c_dirM[7]+d.z*c_dirM[8];
    corr[idx*3+0]=out.x;
    corr[idx*3+1]=out.y;
    corr[idx*3+2]=out.z;
}

__device__ __forceinline__ float lerpf(float a,float b,float t){return a+(b-a)*t;}
__device__ float lookupDensityCUDA(const float* curve,const float* logE,float val){
    if(val<=logE[0]){
        float t=(val-logE[0])/(logE[1]-logE[0]);
        return lerpf(curve[0],curve[1],t);
    }
    if(val>=logE[600]){
        float t=(val-logE[599])/(logE[600]-logE[599]);
        return lerpf(curve[599],curve[600],t);
    }
    int lo=0;
    for(int i=1;i<601;i++){if(val<logE[i]){lo=i-1;break;}}
    float t=(val-logE[lo])/(logE[lo+1]-logE[lo]);
    return lerpf(curve[lo],curve[lo+1],t);
}

__global__ void ApplyCorrectionKernel(float* img,const float* corr,int W,int H){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=W*H) return;
    int base = idx*4;
    // Subtract correction from density and convert to light
    float densR = img[base+0] - corr[idx*3+0];
    float densG = img[base+1] - corr[idx*3+1];
    float densB = img[base+2] - corr[idx*3+2];
    img[base+0] = powf(10.0f, -densR);
    img[base+1] = powf(10.0f, -densG);
    img[base+2] = powf(10.0f, -densB);
}

extern "C" void UploadDirMatrixCUDA(const float* M){cudaMemcpyToSymbol(c_dirM,M,9*sizeof(float));}
extern "C" void UploadDirParamsCUDA(const float* dmax,float highShift,float sigmaPx){
    cudaMemcpyToSymbol(c_dMax,dmax,3*sizeof(float));
    cudaMemcpyToSymbol(c_highShift,&highShift,sizeof(float));
    cudaMemcpyToSymbol(c_sigmaPx,&sigmaPx,sizeof(float));
    if(sigmaPx>0.0f){
        BuildGaussianKernelHost(sigmaPx);
    } else {
        int r=0;
        cudaMemcpyToSymbol(c_radius,&r,sizeof(int));
    }
}

extern "C" void LaunchDirCouplerCUDA(float* img,int W,int H){
    int total=W*H;
    dim3 blk(256);
    dim3 grd((total+255)/256);
    // allocate temp buffers on device
    float* d_corr; cudaMalloc(&d_corr,total*3*sizeof(float));
    float* d_tmp; cudaMalloc(&d_tmp,total*3*sizeof(float));
    // Build correction
    BuildCorrectionKernel<<<grd,blk>>>(img,d_corr,W,H);
    cudaDeviceSynchronize();
    int r; cudaMemcpyFromSymbol(&r,c_radius,sizeof(int));
    if(r>0){
        dim3 b2(16,16); dim3 g2((W+15)/16,(H+15)/16);
        HorizontalBlur<<<g2,b2>>>(d_corr,d_tmp,W,H);
        VerticalBlur<<<g2,b2>>>(d_tmp,d_corr,W,H);
        cudaDeviceSynchronize();
    }
    // apply correction
    ApplyCorrectionKernel<<<grd,blk>>>(img,d_corr,W,H);
    cudaDeviceSynchronize();
    cudaFree(d_corr); cudaFree(d_tmp);
} 