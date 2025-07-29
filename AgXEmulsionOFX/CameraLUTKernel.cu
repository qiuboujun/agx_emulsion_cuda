#include "CameraLUTKernel.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>

static cudaArray_t g_lutArray=nullptr;
static cudaTextureObject_t g_tex=0;
static int g_lutW=0,g_lutH=0;
__constant__ float c_rgb2xyz[9];

extern "C" void UploadCameraMatrixCUDA(const float* m33){
    cudaMemcpyToSymbol(c_rgb2xyz,m33,9*sizeof(float));
}

extern "C" bool IsCameraLUTValid(){
    return g_tex != 0;
}

extern "C" void UploadExposureLUTCUDA(const uint16_t* lut,int w,int h){
    printf("DEBUG: UploadExposureLUTCUDA called with w=%d, h=%d\n", w, h);
    if(!lut) { printf("ERROR: lut pointer is null!\n"); return; }
    
    if(g_tex){cudaDestroyTextureObject(g_tex);g_tex=0;}
    if(g_lutArray){cudaFreeArray(g_lutArray);g_lutArray=nullptr;}
    g_lutW=w;g_lutH=h;
    
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
    cudaError_t err = cudaMallocArray(&g_lutArray,&desc,w,h);
    if(err != cudaSuccess) { printf("ERROR: cudaMallocArray failed: %s\n", cudaGetErrorString(err)); return; }
    printf("DEBUG: Allocated CUDA array %dx%d\n", w, h);
    
    // convert planar half3 to float4 RGBA with A=1
    float4* tmp = new float4[w*h];
    for(int j=0;j<h;j++){
        for(int i=0;i<w;i++){
            int idx=j*w+i;
            // The data is stored as float16 in row-major order, R index varying fastest
            // So for each pixel (i,j), we have 3 consecutive float16 values: R, G, B
            uint16_t half_r = lut[idx*3+0];
            uint16_t half_g = lut[idx*3+1];
            uint16_t half_b = lut[idx*3+2];
            
            // Convert uint16_t to __half, then to float
            __half h_r, h_g, h_b;
            memcpy(&h_r, &half_r, sizeof(uint16_t));
            memcpy(&h_g, &half_g, sizeof(uint16_t));
            memcpy(&h_b, &half_b, sizeof(uint16_t));
            
            tmp[idx].x = __half2float(h_r);
            tmp[idx].y = __half2float(h_g);
            tmp[idx].z = __half2float(h_b);
            tmp[idx].w = 1.0f;
            
            // Debug first few values
            if(i < 3 && j < 3) {
                printf("DEBUG: LUT[%d,%d] = (%f,%f,%f,%f) [half: %04x,%04x,%04x]\n", 
                       i, j, tmp[idx].x, tmp[idx].y, tmp[idx].z, tmp[idx].w, half_r, half_g, half_b);
            }
        }
    }
    printf("DEBUG: Converted LUT data to float4 format\n");
    
    err = cudaMemcpy2DToArray(g_lutArray,0,0,tmp,w*sizeof(float4),w*sizeof(float4),h,cudaMemcpyHostToDevice);
    if(err != cudaSuccess) { printf("ERROR: cudaMemcpy2DToArray failed: %s\n", cudaGetErrorString(err)); delete[] tmp; return; }
    printf("DEBUG: Copied LUT data to GPU\n");
    
    delete[] tmp;
    cudaResourceDesc res{};res.resType=cudaResourceTypeArray;res.res.array.array=g_lutArray;
    cudaTextureDesc tex{};tex.addressMode[0]=cudaAddressModeClamp;tex.addressMode[1]=cudaAddressModeClamp;tex.filterMode=cudaFilterModeLinear;tex.readMode=cudaReadModeElementType;tex.normalizedCoords=1;
    err = cudaCreateTextureObject(&g_tex,&res,&tex,nullptr);
    if(err != cudaSuccess) { printf("ERROR: cudaCreateTextureObject failed: %s\n", cudaGetErrorString(err)); return; }
    printf("DEBUG: Created texture object successfully\n");
    printf("DEBUG: Texture config: normalizedCoords=%d, readMode=%d, filterMode=%d\n", 
           tex.normalizedCoords, tex.readMode, tex.filterMode);
}

__device__ float3 rgb_to_xy(const float3& rgb){
    float X = c_rgb2xyz[0]*rgb.x + c_rgb2xyz[1]*rgb.y + c_rgb2xyz[2]*rgb.z;
    float Y = c_rgb2xyz[3]*rgb.x + c_rgb2xyz[4]*rgb.y + c_rgb2xyz[5]*rgb.z;
    float Z = c_rgb2xyz[6]*rgb.x + c_rgb2xyz[7]*rgb.y + c_rgb2xyz[8]*rgb.z;
    float sum = X+Y+Z+1e-8f;
    float x = X/sum;
    float y = Y/sum;
    // Simple xy coordinates (0-1 range) as used in compute_lut_spectra
    return make_float3(x,y,0.0f);
}

__global__ void CameraLUTKernel(float* img,int W,int H,cudaTextureObject_t texObj){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total=W*H;
    if(idx>=total) return;
    
    // Debug center pixel only
    bool isCenter = (idx == (H/2) * W + W/2);
    
    float4* p4 = reinterpret_cast<float4*>(img);
    float4 p = p4[idx];
    float3 rgb = make_float3(p.x,p.y,p.z);
    
    if(isCenter) {
        printf("CameraLUT DEBUG: Input RGB=(%f,%f,%f)\n", rgb.x, rgb.y, rgb.z);
    }
    
    float3 tc = rgb_to_xy(rgb);
    float u = fminf(fmaxf(tc.x,0.f),1.f);
    float v = fminf(fmaxf(tc.y,0.f),1.f);
    
    if(isCenter) {
        printf("CameraLUT DEBUG: xy=(%f,%f) -> uv=(%f,%f)\n", tc.x, tc.y, u, v);
        printf("CameraLUT DEBUG: uv clamped to [0,1]: u=%f, v=%f\n", u, v);
    }
    
    // Check if coordinates are valid
    if(u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        if(isCenter) {
            printf("CameraLUT ERROR: Invalid texture coordinates u=%f, v=%f\n", u, v);
        }
        return;
    }
    
    float4 samp = tex2D<float4>(texObj,u,v);
    
    if(isCenter) {
        printf("CameraLUT DEBUG: Sampled CMY=(%f,%f,%f)\n", samp.x, samp.y, samp.z);
        // Check for NaN
        if(isnan(samp.x) || isnan(samp.y) || isnan(samp.z)) {
            printf("CameraLUT ERROR: NaN detected in sampled values!\n");
        }
    }
    
    p.x = samp.x; p.y = samp.y; p.z = samp.z;
    p4[idx]=p;
}

extern "C" void LaunchCameraLUTCUDA(float* img,int W,int H){
    if(!g_tex){printf("Camera LUT texture not uploaded!\n");return;}
    int block=256;int grid=(W*H+block-1)/block;
    CameraLUTKernel<<<grid,block>>>(img,W,H,g_tex);
    cudaDeviceSynchronize();
} 