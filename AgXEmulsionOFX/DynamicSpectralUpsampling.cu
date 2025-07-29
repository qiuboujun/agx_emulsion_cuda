#include "DynamicSpectralUpsampling.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

// Constants for spectral processing
static const int SPECTRAL_WAVELENGTHS = 81;
static const int SPECTRAL_LUT_SIZE = 128;
static const float SPECTRAL_MIN_WL = 360.0f;
static const float SPECTRAL_MAX_WL = 800.0f;

// Device memory for spectral LUT
static cudaArray_t g_spectralLUTArray = nullptr;
static cudaTextureObject_t g_spectralLUTTexture = 0;

// === New global/constant buffers for LUT & sensitivities ===
__device__ float* d_spectralLUT = nullptr; // size = lutW*lutH*samples
__device__ int  g_lutW=0, g_lutH=0, g_lutS=0; // stored in device globals for quick access

__constant__ float c_sensR[81];
__constant__ float c_sensG[81];
__constant__ float c_sensB[81];

// ACES2065-1 RGB to XYZ matrix (same as Python)
__constant__ float c_rgb2xyz[9] = {
    0.9525524f, 0.0000000f, 0.0000937f,
    0.3439664f, 0.7281661f, -0.0721325f,
    0.0000000f, 0.0000000f, 1.0088252f
};

// Illuminant xy coordinates (D55)
__constant__ float c_illuminant_xy[2] = {0.3324f, 0.3474f}; // D55

// Band pass filter parameters
__constant__ float c_filter_uv[3] = {1.0f, 410.0f, 8.0f};  // amplitude, wavelength, width
__constant__ float c_filter_ir[3] = {1.0f, 675.0f, 15.0f}; // amplitude, wavelength, width

// Sigmoid error function approximation
__device__ float sigmoid_erf(float x, float center, float width) {
    float t = (x - center) / width;
    // Approximation of erf(t) * 0.5 + 0.5
    float sign = (t >= 0.0f) ? 1.0f : -1.0f;
    float abs_t = fabsf(t);
    float result = 1.0f / (1.0f + expf(-2.0f * abs_t));
    return 0.5f + 0.5f * sign * (2.0f * result - 1.0f);
}

// Convert triangular coordinates to square coordinates
__device__ float2 tri2quad(float2 tc) {
    float tx = tc.x;
    float ty = tc.y;
    float y = ty / fmaxf(1.0f - tx, 1e-10f);
    float x = (1.0f - tx) * (1.0f - tx);
    x = fmaxf(0.0f, fminf(1.0f, x));
    y = fmaxf(0.0f, fminf(1.0f, y));
    return make_float2(x, y);
}

// Convert square coordinates to triangular coordinates
__device__ float2 quad2tri(float2 xy) {
    float x = xy.x;
    float y = xy.y;
    float tx = 1.0f - sqrtf(x);
    float ty = y * sqrtf(x);
    return make_float2(tx, ty);
}

// RGB to XYZ conversion
__device__ float3 rgb_to_xyz(float3 rgb) {
    float X = c_rgb2xyz[0] * rgb.x + c_rgb2xyz[1] * rgb.y + c_rgb2xyz[2] * rgb.z;
    float Y = c_rgb2xyz[3] * rgb.x + c_rgb2xyz[4] * rgb.y + c_rgb2xyz[5] * rgb.z;
    float Z = c_rgb2xyz[6] * rgb.x + c_rgb2xyz[7] * rgb.y + c_rgb2xyz[8] * rgb.z;
    return make_float3(X, Y, Z);
}

// RGB to triangular coordinates and brightness
__device__ void rgb_to_tc_b(float3 rgb, float2* tc, float* b) {
    // Convert RGB to XYZ
    float3 xyz = rgb_to_xyz(rgb);
    
    // Calculate brightness (sum of XYZ)
    *b = xyz.x + xyz.y + xyz.z;
    
    // Convert to xy coordinates
    float sum_xyz = fmaxf(*b, 1e-10f);
    float2 xy = make_float2(xyz.x / sum_xyz, xyz.y / sum_xyz);
    xy.x = fmaxf(0.0f, fminf(1.0f, xy.x));
    xy.y = fmaxf(0.0f, fminf(1.0f, xy.y));
    
    // Convert to triangular coordinates
    *tc = tri2quad(xy);
    
    // Handle NaN values
    if (isnan(*b)) *b = 0.0f;
}

// after constants
__device__ void fetchSpectrum(float2 tc,float* outSpectrum);

// Main spectral upsampling kernel (exact Python implementation)
__global__ void DynamicSpectralUpsamplingKernel(float* img, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    
    // Debug center pixel only
    bool isCenter = (idx == (height/2) * width + width/2);
    
    float4* p4 = reinterpret_cast<float4*>(img);
    float4 p = p4[idx];
    float3 rgb = make_float3(p.x, p.y, p.z);
    
    if (isCenter) {
        printf("DynamicSpectral DEBUG: Input RGB=(%f,%f,%f)\n", rgb.x, rgb.y, rgb.z);
    }
    
    // ==== Real spectral upsampling ====
    // Step 1 RGB->XYZ (ACES matrix already constant)
    float3 xyz = rgb_to_xyz(rgb);
    float sumXYZ = xyz.x+xyz.y+xyz.z + 1e-10f;
    float2 xy = make_float2(xyz.x/sumXYZ, xyz.y/sumXYZ);
    // step tri->quad
    float2 tc = tri2quad(xy);

    // fetch spectrum
    float spectrum[81];
    fetchSpectrum(tc,spectrum);

    // Multiply spectrum by pixel brightness (sum XYZ) as in Python
    float brightness = sumXYZ; // already computed above
    for(int i=0;i<81;i++) spectrum[i]*=brightness;

    // dot with camera sensitivity curves
    float rawR=0,rawG=0,rawB=0;
    for(int i=0;i<81;i++){
        rawR += spectrum[i]*c_sensR[i];
        rawG += spectrum[i]*c_sensG[i];
        rawB += spectrum[i]*c_sensB[i];
    }

    // CMY light = raw counts (prevent zeros)
    float cmy_r = fmaxf(rawR,1e-6f);
    float cmy_g = fmaxf(rawG,1e-6f);
    float cmy_b = fmaxf(rawB,1e-6f);
    
    if (isCenter) {
        printf("DynamicSpectral DEBUG: Output CMY=(%f,%f,%f)\n", cmy_r, cmy_g, cmy_b);
    }
    
    p.x = cmy_r;
    p.y = cmy_g;
    p.z = cmy_b;
    
    p4[idx] = p;
}

// Load spectral LUT from file (placeholder for now)
extern "C" bool LoadSpectralLUTCUDA(const char* filename) {
    printf("DEBUG: Loading spectral LUT from %s (placeholder)\n", filename);
    // For now, we'll use hardcoded values that match Python exactly
    return true;
}

// Launch dynamic spectral upsampling kernel
extern "C" void LaunchDynamicSpectralUpsamplingCUDA(float* img, int width, int height) {
    int total = width * height;
    int block = 256;
    int grid = (total + block - 1) / block;
    
    DynamicSpectralUpsamplingKernel<<<grid, block>>>(img, width, height);
    cudaDeviceSynchronize();
}

// Check if spectral LUT is valid
extern "C" bool IsSpectralLUTValid() {
    // For now, always return true since we're using hardcoded values
    return true;
} 

extern "C" bool UploadSpectralLUTCUDA(const float* h_data,int w,int h,int samples){
    size_t sz = (size_t)w*h*samples*sizeof(float);
    float* dptr=nullptr;
    cudaError_t err=cudaMalloc(&dptr, sz);
    if(err!=cudaSuccess){printf("UploadSpectralLUTCUDA malloc error %s\n",cudaGetErrorString(err));return false;}
    err=cudaMemcpy(dptr,h_data,sz,cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){printf("UploadSpectralLUTCUDA memcpy error %s\n",cudaGetErrorString(err));cudaFree(dptr);return false;}
    cudaMemcpyToSymbol(g_lutW,&w,sizeof(int));
    cudaMemcpyToSymbol(g_lutH,&h,sizeof(int));
    cudaMemcpyToSymbol(g_lutS,&samples,sizeof(int));
    cudaMemcpyToSymbol(d_spectralLUT,&dptr,sizeof(float*));
    printf("Spectral LUT uploaded: %dx%dx%d\n",w,h,samples);
    return true;
}

extern "C" bool UploadCameraSensCUDA(const float* sensR,const float* sensG,const float* sensB,int samples){
    if(samples!=81){printf("Camera sens size must be 81\n");return false;}
    cudaMemcpyToSymbol(c_sensR,sensR,81*sizeof(float));
    cudaMemcpyToSymbol(c_sensG,sensG,81*sizeof(float));
    cudaMemcpyToSymbol(c_sensB,sensB,81*sizeof(float));
    return true;
}

// Bilinear fetch helper
__device__ void fetchSpectrum(float2 tc,float* outSpectrum){
    if(!d_spectralLUT) { for(int i=0;i<81;i++) outSpectrum[i]=0; return;}
    // clamp tc 0..1
    tc.x=fminf(fmaxf(tc.x,0.f),0.99999f);
    tc.y=fminf(fmaxf(tc.y,0.f),0.99999f);
    int x = tc.x * (g_lutW-1);
    int y = tc.y * (g_lutH-1);
    float fx = tc.x*(g_lutW-1)-x;
    float fy = tc.y*(g_lutH-1)-y;
    int x1 = min(x+1,g_lutW-1);
    int y1 = min(y+1,g_lutH-1);
    size_t stride = (size_t)g_lutW*g_lutH;
    // For each spectral sample do bilinear
    for(int k=0;k<g_lutS;k++){
        float s00 = d_spectralLUT[(k* g_lutH + y)*g_lutW + x];
        float s10 = d_spectralLUT[(k* g_lutH + y)*g_lutW + x1];
        float s01 = d_spectralLUT[(k* g_lutH + y1)*g_lutW + x];
        float s11 = d_spectralLUT[(k* g_lutH + y1)*g_lutW + x1];
        float s0 = s00 + (s10-s00)*fx;
        float s1 = s01 + (s11-s01)*fx;
        outSpectrum[k] = s0 + (s1-s0)*fy;
    }
} 