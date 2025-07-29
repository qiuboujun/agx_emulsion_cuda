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

// Mitchell-Netravali cubic interpolation kernel (exact match to Python)
__device__ float mitchell_weight(float t, float B = 1.0f/3.0f, float C = 1.0f/3.0f) {
    float x = fabsf(t);
    if (x < 1.0f) {
        return (1.0f/6.0f) * ((12.0f - 9.0f*B - 6.0f*C)*x*x*x + 
                              (-18.0f + 12.0f*B + 6.0f*C)*x*x + 
                              (6.0f - 2.0f*B));
    } else if (x < 2.0f) {
        return (1.0f/6.0f) * ((-B - 6.0f*C)*x*x*x + 
                              (6.0f*B + 30.0f*C)*x*x + 
                              (-12.0f*B - 48.0f*C)*x + 
                              (8.0f*B + 24.0f*C));
    } else {
        return 0.0f;
    }
}

// Safe index with symmetric reflection (exact match to Python)
__device__ int safe_index(int idx, int L) {
    if (idx < 0) {
        return -idx;
    } else if (idx >= L) {
        return 2*(L - 1) - idx;
    } else {
        return idx;
    }
}

// Mitchell-Netravali cubic interpolation at 2D coordinates (exact match to Python)
__device__ void cubic_interp_lut_at_2d(float2 coord, float* outRaw) {
    if (!d_spectralLUT) {
        outRaw[0] = outRaw[1] = outRaw[2] = 0.0f;
        return;
    }
    
    int L = g_lutW;  // LUT is square (192x192)
    
    // Convert normalized coordinates [0,1] to pixel coordinates [0, L-1]
    float x = coord.x * (L - 1);
    float y = coord.y * (L - 1);
    
    int x_base = (int)floorf(x);
    int y_base = (int)floorf(y);
    float x_frac = x - x_base;
    float y_frac = y - y_base;
    
    // Compute Mitchell-Netravali kernel weights for x and y dimensions
    float wx[4], wy[4];
    wx[0] = mitchell_weight(x_frac + 1.0f);
    wx[1] = mitchell_weight(x_frac);
    wx[2] = mitchell_weight(x_frac - 1.0f);
    wx[3] = mitchell_weight(x_frac - 2.0f);
    wy[0] = mitchell_weight(y_frac + 1.0f);
    wy[1] = mitchell_weight(y_frac);
    wy[2] = mitchell_weight(y_frac - 1.0f);
    wy[3] = mitchell_weight(y_frac - 2.0f);
    
    // Accumulate weighted sum over the 4x4 neighborhood
    float out[3] = {0.0f, 0.0f, 0.0f};
    float weight_sum = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        int xi = safe_index(x_base - 1 + i, L);
        for (int j = 0; j < 4; j++) {
            int yj = safe_index(y_base - 1 + j, L);
            float weight = wx[i] * wy[j];
            weight_sum += weight;
            
            // Memory layout: [coordinate][3 RGB channels]
            int coord_idx = yj * g_lutW + xi;
            for (int c = 0; c < 3; c++) {
                out[c] += weight * d_spectralLUT[coord_idx * 3 + c];
            }
        }
    }
    
    // Normalize by weight sum
    if (weight_sum != 0.0f) {
        for (int c = 0; c < 3; c++) {
            outRaw[c] = out[c] / weight_sum;
        }
    } else {
        outRaw[0] = outRaw[1] = outRaw[2] = 0.0f;
    }
}

// Simple bilinear interpolation (matches scipy.interpolate.RegularGridInterpolator default)
__device__ void bilinear_interp_lut_at_2d(float2 coord, float* outRaw) {
    if (!d_spectralLUT) {
        outRaw[0] = outRaw[1] = outRaw[2] = 0.0f;
        return;
    }
    
    int L = g_lutW;  // LUT is square (192x192)
    
    // Convert normalized coordinates [0,1] to pixel coordinates [0, L-1]
    float x = coord.x * (L - 1);
    float y = coord.y * (L - 1);
    
    // Clamp to valid range
    x = fmaxf(0.0f, fminf((float)(L-1), x));
    y = fmaxf(0.0f, fminf((float)(L-1), y));
    
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = fminf(x0 + 1, L - 1);
    int y1 = fminf(y0 + 1, L - 1);
    
    float fx = x - x0;
    float fy = y - y0;
    
    // Get the four corner values
    int idx00 = y0 * g_lutW + x0;
    int idx01 = y0 * g_lutW + x1;  
    int idx10 = y1 * g_lutW + x0;
    int idx11 = y1 * g_lutW + x1;
    
    // Bilinear interpolation for each RGB channel
    for (int c = 0; c < 3; c++) {
        float v00 = d_spectralLUT[idx00 * 3 + c];
        float v01 = d_spectralLUT[idx01 * 3 + c];
        float v10 = d_spectralLUT[idx10 * 3 + c];
        float v11 = d_spectralLUT[idx11 * 3 + c];
        
        // Bilinear interpolation
        float v0 = v00 * (1.0f - fx) + v01 * fx;
        float v1 = v10 * (1.0f - fx) + v11 * fx;
        outRaw[c] = v0 * (1.0f - fy) + v1 * fy;
    }
}

__device__ void fetchRawFromTC(float2 tc, float* outRaw) {
    // Use cubic interpolation (matches Python's apply_lut_cubic_2d)
    cubic_interp_lut_at_2d(tc, outRaw);
}

__global__ void DynamicSpectralUpsamplingKernel(float* img, int width, int height) {
    int total_threads = blockDim.x * gridDim.x;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to handle all pixels
    for (int idx = thread_id; idx < width * height; idx += total_threads) {
        int x = idx % width;
        int y = idx / width;
        
        int base = idx * 4; // Use 4-float stride for RGBA format
        float3 input_rgb = make_float3(img[base], img[base + 1], img[base + 2]);
    
        // Debug print for ONLY the first thread and first pixel to avoid buffer overflow
        bool is_first_thread = (thread_id == 0 && idx == 0);
    if (is_first_thread) {
        printf("SpectralKernel DEBUG: First thread Input RGB=(%f,%f,%f)\n", input_rgb.x, input_rgb.y, input_rgb.z);
    }

    // Convert RGB to XYZ using ACES2065-1 matrix
    float3 xyz = rgb_to_xyz(input_rgb);
    float brightness = xyz.x + xyz.y + xyz.z;
    
        if (brightness <= 0.0f) {
            img[base + 0] = 0.0f;
            img[base + 1] = 0.0f;
            img[base + 2] = 0.0f;
            continue;
        }
    
    // Get xy chromaticity coordinates
    float2 xy = make_float2(xyz.x / brightness, xyz.y / brightness);
    
    if (is_first_thread) {
        printf("SpectralKernel DEBUG: XYZ=(%f,%f,%f) brightness=%f\n", xyz.x, xyz.y, xyz.z, brightness);
        printf("SpectralKernel DEBUG: xy=(%f,%f)\n", xy.x, xy.y);
    }
    
    // Convert xy to tc using tri2quad transform
    float2 tc = tri2quad(xy);
    
    if (is_first_thread) {
        printf("SpectralKernel DEBUG: tc=(%f,%f)\n", tc.x, tc.y);
    }
    
    // Fetch pre-multiplied raw RGB values using bilinear interpolation
    float raw_rgb[3];
    fetchRawFromTC(tc, raw_rgb);
    
    if (is_first_thread) {
        printf("SpectralKernel DEBUG: raw_before_scaling=(%f,%f,%f)\n", raw_rgb[0], raw_rgb[1], raw_rgb[2]);
        printf("SpectralKernel DEBUG: LUT dimensions=(%d,%d)\n", g_lutW, g_lutH);
        
        // Test: manually fetch a few LUT values to verify data
        if (d_spectralLUT) {
            int test_coord = 100 * g_lutW + 100; // Center of 192x192 LUT
            printf("SpectralKernel DEBUG: LUT[100,100]=[%f,%f,%f]\n", 
                   d_spectralLUT[test_coord * 3 + 0], 
                   d_spectralLUT[test_coord * 3 + 1], 
                   d_spectralLUT[test_coord * 3 + 2]);
            
            // Convert tc to pixel coordinates
            float x_pix = tc.x * (g_lutW - 1);
            float y_pix = tc.y * (g_lutH - 1);
            printf("SpectralKernel DEBUG: tc_pixel_coords=(%f,%f)\n", x_pix, y_pix);
        } else {
            printf("SpectralKernel DEBUG: d_spectralLUT is NULL!\n");
        }
    }
    
    // Scale by brightness (matching Python: raw *= b[...,None])
    raw_rgb[0] *= brightness;
    raw_rgb[1] *= brightness;
    raw_rgb[2] *= brightness;
    
    if (is_first_thread) {
        printf("SpectralKernel DEBUG: raw_after_scaling=(%f,%f,%f)\n", raw_rgb[0], raw_rgb[1], raw_rgb[2]);
    }
    
    // Calculate midgray normalization (exactly as Python: 0.184 linear)
    float3 midgray_rgb = make_float3(0.184f, 0.184f, 0.184f);
    float3 midgray_xyz = rgb_to_xyz(midgray_rgb);
    float midgray_brightness = midgray_xyz.x + midgray_xyz.y + midgray_xyz.z;
    float2 midgray_xy = make_float2(midgray_xyz.x/midgray_brightness, midgray_xyz.y/midgray_brightness);
    float2 midgray_tc = tri2quad(midgray_xy);
    
    // Fetch midgray raw RGB values using cubic interpolation (matches Python)
    float midgray_raw[3];
    fetchRawFromTC(midgray_tc, midgray_raw);
    
    // Scale midgray by brightness
    midgray_raw[0] *= midgray_brightness;
    midgray_raw[1] *= midgray_brightness;
    midgray_raw[2] *= midgray_brightness;
    
    // Normalize by green midgray (as in Python: raw / raw_midgray[1])
    float normalization = 1.0f / fmaxf(midgray_raw[1], 1e-10f);
    float cmy_r = fmaxf(raw_rgb[0] * normalization, 1e-6f);
    float cmy_g = fmaxf(raw_rgb[1] * normalization, 1e-6f);
    float cmy_b = fmaxf(raw_rgb[2] * normalization, 1e-6f);

    if (is_first_thread) {
        printf("SpectralKernel DEBUG: midgray_raw=(%f,%f,%f) norm=%f\n", 
               midgray_raw[0], midgray_raw[1], midgray_raw[2], normalization);
        printf("SpectralKernel DEBUG: Final CMY=(%f,%f,%f)\n", cmy_r, cmy_g, cmy_b);
    }

        // Store final CMY values using correct 4-float stride
        img[base + 0] = cmy_r;     // C
        img[base + 1] = cmy_g;     // M
        img[base + 2] = cmy_b;     // Y
        // img[base + 3] (alpha) left unchanged
    }
}

// Load spectral LUT from file (placeholder for now)
extern "C" bool LoadSpectralLUTCUDA(const char* filename) {
    printf("DEBUG: Loading spectral LUT from %s (placeholder)\n", filename);
    // For now, we'll use hardcoded values that match Python exactly
    return true;
}

// Launch dynamic spectral upsampling kernel
extern "C" void LaunchDynamicSpectralUpsamplingCUDA(float* img, int width, int height) {
    // Increase printf buffer size to handle debug output
    size_t current_size;
    cudaDeviceGetLimit(&current_size, cudaLimitPrintfFifoSize);
    if (current_size < 10 * 1024 * 1024) { // 10MB
        cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024);
        printf("DEBUG: Increased CUDA printf buffer to 10MB\n");
    }
    
    int total = width * height;
    int block = 256;
    int grid = (total + block - 1) / block;
    
    printf("DEBUG: Launching SpectralUpsampling kernel grid=%d block=%d total=%d\n", grid, block, total);
    DynamicSpectralUpsamplingKernel<<<grid, block>>>(img, width, height);
    
    // Force synchronization to flush printf buffer
    cudaDeviceSynchronize();
    printf("DEBUG: SpectralUpsampling kernel completed\n");
}

// Check if spectral LUT is valid
extern "C" bool IsSpectralLUTValid() {
    // For now, always return true since we're using hardcoded values
    return true;
} 

extern "C" bool UploadSpectralLUTCUDA(const float* h_data,int w,int h,int samples){
    printf("DEBUG: UploadSpectralLUTCUDA called with w=%d, h=%d, samples=%d\n", w, h, samples);
    float*dptr; 
    cudaError_t err=cudaMalloc(&dptr,w*h*samples*sizeof(float));
    if(err!=cudaSuccess){printf("UploadSpectralLUTCUDA malloc error %s\n",cudaGetErrorString(err));return false;}
    err=cudaMemcpy(dptr,h_data,w*h*samples*sizeof(float),cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){printf("UploadSpectralLUTCUDA memcpy error %s\n",cudaGetErrorString(err));cudaFree(dptr);return false;}
    cudaMemcpyToSymbol(g_lutW,&w,sizeof(int));
    cudaMemcpyToSymbol(g_lutH,&h,sizeof(int));
    cudaMemcpyToSymbol(g_lutS,&samples,sizeof(int)); // Now 3 for RGB channels, not 81 for spectral
    cudaMemcpyToSymbol(d_spectralLUT,&dptr,sizeof(float*));
    printf("DEBUG: Uploaded pre-multiplied spectral LUT to CUDA device memory\n");
    printf("DEBUG: LUT layout: [%dx%d][%d channels] (RGB channels, not spectral samples)\n", w, h, samples);
    return true;
}

extern "C" bool UploadCameraSensCUDA(const float* sensR,const float* sensG,const float* sensB,int samples){
    if(samples!=81){printf("Camera sens size must be 81\n");return false;}
    cudaMemcpyToSymbol(c_sensR,sensR,81*sizeof(float));
    cudaMemcpyToSymbol(c_sensG,sensG,81*sizeof(float));
    cudaMemcpyToSymbol(c_sensB,sensB,81*sizeof(float));
    return true;
}

// fetchSpectrum function removed - now using fetchRawFromTC with pre-multiplied LUT 