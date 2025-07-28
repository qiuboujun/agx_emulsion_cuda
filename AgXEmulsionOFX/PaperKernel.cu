#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "PaperKernel.cuh"
#include "CIE1931.cuh"

__constant__ float p_logE[601];
__constant__ float p_curveR[601];
__constant__ float p_curveG[601];
__constant__ float p_curveB[601];
__constant__ int p_lutSize; // actual number of valid samples
__constant__ float p_specC[200];
__constant__ float p_specM[200];
__constant__ float p_specY[200];
__constant__ float p_specDmin[200];
__constant__ int  p_specN;
__constant__ float c_viewSPD[81];
__constant__ float c_catScale[3];

extern "C" void UploadViewSPDCUDA(const float* spd,int n){
    if(n>81) n=81;
    cudaMemcpyToSymbol(c_viewSPD,spd,n*sizeof(float));
    printf("UploadViewSPDCUDA DEBUG: Uploaded %d SPD samples (first3=%f %f %f)\n",n, spd[0], spd[1], spd[2]);
}

__device__ __forceinline__ float lerpP(float a,float b,float t){return a+(b-a)*t;}

__device__ float interpCurve(const float* curve,const float* logE,float v){
    // Debug for center pixel only
    if(threadIdx.x == 8 && threadIdx.y == 8 && blockIdx.x == 60 && blockIdx.y == 34) {
        printf("interpCurve DEBUG: v=%f, logE[0]=%f, logE[end]=%f, lutSize=%d\n", 
               v, logE[0], logE[p_lutSize-1], p_lutSize);
    }
    
    int lastIdx = p_lutSize - 1;
    
    if(v<=logE[0]){
        return curve[0];
    }
    if(v>=logE[lastIdx]){
        return curve[lastIdx];
    }
    int idx=0; for(int i=1;i<p_lutSize;i++){ if(v<logE[i]){idx=i-1;break;} }
    float t=(v-logE[idx])/(logE[idx+1]-logE[idx]);
    float result = lerpP(curve[idx],curve[idx+1],t);
    if(threadIdx.x == 8 && threadIdx.y == 8 && blockIdx.x == 60 && blockIdx.y == 34) {
        printf("interpCurve DEBUG: interpolate, idx=%d, t=%f, result=%f\n", idx, t, result);
    }
    return result;
}

__global__ void PaperKernel(float* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;
    float r = img[idx];
    float g = img[idx + 1];
    float b = img[idx + 2];
    
    // Debug first pixel
    if (x == width/2 && y == height/2) {
        printf("PaperKernel DEBUG: Input RGB=(%f,%f,%f)\n", r, g, b);
        printf("PaperKernel DEBUG: p_specN=%d\n", p_specN);
    }

    // Clamp input light values
    r = fmaxf(1e-6f, fminf(1.0f, r));
    g = fmaxf(1e-6f, fminf(1.0f, g));
    b = fmaxf(1e-6f, fminf(1.0f, b));

    // Convert light to density: D = -log10(T)
    float dr = -log10f(r);
    float dg = -log10f(g);
    float db = -log10f(b);
    
    if (x == width/2 && y == height/2) {
        printf("PaperKernel DEBUG: Light to density RGB=(%f,%f,%f)\n", dr, dg, db);
    }

    // Apply paper curves using LUT interpolation
    float pr = interpCurve(p_curveR, p_logE, dr); // R uses curve R
    float pg = interpCurve(p_curveG, p_logE, dg); // G uses curve G
    float pb = interpCurve(p_curveB, p_logE, db); // B uses curve B

    if (x == width/2 && y == height/2) {
        printf("PaperKernel DEBUG: After paper curves CMY=(%f,%f,%f)\n", pr, pg, pb);
    }

    // Clamp output densities
    pr = fminf(3.0f, pr);
    pg = fminf(3.0f, pg);
    pb = fminf(3.0f, pb);

    // === NEW: Spectral to RGB conversion ===
    // Step 1: Convert CMY density to spectral density using dye spectra
    // Step 2: Convert spectral density to transmitted light  
    // Step 3: Integrate with CIE 1931 2° Standard Observer to get XYZ
    
    float X = 0.0f, Y = 0.0f, Z = 0.0f;
    float normY = 0.0f;
    
    // Debug a few spectral samples
    if (x == width/2 && y == height/2) {
        printf("PaperKernel DEBUG: First 5 spectral samples:\n");
        for(int i = 0; i < 5 && i < p_specN; i++) {
            float sd = pr * p_specC[i] + pg * p_specM[i] + pb * p_specY[i] + p_specDmin[i];
            float light = __powf(10.0f, -sd);
            printf("  i=%d: specC=%f specM=%f specY=%f dmin=%f -> sd=%f light=%f\n", 
                   i, p_specC[i], p_specM[i], p_specY[i], p_specDmin[i], sd, light);
        }
    }
    
    // Integrate over CIE wavelengths (380-780nm, 5nm step, 81 samples)
    for(int j = 0; j < CIE_SAMPLES; j++) {
        // Map CIE wavelength index j to nearest dye spectra index i
        // CIE: 380-780nm (400nm range), Dye: variable range over p_specN samples
        int i = (j * (p_specN - 1)) / (CIE_SAMPLES - 1);
        i = fminf(i, p_specN - 1);
        
        // Compute spectral density at this wavelength
        float spectralDensity = pr * p_specC[i] + pg * p_specM[i] + pb * p_specY[i] + p_specDmin[i];
        
        // Convert to transmitted light
        float transmittedLight = __powf(10.0f, -spectralDensity);
        
        // Weight by viewing illuminant and accumulate XYZ
        X += transmittedLight * c_viewSPD[j] * c_xBar[j];
        Y += transmittedLight * c_viewSPD[j] * c_yBar[j];
        Z += transmittedLight * c_viewSPD[j] * c_zBar[j];
        normY += c_viewSPD[j] * c_yBar[j];
    }
    
    // Normalize by illuminant
    if (normY > 0.0f) {
        X /= normY;
        Y /= normY; 
        Z /= normY;
    }
    // Bradford CAT to D65
    // Convert XYZ to LMS, apply scale, convert back
    float L =  0.8951f*X + 0.2664f*Y - 0.1614f*Z;
    float Mv= -0.7502f*X + 1.7135f*Y + 0.0367f*Z;
    float S  =  0.0389f*X - 0.0685f*Y + 1.0296f*Z;
    L*=c_catScale[0];
    Mv*=c_catScale[1];
    S*=c_catScale[2];
    X = 0.9869929f*L + 0.4323053f*Mv - 0.0085287f*S;
    Y =-0.1470543f*L + 0.5183603f*Mv + 0.0400428f*S;
    Z = 0.1599627f*L + 0.0492912f*Mv + 0.9684867f*S;
    
    // Convert XYZ to linear sRGB (D50 → sRGB matrix)
    float R =  3.2406f * X - 1.5372f * Y - 0.4986f * Z;
    float G = -0.9689f * X + 1.8758f * Y + 0.0415f * Z;
    float B =  0.0557f * X - 0.2040f * Y + 1.0570f * Z;
    
    // Clamp to valid range
    R = fmaxf(0.0f, fminf(1.0f, R));
    G = fmaxf(0.0f, fminf(1.0f, G));
    B = fmaxf(0.0f, fminf(1.0f, B));

    if (x == width/2 && y == height/2) {
        printf("PaperKernel DEBUG: XYZ=(%f,%f,%f) normY=%f -> sRGB=(%f,%f,%f)\n", 
               X, Y, Z, normY, R, G, B);
    }

    img[idx] = R;
    img[idx + 1] = G;
    img[idx + 2] = B;
    // Alpha unchanged
}

extern "C" void UploadPaperLUTCUDA(const float* logE,const float* r,const float* g,const float* b){
    printf("UploadPaperLUTCUDA DEBUG: logE[0]=%f, logE[600]=%f\n", logE[0], logE[600]);
    printf("UploadPaperLUTCUDA DEBUG: r[0]=%f, g[0]=%f, b[0]=%f\n", r[0], g[0], b[0]);
    
    // Find actual data size (non-zero entries)
    int actualSize = 601;
    for(int i = 600; i >= 0; i--) {
        if(logE[i] != 0.0f || r[i] != 0.0f || g[i] != 0.0f || b[i] != 0.0f) {
            actualSize = i + 1;
            break;
        }
    }
    printf("UploadPaperLUTCUDA DEBUG: Detected actual LUT size: %d\n", actualSize);
    
    cudaMemcpyToSymbol(p_logE,logE,601*sizeof(float));
    cudaMemcpyToSymbol(p_curveR,r,601*sizeof(float));
    cudaMemcpyToSymbol(p_curveG,g,601*sizeof(float));
    cudaMemcpyToSymbol(p_curveB,b,601*sizeof(float));
    cudaMemcpyToSymbol(p_lutSize,&actualSize,sizeof(int));
    
    printf("UploadPaperLUTCUDA DEBUG: Upload completed\n");
}

extern "C" void UploadPaperSpectraCUDA(const float* c,const float* m,const float* y,const float* dmin,int n){
    printf("UploadPaperSpectraCUDA DEBUG: Called with n=%d\n", n);
    printf("UploadPaperSpectraCUDA DEBUG: First 5 C values: %f %f %f %f %f\n", 
           c[0], c[1], c[2], c[3], c[4]);
    printf("UploadPaperSpectraCUDA DEBUG: First 5 M values: %f %f %f %f %f\n", 
           m[0], m[1], m[2], m[3], m[4]);
    
    cudaMemcpyToSymbol(p_specC,c,n*sizeof(float));
    cudaMemcpyToSymbol(p_specM,m,n*sizeof(float));
    cudaMemcpyToSymbol(p_specY,y,n*sizeof(float));
    cudaMemcpyToSymbol(p_specDmin,dmin,n*sizeof(float));
    cudaMemcpyToSymbol(p_specN,&n,sizeof(int));
    
    printf("UploadPaperSpectraCUDA DEBUG: Upload completed\n");
}

extern "C" void UploadCATScaleCUDA(const float* scale){
    cudaMemcpyToSymbol(c_catScale,scale,3*sizeof(float));
    printf("UploadCATScaleCUDA DEBUG: scale=(%f,%f,%f)\n",scale[0],scale[1],scale[2]);
}

extern "C" void LaunchPaperCUDA(float* img, int width, int height) {
    printf("LaunchPaperCUDA DEBUG: img=%p, width=%d, height=%d\n", img, width, height);
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    printf("LaunchPaperCUDA DEBUG: Launching kernel with grid(%d,%d) block(%d,%d)\n", 
           grid.x, grid.y, block.x, block.y);
    
    PaperKernel<<<grid, block>>>(img, width, height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("LaunchPaperCUDA ERROR: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    printf("LaunchPaperCUDA DEBUG: Kernel completed\n");
} 