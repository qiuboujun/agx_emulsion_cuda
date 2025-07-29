#ifndef DYNAMIC_SPECTRAL_UPSAMPLING_CUH
#define DYNAMIC_SPECTRAL_UPSAMPLING_CUH

#include <cuda_runtime.h>
#include <vector>

// Load spectral LUT from file
extern "C" bool LoadSpectralLUTCUDA(const char* filename);

// Launch dynamic spectral upsampling kernel (exact Python implementation)
extern "C" void LaunchDynamicSpectralUpsamplingCUDA(float* img, int width, int height);

// Check if spectral LUT is valid
extern "C" bool IsSpectralLUTValid();

#endif // DYNAMIC_SPECTRAL_UPSAMPLING_CUH 