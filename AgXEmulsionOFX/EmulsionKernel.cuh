#pragma once
#include <cuda_runtime.h>

extern "C" void LaunchEmulsionCUDA(float* img, int width, int height, float gamma);
extern "C" void UploadLUTCUDA(const float* logE,const float* r,const float* g,const float* b); 