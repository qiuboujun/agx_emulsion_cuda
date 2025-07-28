#pragma once
#include <cuda_runtime.h>

extern "C" void LaunchEmulsionCUDA(float* img, int width, int height, float gamma); 