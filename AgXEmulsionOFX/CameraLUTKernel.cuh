#pragma once
#include <stdint.h>
#include <cuda_fp16.h>
// Upload 2D exposure LUT (half precision, sizeX*sizeY*3 entries)
extern "C" void UploadExposureLUTCUDA(const uint16_t* lut,int sizeX,int sizeY);
// Upload RGB->XYZ 3x3 matrix for input colour space
extern "C" void UploadCameraMatrixCUDA(const float* m33);
// Run camera LUT on image (RGBA float buffer)
extern "C" void LaunchCameraLUTCUDA(float* img,int width,int height);
extern "C" bool IsCameraLUTValid(); 