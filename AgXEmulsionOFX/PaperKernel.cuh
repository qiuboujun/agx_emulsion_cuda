#pragma once

extern "C" void UploadPaperLUTCUDA(const float* logE,const float* r,const float* g,const float* b);
extern "C" void LaunchPaperCUDA(float* img,int width,int height);
extern "C" void UploadPaperSpectraCUDA(const float* c,const float* m,const float* y,const float* dmin,int n); 