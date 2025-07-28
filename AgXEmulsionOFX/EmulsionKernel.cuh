#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void LaunchEmulsionCUDA(float4* d_pixels, int width, int height, float exposureEV);
void UploadLUTCUDA(const float* logE, const float* r, const float* g, const float* b, float gamma);

#ifdef __cplusplus
}
#endif 