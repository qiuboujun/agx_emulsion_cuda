/* DIR Coupler CUDA kernels - apply exposure inhibition between CMY dye layers
   Ported from ref/agx_emulsion/model/couplers.py
*/
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// Upload 3x3 DIR matrix (row-major)
void UploadDirMatrixCUDA(const float* M);
// Upload per-layer density_max[3], high exposure shift scalar, pixel sigma for XY diffusion
void UploadDirParamsCUDA(const float* dmax, float highShift, float sigmaPx);
// Launch negative development pipeline: assumes logE image in-place, outputs CMY density in-place
void LaunchDirCouplerCUDA(float* logE, int width, int height);

#ifdef __cplusplus
}
#endif 