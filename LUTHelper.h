#pragma once

#ifdef __cplusplus
extern "C" {
#endif

bool loadFilmLUT(const char* stock, float* logE, float* r, float* g, float* b);
bool loadPrintLUT(const char* stock, float* logE, float* r, float* g, float* b);
bool loadPaperSpectra(const char* stock, float* c, float* m, float* y, float* dmin, int* count);

#ifdef __cplusplus
}
#endif 