#pragma once

#ifdef __cplusplus
extern "C" {
#endif

bool loadFilmLUT(const char* stock, float* logE, float* r, float* g, float* b);
bool loadPrintLUT(const char* stock, float* logE, float* r, float* g, float* b, char* illuminant_out);
bool loadPaperSpectra(const char* stock, float* c, float* m, float* y, float* dmin, int* count);
bool loadIlluminantSPD(const char* name, float* spd, int* count);

#ifdef __cplusplus
}
#endif 