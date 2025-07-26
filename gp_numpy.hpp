#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <string>
#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>

// ----------------- Error macros -----------------
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                                  \
    do {                                                                                  \
        cudaError_t __err = (expr);                                                       \
        if (__err != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(__err),      \
                    __FILE__, __LINE__);                                                  \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)
#endif

#ifndef CUFFT_CHECK
#define CUFFT_CHECK(expr)                                                                 \
    do {                                                                                  \
        cufftResult __cerr = (expr);                                                      \
        if (__cerr != CUFFT_SUCCESS) {                                                    \
            fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)__cerr, __FILE__, __LINE__);\
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)
#endif

namespace gpnp {

// Type aliases
using dvec  = thrust::device_vector<double>;
using hvec  = thrust::host_vector<double>;
using cvec  = thrust::device_vector<thrust::complex<double>>;
using hcvec = thrust::host_vector<thrust::complex<double>>;

// Creation
hvec arange_h_np(double start, double stop, double step);
dvec  arange(double start, double stop, double step);
hvec  linspace_h(double start, double stop, int num);
dvec  linspace(double start, double stop, int num);
dvec  zeros(size_t n);
dvec  ones(size_t n, double v = 1.0);
dvec  empty(size_t n);

// Unary ops (in-place)
void abs_inplace(dvec& a);
void exp_inplace(dvec& a);
void log_inplace(dvec& a);
void log10_inplace(dvec& a);
void sqrt_inplace(dvec& a);
void power_inplace(dvec& a, double p);
void clip_inplace(dvec& a, double lo, double hi);
void sinc_inplace(dvec& a);

// Reductions / stats
bool   any(const dvec& a);
double sum(const dvec& a);
double prod(const dvec& a);
double max(const dvec& a);
double min(const dvec& a);
double mean(const dvec& a);
double stddev(const dvec& a, bool ddof1 = false);
double median(const dvec& a);
double dot(const dvec& x, const dvec& y);

double trapz(const dvec& y, double dx);

// Array manipulation
void concatenate(const dvec& a, const dvec& b, dvec& out);
void roll(const dvec& a, int shift, dvec& out);
void pad_constant(const dvec& a, size_t pl, size_t pr, double cval, dvec& out);
void gradient(const dvec& a, double dx, dvec& out);
void union1d(const dvec& a, const dvec& b, dvec& out);

// Linear algebra-ish helpers
std::vector<double> corrcoef(const dvec& x, const dvec& y);
std::vector<double> cov(const dvec& x, const dvec& y, bool bias=false);

// FFT
void fft (const cvec& in, cvec& out);
void ifft(const cvec& in, cvec& out);

// IO helpers (host side)
void savetxt(const std::string& path, const hvec& data);
hvec genfromtxt(const std::string& path);

} // namespace gpnp
