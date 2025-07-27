#include "gp_numpy.hpp"
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/inner_product.h>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace gpnp {

__global__ void clip_kernel(double* a, size_t n, double lo, double hi){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        double v=a[i];
        if(v<lo) v=lo; else if(v>hi) v=hi;
        a[i]=v;
    }
}
__global__ void power_kernel(double* a, size_t n, double p){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n) a[i] = pow(a[i], p);
}
__global__ void roll_kernel(const double* in, double* out, size_t n, int shift){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        size_t j = (i + shift + n) % n;
        out[j] = in[i];
    }
}
__global__ void pad_kernel(const double* in, double* out, size_t n, size_t pl, size_t pr, double c){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nt = n + pl + pr;
    if(i<nt){
        if(i<pl || i>=pl+n) out[i] = c;
        else out[i] = in[i-pl];
    }
}
__global__ void gradient_kernel(const double* a, double* g, size_t n, double dx){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        if(i==0)        g[i] = (a[1]-a[0])/dx;
        else if(i==n-1) g[i] = (a[n-1]-a[n-2])/dx;
        else            g[i] = (a[i+1]-a[i-1])/(2.0*dx);
    }
}
__global__ void sinc_kernel(double* a, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        double x = a[i];
        if(fabs(x) < 1e-12) { a[i] = 1.0; return; }
        double pix = M_PI * x;
        a[i] = sin(pix) / pix;
    }
}

static inline dim3 blocks_for(size_t n, int t=256){ return dim3((n + t - 1)/t); }

// ---------------- Creation ----------------
hvec arange_h_np(double start, double stop, double step){
    hvec v;
    // Handle positive or negative step
    if (step > 0) {
        for (double x = start; x < stop; x += step) v.push_back(x);
    } else if (step < 0) {
        for (double x = start; x > stop; x += step) v.push_back(x);
    } else {
        // step == 0: mimic NumPy error
        throw std::runtime_error("arange: step must be non-zero");
    }
    return v;
}

dvec arange(double start, double stop, double step){
    hvec h = arange_h_np(start, stop, step);
    return dvec(h.begin(), h.end());
}

hvec linspace_h(double start, double stop, int num){
    hvec v(num);
    double step = (stop-start)/(num-1);
    for(int i=0;i<num;++i) v[i] = start + i*step;
    return v;
}

dvec linspace(double start, double stop, int num){
    hvec h = linspace_h(start, stop, num);
    return dvec(h.begin(), h.end());
}

dvec zeros(size_t n){ return dvec(n, 0.0); }

dvec ones(size_t n, double v){ return dvec(n, v); }

dvec empty(size_t n){ dvec a; a.resize(n); return a; }

// --------------- Unary ops ---------------
void abs_inplace(dvec& a){ thrust::transform(a.begin(), a.end(), a.begin(), [] __device__ (double x){return fabs(x);}); }
void exp_inplace(dvec& a){ thrust::transform(a.begin(), a.end(), a.begin(), [] __device__ (double x){return exp(x);}); }
void log_inplace(dvec& a){ thrust::transform(a.begin(), a.end(), a.begin(), [] __device__ (double x){return log(x);}); }
void log10_inplace(dvec& a){ thrust::transform(a.begin(), a.end(), a.begin(), [] __device__ (double x){return log10(x);}); }
void sqrt_inplace(dvec& a){ thrust::transform(a.begin(), a.end(), a.begin(), [] __device__ (double x){return sqrt(x);}); }

void power_inplace(dvec& a, double p){
    int threads=256; auto blocks=blocks_for(a.size(),threads);
    power_kernel<<<blocks,threads>>>(thrust::raw_pointer_cast(a.data()), a.size(), p);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void clip_inplace(dvec& a, double lo, double hi){
    int threads=256; auto blocks=blocks_for(a.size(),threads);
    clip_kernel<<<blocks,threads>>>(thrust::raw_pointer_cast(a.data()), a.size(), lo, hi);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// --------------- Reductions ---------------
bool any(const dvec& a){
    auto it = thrust::find_if(a.begin(), a.end(), [] __device__ (double v){ return v != 0.0; });
    return it != a.end();
}

double sum(const dvec& a){ return thrust::reduce(a.begin(), a.end(), 0.0); }

double prod(const dvec& a){ return thrust::reduce(a.begin(), a.end(), 1.0, thrust::multiplies<double>()); }

double max(const dvec& a){ return *thrust::max_element(a.begin(), a.end()); }

double min(const dvec& a){ return *thrust::min_element(a.begin(), a.end()); }

double mean(const dvec& a){ return sum(a)/a.size(); }

double stddev(const dvec& a, int ddof = 0){
    double m = mean(a);
    dvec tmp(a.size());
    thrust::transform(a.begin(), a.end(), tmp.begin(),
                      [m] __device__ (double x){double d=x-m; return d*d;});
    double var = sum(tmp)/(a.size() - ddof);
    return sqrt(var);
}

double median(const dvec& a){
    hvec h(a.begin(), a.end());
    std::sort(h.begin(), h.end());
    size_t n=h.size();
    if(n%2) return h[n/2];
    return 0.5*(h[n/2-1]+h[n/2]);
}

double dot(const dvec& x, const dvec& y){
    return thrust::inner_product(x.begin(), x.end(), y.begin(), 0.0);
}

// --------------- Manipulation ---------------
void concatenate(const dvec& a, const dvec& b, dvec& out){
    out.resize(a.size()+b.size());
    thrust::copy(a.begin(), a.end(), out.begin());
    thrust::copy(b.begin(), b.end(), out.begin()+a.size());
}

void roll(const dvec& a, int shift, dvec& out){
    out.resize(a.size());
    int threads=256; auto blocks=blocks_for(a.size(),threads);
    roll_kernel<<<blocks,threads>>>(thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(out.data()), a.size(), shift);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void pad_constant(const dvec& a, size_t pl, size_t pr, double cval, dvec& out){
    out.resize(a.size()+pl+pr);
    int threads=256; auto blocks=blocks_for(out.size(),threads);
    pad_kernel<<<blocks,threads>>>(thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(out.data()), a.size(), pl, pr, cval);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gradient(const dvec& a, double dx, dvec& out){
    out.resize(a.size());
    int threads=256; auto blocks=blocks_for(a.size(),threads);
    gradient_kernel<<<blocks,threads>>>(thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(out.data()), a.size(), dx);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sinc_inplace(dvec& a){
    int threads=256; auto blocks=blocks_for(a.size(),threads);
    sinc_kernel<<<blocks,threads>>>(thrust::raw_pointer_cast(a.data()), a.size());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------- Standard normal PDF / CDF ----------------
namespace {
const double INV_SQRT_2PI = 0.39894228040143267794;   // 1/sqrt(2π)
const double INV_SQRT2    = 0.70710678118654752440;    // 1/√2
}

void norm_pdf(const dvec& x, dvec& out){
    out.resize(x.size());
    thrust::transform(x.begin(), x.end(), out.begin(), [] __device__ (double v){
        return INV_SQRT_2PI * exp(-0.5 * v * v);
    });
}

void norm_cdf(const dvec& x, dvec& out){
    out.resize(x.size());
    thrust::transform(x.begin(), x.end(), out.begin(), [] __device__ (double v){
        return 0.5 * (1.0 + erf(v * INV_SQRT2));
    });
}

// --------------- Integrals ---------------
double trapz(const dvec& y, double dx){
    double total = sum(y);
    if(y.size()>=2){
        double first, last;
        CUDA_CHECK(cudaMemcpy(&first, thrust::raw_pointer_cast(y.data()), sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last,  thrust::raw_pointer_cast(y.data()) + (y.size()-1), sizeof(double), cudaMemcpyDeviceToHost));
        total -= 0.5*(first+last);
    }
    return total * dx;
}

// --------------- Set ops ---------------
void union1d(const dvec& a, const dvec& b, dvec& out){
    dvec tmp(a.begin(), a.end());
    tmp.insert(tmp.end(), b.begin(), b.end());
    thrust::sort(tmp.begin(), tmp.end());
    auto end_it = thrust::unique(tmp.begin(), tmp.end());
    tmp.erase(end_it, tmp.end());
    out = tmp;
}

// --------------- Corr/Cov ---------------
std::vector<double> corrcoef(const dvec& x, const dvec& y){
    // assume x.size()==y.size()
    size_t n = x.size();
    double mx = mean(x);
    double my = mean(y);

    dvec dx(n), dy(n);
    thrust::transform(x.begin(), x.end(), dx.begin(),
                      [mx] __device__ (double v){return v-mx;});
    thrust::transform(y.begin(), y.end(), dy.begin(),
                      [my] __device__ (double v){return v-my;});

    double cxy = dot(dx, dy) / (n - 1);       // cov(x,y), ddof=1
    double sdx = stddev(x, 1);                // std dev with ddof=1
    double sdy = stddev(y, 1);
    double r   = cxy / (sdx * sdy);

    return {1.0, r, r, 1.0};
}

std::vector<double> cov(const dvec& x, const dvec& y, bool bias /*=false*/) {
    size_t n = x.size();
    double mx = mean(x), my = mean(y);
    dvec dx(n), dy(n);
    thrust::transform(x.begin(), x.end(), dx.begin(),
                      [mx] __device__ (double v){ return v - mx; });
    thrust::transform(y.begin(), y.end(), dy.begin(),
                      [my] __device__ (double v){ return v - my; });

    double denom = bias ? n : (n - 1);
    double cxx = dot(dx, dx) / denom;
    double cyy = dot(dy, dy) / denom;
    double cxy = dot(dx, dy) / denom;
    return {cxx, cxy, cxy, cyy};
}

// --------------- FFT / IFFT ---------------
void fft(const cvec& in, cvec& out){
    size_t n = in.size();
    out.resize(n);
    cufftHandle plan; CUFFT_CHECK(cufftPlan1d(&plan, (int)n, CUFFT_Z2Z, 1));
    const cufftDoubleComplex* in_const = reinterpret_cast<const cufftDoubleComplex*>(thrust::raw_pointer_cast(in.data()));
    cufftDoubleComplex* d_in  = const_cast<cufftDoubleComplex*>(in_const);
    cufftDoubleComplex* d_out = reinterpret_cast<cufftDoubleComplex*>(thrust::raw_pointer_cast(out.data()));
    CUFFT_CHECK(cufftExecZ2Z(plan, d_in, d_out, CUFFT_FORWARD));
    CUFFT_CHECK(cufftDestroy(plan));
}

void ifft(const cvec& in, cvec& out){
    size_t n = in.size();
    out.resize(n);
    cufftHandle plan; CUFFT_CHECK(cufftPlan1d(&plan, (int)n, CUFFT_Z2Z, 1));
    const cufftDoubleComplex* in_const = reinterpret_cast<const cufftDoubleComplex*>(thrust::raw_pointer_cast(in.data()));
    cufftDoubleComplex* d_in  = const_cast<cufftDoubleComplex*>(in_const);
    cufftDoubleComplex* d_out = reinterpret_cast<cufftDoubleComplex*>(thrust::raw_pointer_cast(out.data()));
    CUFFT_CHECK(cufftExecZ2Z(plan, d_in, d_out, CUFFT_INVERSE));
    CUFFT_CHECK(cufftDestroy(plan));
    thrust::transform(out.begin(), out.end(), out.begin(), [n] __device__ (thrust::complex<double> v){ return v / (double)n; });
}

// --------------- IO helpers ---------------
void savetxt(const std::string& path, const hvec& data){
    std::ofstream f(path);
    for(size_t i=0;i<data.size();++i) f<<data[i]<<"\n";
}

hvec genfromtxt(const std::string& path){
    std::ifstream f(path);
    std::vector<double> vals; double v;
    while(f>>v) vals.push_back(v);
    hvec h(vals.size());
    std::copy(vals.begin(), vals.end(), h.begin());
    return h;
}

} // namespace gpnp

