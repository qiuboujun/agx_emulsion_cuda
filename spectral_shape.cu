#include "spectral_shape.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#define CUDA_CHECK(err)                                                     \
    do {                                                                    \
        cudaError_t _e = (err);                                             \
        if(_e != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while(0)

static inline bool is_numeric(double v) { return std::isfinite(v); }

__global__ void fill_range_kernel(double* out, double start, double step, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) out[i] = start + i * step;
}

// Compare by rounding to a factor = 10^digits
__global__ void contains_scalar_kernel(const double* data, int n,
                                       double value, double factor,
                                       int* flag)
{
    long long target = llround(value * factor);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n;
        i += blockDim.x * gridDim.x)
    {
        long long val = llround(data[i] * factor);
        if(val == target) {
            atomicExch(flag, 1);
            return;
        }
    }
}

__global__ void contains_vector_kernel(const double* data, int n,
                                       const double* query, int m,
                                       double factor,
                                       int* per_query_ok)
{
    int q = blockIdx.x; // one block per query value
    if(q >= m) return;

    long long target = llround(query[q] * factor);

    // each thread scans part of data
    int found = 0;
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
        long long val = llround(data[i] * factor);
        if(val == target) { found = 1; break; }
    }

    __shared__ int block_found;
    if(threadIdx.x == 0) block_found = 0;
    __syncthreads();

    if(found) atomicExch(&block_found, 1);
    __syncthreads();

    if(threadIdx.x == 0) {
        per_query_ok[q] = block_found ? 1 : 0;
    }
}

/* ------------------- SpectralShape methods ------------------- */

SpectralShape::SpectralShape(Real start, Real end, Real interval)
{
    this->start(start);
    this->end(end);
    this->interval(interval);
    allocate_and_fill_on_device();
}

SpectralShape::~SpectralShape()
{
    free_device();
}

SpectralShape::SpectralShape(const SpectralShape& other)
{
    _start    = other._start;
    _end      = other._end;
    _interval = other._interval;
    _count    = other._count;
    allocate_and_fill_on_device();
    _h_cache.reset();
}

SpectralShape& SpectralShape::operator=(const SpectralShape& other)
{
    if(this == &other) return *this;
    free_device();
    _start    = other._start;
    _end      = other._end;
    _interval = other._interval;
    _count    = other._count;
    allocate_and_fill_on_device();
    _h_cache.reset();
    return *this;
}

SpectralShape::SpectralShape(SpectralShape&& other) noexcept
{
    _start    = other._start;
    _end      = other._end;
    _interval = other._interval;
    _count    = other._count;
    _d_wavelengths      = other._d_wavelengths;
    other._d_wavelengths = nullptr;
    _h_cache.reset();
}

SpectralShape& SpectralShape::operator=(SpectralShape&& other) noexcept
{
    if(this == &other) return *this;
    free_device();
    _start    = other._start;
    _end      = other._end;
    _interval = other._interval;
    _count    = other._count;
    _d_wavelengths      = other._d_wavelengths;
    other._d_wavelengths = nullptr;
    _h_cache.reset();
    return *this;
}

void SpectralShape::start(Real v)
{
    if(!is_numeric(v))
        throw std::invalid_argument("\"start\" is not a number");
    if(v >= _end && _end != 0) // ignore this check until end is set
        throw std::invalid_argument("\"start\" must be strictly less than end");
    _start = v;
    _h_cache.reset();
}

void SpectralShape::end(Real v)
{
    if(!is_numeric(v))
        throw std::invalid_argument("\"end\" is not a number");
    if(v <= _start)
        throw std::invalid_argument("\"end\" must be strictly greater than start");
    _end = v;
    _h_cache.reset();
}

void SpectralShape::interval(Real v)
{
    if(!is_numeric(v))
        throw std::invalid_argument("\"interval\" is not a number");
    if(v <= 0)
        throw std::invalid_argument("\"interval\" must be > 0");
    _interval = v;
    _h_cache.reset();
}

void SpectralShape::boundaries(const std::pair<Real, Real>& b)
{
    start(b.first);
    end(b.second);
    _h_cache.reset();
}

void SpectralShape::allocate_and_fill_on_device()
{
    if(_d_wavelengths) {
        cudaFree(_d_wavelengths);
        _d_wavelengths = nullptr;
    }
    _count = static_cast<int>(std::round((_end - _start) / _interval)) + 1;
    if(_count <= 0)
        throw std::runtime_error("Invalid range (count <= 0)");
    CUDA_CHECK(cudaMalloc(&_d_wavelengths, _count * sizeof(double)));

    int block = 256;
    int grid  = (_count + block - 1) / block;
    fill_range_kernel<<<grid, block>>>(_d_wavelengths, _start, _interval, _count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void SpectralShape::free_device()
{
    if(_d_wavelengths) {
        cudaFree(_d_wavelengths);
        _d_wavelengths = nullptr;
    }
}

void SpectralShape::regenerate_if_needed() const
{
    if(_h_cache &&
       _cached_start    == _start &&
       _cached_end      == _end &&
       _cached_interval == _interval) {
        return;
    }

    // Refill device buffer if needed
    const_cast<SpectralShape*>(this)->allocate_and_fill_on_device();

    // Copy to host
    Vector host(_count);
    CUDA_CHECK(cudaMemcpy(host.data(), _d_wavelengths,
                          _count * sizeof(double),
                          cudaMemcpyDeviceToHost));
    _h_cache = std::move(host);

    _cached_start    = _start;
    _cached_end      = _end;
    _cached_interval = _interval;
}

const SpectralShape::Vector& SpectralShape::range() const
{
    regenerate_if_needed();
    return *_h_cache;
}

bool SpectralShape::cuda_contains_scalar(const double* d_data,
                                         int n,
                                         double value,
                                         double eps_digits10)
{
    int* d_flag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));

    double factor = std::pow(10.0, eps_digits10);
    int block = 256;
    int grid  = (n + block - 1) / block;
    contains_scalar_kernel<<<grid, block>>>(d_data, n, value, factor, d_flag);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_flag = 0;
    CUDA_CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_flag);
    return h_flag != 0;
}

bool SpectralShape::cuda_contains_vector(const double* d_data,
                                         int n,
                                         const double* d_query,
                                         int m,
                                         double eps_digits10)
{
    int* d_ok = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ok, m * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ok, 1, m * sizeof(int)));

    double factor = std::pow(10.0, eps_digits10);
    int threads = 256;
    contains_vector_kernel<<<m, threads>>>(d_data, n, d_query, m, factor, d_ok);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> host_ok(m);
    CUDA_CHECK(cudaMemcpy(host_ok.data(), d_ok, m * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_ok);

    for(int v : host_ok) {
        if(v == 0) return false;
    }
    return true;
}

bool SpectralShape::contains(Real wavelength) const
{
    double digits = std::numeric_limits<double>::digits10;
    return cuda_contains_scalar(_d_wavelengths, _count, wavelength, digits);
}

bool SpectralShape::contains(const Vector& wavelengths) const
{
    double digits = std::numeric_limits<double>::digits10;

    // Copy query to device
    double* d_query = nullptr;
    CUDA_CHECK(cudaMalloc(&d_query, wavelengths.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_query, wavelengths.data(),
                          wavelengths.size() * sizeof(double),
                          cudaMemcpyHostToDevice));

    bool ok = cuda_contains_vector(_d_wavelengths, _count,
                                   d_query, wavelengths.size(),
                                   digits);
    cudaFree(d_query);
    return ok;
}

std::size_t SpectralShape::size() const
{
    return static_cast<std::size_t>(_count);
}

bool SpectralShape::operator==(const SpectralShape& other) const
{
    // shape parameters define equality
    return _start == other._start &&
           _end   == other._end &&
           _interval == other._interval;
}

std::string SpectralShape::str() const
{
    std::ostringstream os;
    os << "(" << _start << ", " << _end << ", " << _interval << ")";
    return os.str();
}

std::string SpectralShape::repr() const
{
    std::ostringstream os;
    os << "SpectralShape(" << _start << ", " << _end << ", " << _interval << ")";
    return os.str();
}

std::size_t SpectralShape::hash() const
{
    auto h1 = std::hash<double>{}(_start);
    auto h2 = std::hash<double>{}(_end);
    auto h3 = std::hash<double>{}(_interval);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
}

