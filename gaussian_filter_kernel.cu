#include "kernels.h" // Include our new header
#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

// --- Eigen Configuration ---
#define EIGEN_DONT_USE_CUDA
#include <Eigen/Dense>

// --- CUDA Error Checking Macro ---
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(err_) \
                  << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
}

/* ------------------------- host helpers ------------------------- */
Eigen::VectorXd gaussian_kernel1d(double sigma, int radius) {
    int size = 2 * radius + 1;
    Eigen::VectorXd k(size);
    double sum = 0.0;
    double sigma2 = 2.0 * sigma * sigma;
    for (int i = 0; i < size; ++i) {
        double x = i - radius;
        k(i) = std::exp(-x * x / sigma2);
        sum += k(i);
    }
    return k / sum; // normalise
}

/* ------------------------- device kernels ------------------------- */
__device__ __forceinline__ int reflect_idx(int idx, int len) {
    if (idx < 0) return -idx - 1;
    if (idx >= len) return 2 * len - idx - 1;
    return idx;
}

__global__ void horiz_kernel(const double* in, double* out, int W, int H, const double* k, int R) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= W || r >= H) return;

    double sum = 0.0;
    for (int t = -R; t <= R; ++t) {
        int ci = reflect_idx(c + t, W);
        sum += in[r * W + ci] * k[t + R];
    }
    out[r * W + c] = sum;
}

__global__ void vert_kernel(const double* in, double* out, int W, int H, const double* k, int R) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= W || r >= H) return;

    double sum = 0.0;
    for (int t = -R; t <= R; ++t) {
        int ri = reflect_idx(r + t, H);
        sum += in[ri * W + c] * k[t + R];
    }
    out[r * W + c] = sum;
}

/* ------------------------- public API implementation ------------------------- */
Matrix gaussian_filter1d(const Matrix& input, double sigma, int axis, int order, const std::string& mode, double /*cval*/, double truncate, int radius) {
    if (order != 0) throw std::invalid_argument("Only order=0 supported");
    if (mode != "reflect") throw std::invalid_argument("Only mode=\"reflect\" supported");
    if (radius < 0) radius = static_cast<int>(truncate * sigma + 0.5);
    if (radius < 0) throw std::invalid_argument("Negative radius");

    Eigen::VectorXd h_k = gaussian_kernel1d(sigma, radius);

    int H = input.rows(), W = input.cols();
    size_t bytes_img = static_cast<size_t>(H) * W * sizeof(double);
    size_t bytes_k = h_k.size() * sizeof(double);

    double *d_in, *d_out, *d_k;
    CUDA_CHECK(cudaMalloc(&d_in, bytes_img));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_img));
    CUDA_CHECK(cudaMalloc(&d_k, bytes_k));

    CUDA_CHECK(cudaMemcpy(d_in, input.data(), bytes_img, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), bytes_k, cudaMemcpyHostToDevice));

    dim3 blk(16, 16);
    dim3 grd((W + blk.x - 1) / blk.x, (H + blk.y - 1) / blk.y);

    if (axis == 1 || axis == -1) { // horizontal
        horiz_kernel<<<grd, blk>>>(d_in, d_out, W, H, d_k, radius);
    } else if (axis == 0) { // vertical
        vert_kernel<<<grd, blk>>>(d_in, d_out, W, H, d_k, radius);
    } else {
        throw std::invalid_argument("Axis must be 0 or 1");
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Matrix result(H, W);
    CUDA_CHECK(cudaMemcpy(result.data(), d_out, bytes_img, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_k);

    return result;
}

Matrix gaussian_filter(const Matrix& input, std::pair<double, double> sigma, const std::string& mode, double truncate, std::pair<int, int> radius) {
    if (mode != "reflect") throw std::invalid_argument("Only mode=\"reflect\" supported");

    double sig_y = sigma.first;
    double sig_x = sigma.second;
    int rad_y = radius.first;
    int rad_x = radius.second;

    Matrix tmp = gaussian_filter1d(input, sig_y, 0, 0, mode, 0.0, truncate, rad_y);
    Matrix out = gaussian_filter1d(tmp, sig_x, 1, 0, mode, 0.0, truncate, rad_x);

    return out;
}

Matrix gaussian_filter(const Matrix& input, double sigma, const std::string& mode, double truncate, int radius) {
    return gaussian_filter(input, {sigma, sigma}, mode, truncate, {radius, radius});
}

