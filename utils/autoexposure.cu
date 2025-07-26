#include "autoexposure.hpp"
#include <cstdio>
#include <cmath>

// -----------------------------------------------------------------------------
// CUDA kernel : build a Gaussian centre‑weighted mask
// -----------------------------------------------------------------------------
__global__ void mask_kernel(double* mask,
                            int H, int W,
                            double norm_h, double norm_w,
                            double sigma2_inv)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t N   = size_t(H) * W;
    if (idx >= N) return;

    int r = int(idx / W);          // row
    int c = int(idx % W);          // column

    double x = (double(c) / W) - 0.5;
    double y = (double(r) / H) - 0.5;
    x *= norm_w;
    y *= norm_h;

    mask[idx] = exp(-(x * x + y * y) * sigma2_inv);
}

// -----------------------------------------------------------------------------
// helper : centre‑weighted <Y> computation
// -----------------------------------------------------------------------------
static double center_weighted_exposure(const gpnp::dvec& Y,
                                       int H, int W,
                                       double sigma = 0.2)
{
    size_t N = Y.size();
    gpnp::dvec mask(N);

    double norm_h = double(H) / std::max(H, W);
    double norm_w = double(W) / std::max(H, W);
    double sigma2_inv = 1.0 / (2.0 * sigma * sigma);

    // ----- local grid‑size computation -----
    int  threads = 256;
    dim3 blocks((N + threads - 1) / threads);

    mask_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(mask.data()),
                                     H, W, norm_h, norm_w, sigma2_inv);
    CUDA_CHECK(cudaDeviceSynchronize());

    // normalise mask
    double denom = gpnp::sum(mask);
    thrust::transform(mask.begin(), mask.end(), mask.begin(),
                      [denom] __device__ (double w) { return w / denom; });

    // Y * mask
    gpnp::dvec tmp(N);
    thrust::transform(Y.begin(), Y.end(), mask.begin(), tmp.begin(),
                      [] __device__ (double y, double w) { return y * w; });

    return gpnp::sum(tmp);
}

// -----------------------------------------------------------------------------
// public entry point
// -----------------------------------------------------------------------------
double measure_autoexposure_ev(const gpnp::dvec& Y,
                               int H, int W,
                               const std::string& method)
{
    double Y_exposure = 0.0;

    if (method == "median") {
        Y_exposure = gpnp::median(Y);
    } else {                      // default : centre‑weighted
        Y_exposure = center_weighted_exposure(Y, H, W);
    }

    const double gray_card = 0.184;
    double exposure = Y_exposure / gray_card;
    double ev = -std::log2(exposure);

    if (!std::isfinite(ev)) {
        ev = 0.0;
        std::fprintf(stderr,
                     "Warning: auto‑exposure compensation is Inf, "
                     "defaulting to 0 EV.\n");
    }
    return ev;
}

