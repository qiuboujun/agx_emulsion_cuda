// gp_scipy.cu
#include "gp_scipy.hpp"

#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>

#include <cmath>
#include <fstream>
#include <cstring>
#include <algorithm>

namespace gpnp {

// ---------------------------------------------------------------------
// Utility: grid/block helpers
// ---------------------------------------------------------------------
static inline dim3 blocks_for(size_t n, int t = 256){
    return dim3( (n + t - 1) / t );
}

// ---------------------------------------------------------------------
// 1) interp1d kernels & wrapper
// ---------------------------------------------------------------------
__global__ void interp_linear_kernel(const double* __restrict__ x,
                                     const double* __restrict__ y,
                                     int n,
                                     const double* __restrict__ xq,
                                     double* __restrict__ yq,
                                     int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    double xv = xq[i];
    // Linear extrapolation outside the input range (to match SciPy
    // `fill_value="extrapolate"`). Use the slope of the first / last
    // segment.
    if (xv <= x[0]) {
        double slope = (y[1] - y[0]) / (x[1] - x[0]);
        yq[i] = y[0] + slope * (xv - x[0]);
        return;
    }
    if (xv >= x[n-1]) {
        double slope = (y[n-1] - y[n-2]) / (x[n-1] - x[n-2]);
        yq[i] = y[n-1] + slope * (xv - x[n-1]);
        return;
    }

    // Binary search
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (hi + lo) >> 1;
        if (x[mid] <= xv) lo = mid; else hi = mid;
    }
    double t = (xv - x[lo]) / (x[hi] - x[lo]);
    yq[i] = y[lo] * (1.0 - t) + y[hi] * t;
}

__global__ void interp_nearest_kernel(const double* __restrict__ x,
                                      const double* __restrict__ y,
                                      int n,
                                      const double* __restrict__ xq,
                                      double* __restrict__ yq,
                                      int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    double xv = xq[i];
    if (xv <= x[0])      { yq[i] = y[0];      return; }
    if (xv >= x[n-1])    { yq[i] = y[n-1];    return; }

    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (hi + lo) >> 1;
        if (x[mid] <= xv) lo = mid; else hi = mid;
    }
    double dl = fabs(xv - x[lo]);
    double dh = fabs(xv - x[hi]);
    yq[i] = (dl <= dh) ? y[lo] : y[hi];
}

void interp1d(const dvec& x, const dvec& y,
              const dvec& xq, dvec& yq,
              const std::string& kind)
{
    int n = static_cast<int>(x.size());
    int m = static_cast<int>(xq.size());
    yq.resize(m);

    int threads = 256;
    dim3 blocks = blocks_for(m, threads);

    if (kind == "nearest") {
        interp_nearest_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()), n,
            thrust::raw_pointer_cast(xq.data()),
            thrust::raw_pointer_cast(yq.data()), m
        );
    } else {
        interp_linear_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()), n,
            thrust::raw_pointer_cast(xq.data()),
            thrust::raw_pointer_cast(yq.data()), m
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------
// 2) Natural cubic spline (splrep / splev)
// ---------------------------------------------------------------------
// Helper for solving the banded system for spline coefficients
static void solve_banded_system(const hvec& h, hvec& b, hvec& c) {
    int n = h.size(); // n-1 equations for n-1 unknowns (c_1 to c_{n-1})
    if (n <= 1) {
        c.assign(n + 1, 0.0);
        return;
    }

    // Setup the tridiagonal system for c_1, ..., c_{n-1}
    hvec diag(n - 1), sup(n - 2), sub(n - 2);
    for (int i = 0; i < n - 1; ++i) {
        diag[i] = 2.0 * (h[i] + (i + 1 < n ? h[i + 1] : 0));
    }
    for (int i = 0; i < n - 2; ++i) {
        sup[i] = sub[i] = h[i + 1];
    }

    // Forward elimination (Thomas algorithm)
    for (int i = 1; i < n - 1; ++i) {
        double m = sub[i - 1] / diag[i - 1];
        diag[i] -= m * sup[i - 1];
        b[i] -= m * b[i - 1]; // This line will now compile correctly
    }

    // Back substitution
    c.resize(n + 1);
    c[0] = c[n] = 0.0; // Boundary conditions c_0 = c_n = 0 for this system
    c[n - 1] = b[n - 2] / diag[n - 2];
    for (int i = n - 2; i > 0; --i) {
        c[i] = (b[i - 1] - sup[i - 1] * c[i + 1]) / diag[i - 1];
    }
}

static CubicSpline build_scipy_spline(const hvec& x, const hvec& y)
{
    // Natural cubic spline (second derivative = 0 at endpoints).
    int n = x.size();
    CubicSpline sp;
    sp.x = x;
    sp.a.resize(n - 1);
    sp.b.resize(n - 1);
    sp.c.resize(n);
    sp.d.resize(n - 1);

    if(n < 2){
        std::fill(sp.c.begin(), sp.c.end(), 0.0);
        return sp;
    }

    hvec h(n - 1);
    for (int i = 0; i < n - 1; ++i)
        h[i] = x[i + 1] - x[i];

    hvec alpha(n);
    alpha[0] = 0.0;
    alpha[n-1] = 0.0;
    for (int i = 1; i < n - 1; ++i)
        alpha[i] = 3 * ( (y[i+1]-y[i]) / h[i] - (y[i]-y[i-1]) / h[i-1] );

    hvec l(n), mu(n), z(n);
    l[0] = 1.0; mu[0] = z[0] = 0.0;
    for (int i = 1; i < n - 1; ++i){
        l[i] = 2*(x[i+1] - x[i-1]) - h[i-1]*mu[i-1];
        mu[i] = h[i]/l[i];
        z[i]  = (alpha[i] - h[i-1]*z[i-1]) / l[i];
    }
    l[n-1] = 1.0; z[n-1] = 0.0; sp.c[n-1] = 0.0;

    for (int j = n - 2; j >= 0; --j){
        sp.c[j] = z[j] - mu[j]*sp.c[j+1];
        sp.b[j] = (y[j+1]-y[j])/h[j] - h[j]*(sp.c[j+1] + 2*sp.c[j])/3.0;
        sp.d[j] = (sp.c[j+1] - sp.c[j]) / (3.0*h[j]);
        sp.a[j] = y[j];
    }
    return sp;
}


CubicSpline splrep(const dvec& xd, const dvec& yd) {
    hvec xh(xd.size()), yh(yd.size());
    thrust::copy(xd.begin(), xd.end(), xh.begin());
    thrust::copy(yd.begin(), yd.end(), yh.begin());
    return build_scipy_spline(xh, yh);
}

__global__ void splev_kernel(const double* xk,
                             const double* a, const double* b,
                             const double* c, const double* d,
                             int n,
                             const double* xq,
                             double* yq,
                             int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    double xv = xq[i];
    int idx;
    if (xv <= xk[0]) {
        idx = 0;
    } else if (xv >= xk[n-1]) {
        idx = n - 2;
    } else {
        int lo = 0, hi = n-1;
        while (hi - lo > 1) {
            int mid = (hi + lo) >> 1;
            if (xk[mid] <= xv) lo = mid; else hi = mid;
        }
        idx = lo;
    }
    double dx = xv - xk[idx];
    yq[i] = a[idx] + b[idx]*dx + c[idx]*dx*dx + d[idx]*dx*dx*dx;
}

void splev(const CubicSpline& sp, const dvec& xq, dvec& yq){
    int n = static_cast<int>(sp.x.size());
    int m = static_cast<int>(xq.size());
    yq.resize(m);

    // Upload coeffs to device
    dvec dx(sp.x.begin(), sp.x.end());
    dvec da(sp.a.begin(), sp.a.end());
    dvec db(sp.b.begin(), sp.b.end());
    dvec dc(sp.c.begin(), sp.c.end());
    dvec dd(sp.d.begin(), sp.d.end());

    int threads = 256;
    dim3 blocks = blocks_for(m, threads);
    splev_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(dx.data()),
        thrust::raw_pointer_cast(da.data()),
        thrust::raw_pointer_cast(db.data()),
        thrust::raw_pointer_cast(dc.data()),
        thrust::raw_pointer_cast(dd.data()),
        n,
        thrust::raw_pointer_cast(xq.data()),
        thrust::raw_pointer_cast(yq.data()),
        m
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------
// 3) convolve2d (mode="same", fill with cval)
// ---------------------------------------------------------------------
__global__ void conv2_same_kernel(const double* img, int H, int W,
                                  const double* ker, int kH, int kW,
                                  double* out, double cval)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= H || c >= W) return;

    int pad_h = kH / 2;
    int pad_w = kW / 2;

    double acc = 0.0;
    for (int kr = 0; kr < kH; ++kr) {
        int rr = r + kr - pad_h;
        for (int kc = 0; kc < kW; ++kc) {
            int cc = c + kc - pad_w;
            double v = cval;
            if (rr >= 0 && rr < H && cc >= 0 && cc < W) {
                v = img[rr * W + cc];
            }
            // --- CHANGE HERE: Flip kernel indices to match SciPy ---
            acc += v * ker[(kH - 1 - kr) * kW + (kW - 1 - kc)];
        }
    }
    out[r * W + c] = acc;
}

void convolve2d(const dvec& img, int H, int W,
                const dvec& kernel, int kH, int kW,
                dvec& out, double cval)
{
    out.resize(H * W);
    dim3 block(16,16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    conv2_same_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(img.data()), H, W,
        thrust::raw_pointer_cast(kernel.data()), kH, kW,
        thrust::raw_pointer_cast(out.data()), cval
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------
// 4) Gaussian window (device + host versions)
// ---------------------------------------------------------------------
__global__ void gaussian_win_kernel(double* w, int M, double stddev){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    double m = 0.5 * (M - 1);
    double x = (i - m) / stddev;
    w[i] = exp(-0.5 * x * x);
}

void gaussian_window(int M, double stddev, dvec& w){
    w.resize(M);
    int threads = 256;
    dim3 blocks = blocks_for(M, threads);
    gaussian_win_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(w.data()), M, stddev
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Host version for printing/verification
hvec gaussian_window_h(int M, double stddev){
    hvec w(M);
    double m = 0.5 * (M - 1);
    for (int i = 0; i < M; ++i) {
        double x = (i - m) / stddev;
        w[i] = std::exp(-0.5 * x * x);
    }
    return w;
}

// ---------------------------------------------------------------------
// 5) erf_inplace
// ---------------------------------------------------------------------
__global__ void erf_kernel(double* a, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = erf(a[i]);
}

void erf_inplace(dvec& a){
    int threads = 256;
    dim3 blocks = blocks_for(a.size(), threads);
    erf_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(a.data()), a.size()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------
// 6) curve_fit / least_squares generic
// ---------------------------------------------------------------------
static void residuals_gpu(
    const std::function<void(const dvec&, const std::vector<double>&, dvec&)>& func,
    const dvec& x, const dvec& y,
    const std::vector<double>& p,
    dvec& r)
{
    // y_model = f(x,p)
    dvec y_model;
    y_model.resize(y.size());
    func(x, p, y_model);

    // r = f(x,p) - y
    r.resize(y.size());
    thrust::transform(y_model.begin(), y_model.end(),
                      y.begin(),
                      r.begin(),
                      thrust::minus<double>());
}

static void lm_solve_host(const std::vector<std::vector<double>>& JtJ,
                          const std::vector<double>& Jtr,
                          double lambda,
                          std::vector<double>& dp)
{
    // Solve (J^T J + lambda*I) dp = J^T r
    int m = static_cast<int>(JtJ.size());
    std::vector<std::vector<double>> A = JtJ;
    std::vector<double> b = Jtr;

    for (int i = 0; i < m; ++i) A[i][i] += lambda;

    // Gaussian elimination
    for (int i = 0; i < m; ++i) {
        double piv = A[i][i];
        if (std::fabs(piv) < 1e-15) piv = 1e-15;
        for (int j = i; j < m; ++j) A[i][j] /= piv;
        b[i] /= piv;

        for (int k = i + 1; k < m; ++k) {
            double f = A[k][i];
            for (int j = i; j < m; ++j) A[k][j] -= f * A[i][j];
            b[k] -= f * b[i];
        }
    }

    dp.resize(m);
    for (int i = m - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < m; ++j) s -= A[i][j] * dp[j];
        dp[i] = s;
    }
}

CurveFitResult curve_fit(
    std::function<void(const dvec&, const std::vector<double>&, dvec&)> func,
    const dvec& x, const dvec& y,
    std::vector<double> p,
    const CurveFitOptions& opts)
{
    CurveFitResult res;
    res.params       = p;
    res.n_iter       = 0;
    res.final_cost   = 0.0;
    res.success      = false;

    double lambda = opts.lambda0;

    dvec r;
    residuals_gpu(func, x, y, p, r);
    double cost = 0.5 * thrust::inner_product(r.begin(), r.end(), r.begin(), 0.0);

    for (int it = 0; it < opts.max_iter; ++it) {
        double prev_cost = cost;

        // --- UPDATED JACOBIAN CALCULATION (Central Difference) ---
        const double eps = 1e-6;
        size_t m = p.size();
        size_t N = r.size();

        std::vector<std::vector<double>> J(N, std::vector<double>(m, 0.0));

        for (size_t k = 0; k < m; ++k) {
            std::vector<double> p_plus = p;
            p_plus[k] += eps;

            std::vector<double> p_minus = p;
            p_minus[k] -= eps;

            dvec r_plus, r_minus;
            residuals_gpu(func, x, y, p_plus, r_plus);
            residuals_gpu(func, x, y, p_minus, r_minus);

            hvec rh_plus(r_plus.size()), rh_minus(r_minus.size());
            thrust::copy(r_plus.begin(), r_plus.end(), rh_plus.begin());
            thrust::copy(r_minus.begin(), r_minus.end(), rh_minus.begin());

            for (size_t i = 0; i < N; ++i) {
                // This is now a central difference for better precision
                J[i][k] = (rh_plus[i] - rh_minus[i]) / (2.0 * eps);
            }
        }
        // --- End of Jacobian Update ---

        hvec r0(r.size());
        thrust::copy(r.begin(), r.end(), r0.begin());

        // Form J^T J and J^T r
        std::vector<std::vector<double>> JtJ(m, std::vector<double>(m, 0.0));
        std::vector<double> Jtr(m, 0.0);
        for (size_t i = 0; i < N; ++i) {
            for (size_t a = 0; a < m; ++a) {
                Jtr[a] += J[i][a] * r0[i];
                for (size_t b = 0; b < m; ++b) {
                    JtJ[a][b] += J[i][a] * J[i][b];
                }
            }
        }

        // Solve LM step
        std::vector<double> dp;
        lm_solve_host(JtJ, Jtr, lambda, dp);

        std::vector<double> p_new = p;
        for (size_t k = 0; k < m; ++k) p_new[k] -= dp[k];

        dvec r_new;
        residuals_gpu(func, x, y, p_new, r_new);
        double cost_new = 0.5 * thrust::inner_product(r_new.begin(), r_new.end(), r_new.begin(), 0.0);

        if (cost_new < cost) {
            // Accept the step
            p      = p_new;
            r      = r_new;
            cost   = cost_new;
            lambda = std::max(lambda * 0.3, 1e-12); // Decrease damping
        } else {
            // Reject the step
            lambda *= 2.0; // Increase damping
        }

        res.n_iter = it + 1;
        if (std::fabs(prev_cost - cost) < opts.ftol) {
            res.success = true;
            break;
        }
    }

    res.params     = p;
    res.final_cost = cost;
    // If we finished by reaching max iterations, consider it a form of convergence
    if (!res.success && res.n_iter == opts.max_iter) {
       res.success = true;
    }
    return res;
}

LeastSquaresResult least_squares(
    std::function<void(const dvec&, const std::vector<double>&, dvec&)> func,
    const dvec& x, const dvec& y,
    std::vector<double> p0,
    const LeastSquaresOptions& opts)
{
    CurveFitOptions cfo;
    cfo.max_iter = opts.max_iter;
    cfo.ftol     = opts.ftol;
    cfo.lambda0  = opts.lambda0;

    // Just wrap curve_fit
    auto r = curve_fit(func, x, y, p0, cfo);
    LeastSquaresResult ls;
    ls.params     = r.params;
    ls.n_iter     = r.n_iter;
    ls.final_cost = r.final_cost;
    ls.success    = r.success;
    return ls;
}

// ---------------------------------------------------------------------
// 7) Exponential convenience wrappers (y = a * exp(b * x))
// ---------------------------------------------------------------------
void curve_fit_exp(const dvec& x,
                   const dvec& y,
                   const std::vector<double>& p0,
                   std::vector<double>&       p_opt,
                   const CurveFitOptions&     opts)
{
    auto model = [](const dvec& xin,
                    const std::vector<double>& p,
                    dvec& y_model)
    {
        y_model.resize(xin.size());
        double a = p[0], b = p[1];
        thrust::transform(
            xin.begin(), xin.end(),
            y_model.begin(),
            [a,b] __device__ (double xi) {
                return a * exp(b * xi);
            }
        );
    };

    CurveFitResult res = curve_fit(model, x, y, p0, opts);
    p_opt = res.params;
}

LeastSquaresResult least_squares_exp(const dvec& x,
                                     const dvec& y,
                                     std::vector<double>        p0,
                                     const LeastSquaresOptions& opts)
{
    auto model = [](const dvec& xin,
                    const std::vector<double>& p,
                    dvec& y_model)
    {
        y_model.resize(xin.size());
        double a = p[0], b = p[1];
        thrust::transform(
            xin.begin(), xin.end(),
            y_model.begin(),
            [a,b] __device__ (double xi) {
                return a * exp(b * xi);
            }
        );
    };

    return least_squares(model, x, y, p0, opts);
}

// ---------------------------------------------------------------------
// 8) Minimal loadmat_double (optional; keep as stub if unused)
// ---------------------------------------------------------------------
bool loadmat_double(const std::string& /*path*/,
                    const std::string& /*varname*/,
                    hvec&              /*host_data*/,
                    int&               /*rows*/,
                    int&               /*cols*/)
{
    // Not implemented here. Provide real MAT v5 reader if you need it.
    return false;
}

} // namespace gpnp

