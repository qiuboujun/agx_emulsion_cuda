#pragma once

#include "gp_numpy.hpp"
#include <vector>
#include <string>
#include <functional>

namespace gpnp {

// -------------------------------------------------------------------------
// 1) interp1d  (like scipy.interpolate.interp1d)
//    x, y, xq are device vectors; yq is output device vector.
//    kind: "linear" or "nearest"; extrapolate by clamping.
// -------------------------------------------------------------------------
void interp1d(const dvec&       x,
              const dvec&       y,
              const dvec&       xq,
              dvec&             yq,
              const std::string& kind = "linear");

// -------------------------------------------------------------------------
// 2) Natural cubic spline (splrep / splev)
//    build coeffs on host, evaluate on device.
// -------------------------------------------------------------------------
struct CubicSpline {
    hvec x;      // knot positions, length n
    hvec a, b;   // coeffs a,b length n-1
    hvec c;      // coeffs c length n
    hvec d;      // coeffs d length n-1
};

// build on host
CubicSpline splrep(const dvec& x, const dvec& y);

// evaluate on device
void splev(const CubicSpline& sp,
           const dvec&        xq,
           dvec&              yq);

// -------------------------------------------------------------------------
// 3) 2D convolution (mode="same", boundary="fill")
// -------------------------------------------------------------------------
void convolve2d(const dvec& img,    int H, int W,
                const dvec& kernel, int kH, int kW,
                dvec&       out,
                double      cval = 0.0);

// -------------------------------------------------------------------------
// 4) Gaussian window
//    — host version (for quick inspection)
//    — device version (to generate on GPU)
// -------------------------------------------------------------------------
hvec gaussian_window_h(int M, double stddev);
void gaussian_window(int M, double stddev, dvec& w);

// -------------------------------------------------------------------------
// 5) special.erf  (in-place on device)
// -------------------------------------------------------------------------
void erf_inplace(dvec& a);

// -------------------------------------------------------------------------
// 6) Generic LM / Gauss–Newton solvers
// -------------------------------------------------------------------------
struct CurveFitOptions {
    int    max_iter = 100;
    double ftol     = 1e-8;
    double lambda0  = 1e-3;
};
struct CurveFitResult {
    std::vector<double> params;
    int    n_iter;
    double final_cost;
    bool   success;
};

struct LeastSquaresOptions {
    int    max_iter = 100;
    double ftol     = 1e-8;
    double lambda0  = 0.0;
};
struct LeastSquaresResult {
    std::vector<double> params;
    int    n_iter;
    double final_cost;
    bool   success;
};

CurveFitResult   curve_fit   ( std::function<void(const dvec&, const std::vector<double>&, dvec&)> func,
                               const dvec& x, const dvec& y,
                               std::vector<double> p0,
                               const CurveFitOptions&   opts = CurveFitOptions() );

LeastSquaresResult least_squares( std::function<void(const dvec&, const std::vector<double>&, dvec&)> func,
                                  const dvec& x, const dvec& y,
                                  std::vector<double> p0,
                                  const LeastSquaresOptions& opts = LeastSquaresOptions() );

// -------------------------------------------------------------------------
// 7) Exponential‐model convenience wrappers
//    y = p[0] * exp(p[1] * x)
// -------------------------------------------------------------------------
void curve_fit_exp(const dvec&               x,
                   const dvec&               y,
                   const std::vector<double>& p0,
                   std::vector<double>&       p_opt,
                   const CurveFitOptions&     opts = CurveFitOptions());

LeastSquaresResult least_squares_exp(const dvec&               x,
                                     const dvec&               y,
                                     std::vector<double>       p0,
                                     const LeastSquaresOptions& opts = LeastSquaresOptions());

} // namespace gpnp

