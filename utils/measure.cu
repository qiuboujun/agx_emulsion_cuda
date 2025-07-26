#include "measure.hpp"
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <cmath>

/* ------------------------------------------------------------------ */
/* helper: copy vector and replace NaNs with 0 on‑GPU                  */
/* ------------------------------------------------------------------ */
static void clean_nan(const gpnp::dvec& in, gpnp::dvec& out)
{
    out.resize(in.size());
    thrust::transform(
        in.begin(), in.end(), out.begin(),
        [] __device__ (double v) { return ::isnan(v) ? 0.0 : v; });
}

/* ================================================================== */
/* 1.  γ between two density levels                                   */
/* ================================================================== */
std::array<double,3> measure_gamma(const gpnp::dvec&               logE,
                                   const std::array<gpnp::dvec,3>& dens,
                                   double d0, double d1)
{
    std::array<double,3> gamma{};

    /* query vector on host & device */
    gpnp::hvec query_h(2);
    query_h[0] = d0;
    query_h[1] = d1;
    gpnp::dvec query_d(query_h.begin(), query_h.end());
    gpnp::dvec eval_d;

    for (int ch = 0; ch < 3; ++ch)
    {
        gpnp::dvec dens_clean;
        clean_nan(dens[ch], dens_clean);

        gpnp::CubicSpline sp = gpnp::splrep(dens_clean, logE);  // inverse spline
        gpnp::splev(sp, query_d, eval_d);

        gpnp::hvec eval_h(2);
        thrust::copy(eval_d.begin(), eval_d.end(), eval_h.begin());

        double loge0 = eval_h[0], loge1 = eval_h[1];
        gamma[ch] = (d1 - d0) / (loge1 - loge0);
    }
    return gamma;
}

/* ================================================================== */
/* 2.  slope at reference exposure                                    */
/* ================================================================== */
std::array<double,3> measure_slopes_at_exposure(
        const gpnp::dvec&               logE,
        const std::array<gpnp::dvec,3>& dens,
        double le_ref,
        double le_range)
{
    std::array<double,3> slope{};

    double x0 = le_ref - 0.5 * le_range;
    double x1 = le_ref + 0.5 * le_range;

    gpnp::hvec query_h(2);          // host
    query_h[0] = x0;
    query_h[1] = x1;
    gpnp::dvec query_d(query_h.begin(), query_h.end());
    gpnp::dvec eval_d;

    for (int ch = 0; ch < 3; ++ch)
    {
        gpnp::dvec dens_clean;
        clean_nan(dens[ch], dens_clean);

        gpnp::CubicSpline sp = gpnp::splrep(logE, dens_clean);  // forward spline
        gpnp::splev(sp, query_d, eval_d);

        gpnp::hvec eval_h(2);
        thrust::copy(eval_d.begin(), eval_d.end(), eval_h.begin());

        double d0 = eval_h[0], d1 = eval_h[1];
        slope[ch] = (d1 - d0) / (x1 - x0);
    }
    return slope;
}

