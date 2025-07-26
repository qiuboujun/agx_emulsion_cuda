#pragma once
#include "gp_numpy.hpp"   // gpnp::dvec
#include "gp_scipy.hpp"   // gpnp::splrep / gpnp::splev
#include <array>
#include <cmath>

/**
 * Compute γ (slope of the characteristic curve) for each colour channel
 * between two density levels.
 *
 * @param log_exposure   gpnp::dvec of length N (ascending)
 * @param density_curves array of three gpnp::dvec, length N each
 * @param density_0      lower density threshold (default 0.25)
 * @param density_1      upper density threshold (default 1.0)
 */
std::array<double,3> measure_gamma(
        const gpnp::dvec&                 log_exposure,
        const std::array<gpnp::dvec,3>&   density_curves,
        double density_0 = 0.25,
        double density_1 = 1.0);

/**
 * Compute slope around a reference exposure ± range/2.
 *
 * @param log_exposure         gpnp::dvec (N)
 * @param density_curves       array of three gpnp::dvec
 * @param log_exposure_ref     reference log‑E (default 0)
 * @param log_exposure_range   total range (default log10(4) = ±2 stops)
 */
std::array<double,3> measure_slopes_at_exposure(
        const gpnp::dvec&                 log_exposure,
        const std::array<gpnp::dvec,3>&   density_curves,
        double log_exposure_ref   = 0.0,
        double log_exposure_range = std::log10(4.0));

