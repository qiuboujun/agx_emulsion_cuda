#pragma once
#include "gp_numpy.hpp"
#include <string>

/**
 * Compute automatic exposure compensation in EV.
 *
 * @param Y       luminance channel, flattened row‑major (length = H * W)
 * @param H, W    image height and width
 * @param method  "median" or "center_weighted"
 * @return        exposure compensation in EV
 */
double measure_autoexposure_ev(const gpnp::dvec& Y,
                               int H, int W,
                               const std::string& method = "center_weighted");

