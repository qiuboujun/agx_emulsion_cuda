#pragma once
#include <vector>
#include <array>
#include <cmath>

namespace dc {
using dvec = std::vector<double>;

enum class CurveType { Negative, Positive, Paper };

// -------- Gaussian CDF composite (three layers) --------
// params: centers[3], amplitudes[3], sigmas[3]
void density_curve_norm_cdfs(const dvec& logE,
                             const std::array<double,9>& params,
                             CurveType type,
                             dvec& out);

// convenience for RGB parameters array[3][9]
void density_curves_rgb_norm(const dvec& logE,
                             const std::array<std::array<double,9>,3>& params,
                             CurveType type,
                             std::vector<std::array<double,3>>& out);

// -------- log-line analytic model --------
// params size 10 as per python comment
void density_curve_log_line(const dvec& logE,
                            const std::array<double,10>& params,
                            CurveType type,
                            dvec& out);
} 