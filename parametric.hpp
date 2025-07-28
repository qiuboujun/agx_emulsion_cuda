#pragma once
#include <vector>
#include <array>

namespace prm {
using dvec=std::vector<double>;

void parametric_density_curves(const dvec& logE,
                               const std::array<double,3>& gamma,
                               const std::array<double,3>& logE0,
                               const std::array<double,3>& densityMax,
                               const std::array<double,3>& toeSize,
                               const std::array<double,3>& shoulderSize,
                               std::vector<std::array<double,3>>& out);
} 