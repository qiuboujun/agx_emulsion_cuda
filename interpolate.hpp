#pragma once
#include <vector>
#include <array>

namespace interp {
using dvec=std::vector<double>;

// density_curves is Nx3
void exposure_to_density(const dvec& log_raw,
                         const std::vector<std::array<double,3>>& density_curves,
                         const dvec& log_exposure,
                         const std::array<double,3>& gamma_factor,
                         std::vector<std::array<double,3>>& out);
} 