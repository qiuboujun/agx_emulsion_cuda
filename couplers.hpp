#pragma once
#include <vector>
#include <array>
#include <cmath>

namespace cp {
using dvec=std::vector<double>;
using Matrix3 = std::array<std::array<double,3>,3>;

// Compute DIR couplers matrix (inhibitors) same semantics as python
Matrix3 compute_dir_couplers_matrix(const std::array<double,3>& amount_rgb, double layer_diffusion);

// Given density curves after couplers and exposure grid, compute curves before couplers
void compute_density_curves_before_dir_couplers(const std::vector<std::array<double,3>>& density_curves,
                                               const dvec& log_exposure,
                                               const Matrix3& dir_couplers_matrix,
                                               double high_exposure_couplers_shift,
                                               std::vector<std::array<double,3>>& out);

// Apply coupler exposure correction (1D no spatial blur)
void compute_exposure_correction_dir_couplers(const std::vector<std::array<double,3>>& log_raw,
                                              const std::vector<std::array<double,3>>& density_cmy,
                                              const std::array<double,3>& density_max,
                                              const Matrix3& dir_couplers_matrix,
                                              double high_exposure_shift,
                                              std::vector<std::array<double,3>>& log_raw_corrected);
} 