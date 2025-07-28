#pragma once
#include <vector>
#include <array>
#include "interpolate.hpp"
#include "couplers.hpp"

namespace emu {
class FilmSimple {
public:
    // Provide density curves, sensitivities etc.
    FilmSimple(const interp::dvec& log_exposure,
               const std::vector<std::array<double,3>>& density_curves,
               double gamma_factor,
               bool use_couplers=false,
               const cp::Matrix3& dir_matrix = cp::Matrix3{{{0,0,0},{0,0,0},{0,0,0}}},
               double high_exposure_shift=0.0);

    void develop(const interp::dvec& log_raw,
                 std::vector<std::array<double,3>>& density_out) const;
private:
    interp::dvec m_logExp;
    std::vector<std::array<double,3>> m_densityCurves;
    std::array<double,3> m_gamma;
    bool m_useCouplers;
    cp::Matrix3 m_dirM;
    double m_shift;
};
} 