#pragma once
#include <vector>
#include <array>
#include <string>

namespace cf {
using dvec = std::vector<double>;

// Simple linear interpolation helper (assumes x monotonically increasing)
double lerp(const std::vector<double>& x, const std::vector<double>& y, double xv);

// Load two-column CSV (wavelength nm, value) returns pair of vector.
bool load_csv(const std::string& path, std::vector<double>& wl, std::vector<double>& val);

/* Compute combined dichroic filter transmittance (Y,M,C) using error-function edges.
   filtering_amount_percent: Y,M,C intensities in % (0-100)
   transitions, edges size4 each
   Returns vector same length as wavelength grid.
*/
void combined_dichroic(const std::vector<double>& wavelength,
                       const std::array<double,3>& filtering_amount_percent,
                       const std::array<double,4>& transitions,
                       const std::array<double,4>& edges,
                       std::vector<double>& out,
                       double neutral_density_percent=0.0);

// Sigmoid using error-function
inline double sigmoid_erf(double x,double center,double width){return 0.5*(1.0+erf((x-center)/width));}

// Band-pass filter helper
dvec band_pass_filter(const std::vector<double>& wavelength,
                     std::array<double,3> filter_uv={1,410,8},
                     std::array<double,3> filter_ir={1,675,15});
} 