#pragma once
#include <vector>
#include <string>
#include <array>

namespace ds {
using dvec = std::vector<double>;

struct ChannelSpline { // simple linear for now
    dvec x, y;
    double operator()(double xv) const;
};

class DensitySpline {
public:
    bool load_csv_triplet(const std::string& folder); // expects density_curve_r.csv etc.
    // Evaluate RGB densities for given logE array
    void evaluate(const dvec& logE, std::vector<std::array<double,3>>& out) const;
private:
    std::array<ChannelSpline,3> _spl;
};
} 