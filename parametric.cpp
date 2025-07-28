#include "parametric.hpp"
#include <cmath>
#include <algorithm>

namespace prm {

double log10_safe(double v){return std::log10(v);}

void parametric_density_curves(const dvec& logE,
                               const std::array<double,3>& gamma,
                               const std::array<double,3>& logE0,
                               const std::array<double,3>& densityMax,
                               const std::array<double,3>& toeSize,
                               const std::array<double,3>& shoulderSize,
                               std::vector<std::array<double,3>>& out){
    size_t n=logE.size();
    out.resize(n);
    for(size_t i=0;i<n;i++)
        for(int c=0;c<3;c++) out[i][c]=0.0;
    for(int c=0;c<3;c++){
        double g=gamma[c];
        double loge0=logE0[c];
        double dmax=densityMax[c];
        double ts=toeSize[c];
        double ss=shoulderSize[c];
        for(size_t i=0;i<n;i++){
            double le=logE[i];
            double term1= g*ts*log10_safe(1.0 + std::pow(10.0,(le - loge0)/ts));
            double term2= g*ss*log10_safe(1.0 + std::pow(10.0,(le - loge0 - dmax/g)/ss));
            out[i][c]=term1 - term2;
        }
    }
}
} 