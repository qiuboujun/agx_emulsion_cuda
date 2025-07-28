#include "interpolate.hpp"
#include <algorithm>

namespace interp {

static double interp_linear(const dvec& x,const dvec& y,double xv){
    if(xv<=x.front()) return y.front();
    if(xv>=x.back()) return y.back();
    auto it=std::lower_bound(x.begin(),x.end(),xv);
    size_t idx=std::distance(x.begin(),it);
    double x0=x[idx-1], x1=x[idx];
    double t=(xv-x0)/(x1-x0);
    return y[idx-1]*(1-t)+y[idx]*t;
}

void exposure_to_density(const dvec& log_raw,
                         const std::vector<std::array<double,3>>& density_curves,
                         const dvec& log_exposure,
                         const std::array<double,3>& gamma_factor,
                         std::vector<std::array<double,3>>& out){
    size_t n=log_raw.size();
    out.resize(n);
    // Prepare per-channel x arrays scaled by gamma
    size_t m=log_exposure.size();
    dvec xR(m),xG(m),xB(m);
    for(size_t i=0;i<m;i++){
        xR[i]=log_exposure[i]/gamma_factor[0];
        xG[i]=log_exposure[i]/gamma_factor[1];
        xB[i]=log_exposure[i]/gamma_factor[2];
    }
    dvec yR(m),yG(m),yB(m);
    for(size_t i=0;i<m;i++){
        yR[i]=density_curves[i][0];
        yG[i]=density_curves[i][1];
        yB[i]=density_curves[i][2];
    }
    for(size_t i=0;i<n;i++){
        double v=log_raw[i];
        out[i][0]=interp_linear(xR,yR,v);
        out[i][1]=interp_linear(xG,yG,v);
        out[i][2]=interp_linear(xB,yB,v);
    }
}
} 