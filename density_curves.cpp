#include "density_curves.hpp"
#include <algorithm>
#include <numeric>
#include <cassert>

namespace dc {

static inline double norm_cdf(double x){
    // approximate using std::erf
    return 0.5*(1.0+std::erf(x/std::sqrt(2.0)));
}

void density_curve_norm_cdfs(const dvec& logE,const std::array<double,9>& p,CurveType type,dvec& out){
    size_t n=logE.size(); out.assign(n,0.0);
    std::array<double,3> centers{p[0],p[1],p[2]};
    std::array<double,3> amps   {p[3],p[4],p[5]};
    std::array<double,3> sigmas {p[6],p[7],p[8]};
    for(int layer=0;layer<3;++layer){
        double c=centers[layer]; double a=amps[layer]; double s=sigmas[layer];
        for(size_t i=0;i<n;++i){
            double arg=(logE[i]-c)/s;
            double val = norm_cdf((type==CurveType::Positive)?-arg:arg)*a;
            out[i]+=val;
        }
    }
}

void density_curves_rgb_norm(const dvec& logE,const std::array<std::array<double,9>,3>& params,CurveType type,std::vector<std::array<double,3>>& out){
    size_t n=logE.size(); out.resize(n);
    dvec tmp(n);
    for(int ch=0;ch<3;++ch){
        density_curve_norm_cdfs(logE,params[ch],type,tmp);
        for(size_t i=0;i<n;++i) out[i][ch]=tmp[i];
    }
}

// ----- log-line model -----
void density_curve_log_line(const dvec& logE,const std::array<double,10>& x,CurveType type,dvec& out){
    size_t n=logE.size(); out.resize(n);
    double D_min=x[0];
    double gamma=x[1];
    double H_reference=x[2];
    double D_range=x[3];
    double curvature_toe=x[4];
    double curvature_shoulder=x[5];
    double curvature_toe_slope=x[6];
    double curvature_toe_max=x[7];
    double curvature_shoulder_slope=x[8];
    double curvature_shoulder_max=x[9];

    double H0;
    if(type==CurveType::Negative)
        H0=H_reference-1.0/gamma;
    else
        H0=H_reference-0.735/gamma;

    if(type==CurveType::Positive){
        gamma=-gamma;
        curvature_toe_slope=-curvature_toe_slope;
        curvature_shoulder_slope=-curvature_shoulder_slope;
    }

    auto sigmoid=[&](double v){return 1.0/(1.0+std::exp(-4.0*gamma*v));};
    for(size_t i=0;i<n;++i){
        double LE=logE[i];
        double morph_shoulder = gamma*curvature_shoulder*(1.0+curvature_shoulder_max*sigmoid(-curvature_shoulder_slope*(LE - H0 - D_range/gamma)));
        double morph_toe     = gamma*curvature_toe    *(1.0+curvature_toe_max    *sigmoid( curvature_toe_slope    *(LE - H0)));
        double rise = gamma/morph_toe * std::log10(1.0 + std::pow(10.0, morph_toe*(LE - H0)));
        double stop = gamma/morph_shoulder * std::log10(1.0 + std::pow(10.0, morph_shoulder*(LE - D_range/std::abs(gamma) - H0)));
        double D;
        if(type==CurveType::Positive)
            D = D_min - rise + stop;
        else
            D = D_min + rise - stop;
        out[i]=D;
    }
}

} // namespace dc 