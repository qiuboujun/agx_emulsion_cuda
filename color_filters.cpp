#include "color_filters.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace cf {

double lerp(const std::vector<double>& x,const std::vector<double>& y,double xv){
    if(xv<=x.front()) return y.front();
    if(xv>=x.back()) return y.back();
    auto it = std::lower_bound(x.begin(),x.end(),xv);
    size_t idx = std::distance(x.begin(),it);
    if(idx==0) return y[0];
    double x0=x[idx-1], x1=x[idx];
    double t=(xv-x0)/(x1-x0);
    return y[idx-1]*(1.0-t)+y[idx]*t;
}

bool load_csv(const std::string& path,std::vector<double>& wl,std::vector<double>& val){
    std::ifstream f(path);
    if(!f.good()) return false;
    std::string line;
    while(std::getline(f,line)){
        if(line.empty()) continue;
        std::istringstream ss(line);
        double a,b; char comma;
        if(!(ss>>a)) continue;
        if(ss.peek()==','||ss.peek()=='\t') ss>>comma; // consume delimiter
        if(!(ss>>b)) continue;
        wl.push_back(a);
        val.push_back(b);
    }
    return !wl.empty();
}

void combined_dichroic(const std::vector<double>& wavelength,
                       const std::array<double,3>& filtering_amount_percent,
                       const std::array<double,4>& transitions,
                       const std::array<double,4>& edges,
                       std::vector<double>& out,
                       double nd_percent){
    size_t n=wavelength.size();
    out.resize(n);
    std::array<double,3> filtAmt;
    for(int i=0;i<3;++i) filtAmt[i]=filtering_amount_percent[i]/100.0;

    for(size_t i=0;i<n;++i){
        double wl=wavelength[i];
        double d0 = 0.5*(1.0+erf((wl-edges[0])/transitions[0]));
        double d1;
        if(wl<=550)
            d1 = 0.5*(1.0-erf((wl-edges[1])/transitions[1]));
        else
            d1 = 0.5*(1.0+erf((wl-edges[2])/transitions[2]));
        double d2 = 0.5*(1.0-erf((wl-edges[3])/transitions[3]));
        double total = ((1-filtAmt[0])+d0*filtAmt[0]) * ((1-filtAmt[1])+d1*filtAmt[1]) * ((1-filtAmt[2])+d2*filtAmt[2]);
        total *= (100.0-nd_percent)/100.0;
        out[i]=total;
    }
}

dvec band_pass_filter(const std::vector<double>& wavelength,
                      std::array<double,3> filter_uv,
                      std::array<double,3> filter_ir){
    double amp_uv=std::clamp(filter_uv[0],0.0,1.0);
    double wl_uv=filter_uv[1];
    double wid_uv=filter_uv[2];
    double amp_ir=std::clamp(filter_ir[0],0.0,1.0);
    double wl_ir=filter_ir[1];
    double wid_ir=filter_ir[2];
    dvec out(wavelength.size());
    for(size_t i=0;i<wavelength.size();++i){
        double wl=wavelength[i];
        double fuv = 1-amp_uv + amp_uv*sigmoid_erf(wl, wl_uv, wid_uv);
        double fir = 1-amp_ir + amp_ir*sigmoid_erf(wl, wl_ir, -wid_ir);
        out[i]=fuv*fir;
    }
    return out;
}

} // namespace cf 