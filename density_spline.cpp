#include "density_spline.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>

namespace ds {
static bool read_csv(const std::string& path, dvec& x, dvec& y){
    std::ifstream f(path);
    if(!f.good()) return false;
    std::string line; double a,b; char c;
    while(std::getline(f,line)){
        if(line.empty()) continue;
        std::replace(line.begin(),line.end(),';',',');
        std::istringstream ss(line);
        if(!(ss>>a)) continue;
        if(ss.peek()==','||ss.peek()=='\t') ss>>c;
        if(!(ss>>b)) continue;
        x.push_back(a); y.push_back(b);
    }
    return !x.empty();
}

double ChannelSpline::operator()(double xv) const{
    if(xv<=x.front()) return y.front();
    if(xv>=x.back()) return y.back();
    auto it = std::lower_bound(x.begin(),x.end(),xv);
    size_t idx=std::distance(x.begin(),it);
    double x0=x[idx-1], x1=x[idx];
    double t=(xv-x0)/(x1-x0);
    return y[idx-1]*(1-t)+y[idx]*t;
}

bool DensitySpline::load_csv_triplet(const std::string& folder){
    const char* names[3]={"density_curve_r.csv","density_curve_g.csv","density_curve_b.csv"};
    for(int ch=0;ch<3;++ch){
        dvec x,y; if(!read_csv(folder+"/"+names[ch],x,y)) return false;
        _spl[ch].x=std::move(x); _spl[ch].y=std::move(y);
    }
    return true;
}

void DensitySpline::evaluate(const dvec& logE,std::vector<std::array<double,3>>& out) const{
    out.resize(logE.size());
    for(size_t i=0;i<logE.size();++i){
        double v=logE[i];
        out[i][0]=_spl[0](v);
        out[i][1]=_spl[1](v);
        out[i][2]=_spl[2](v);
    }
}

} 