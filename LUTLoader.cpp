#include "LUTLoader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

static bool loadCurve(const std::string& path,std::vector<float>& x, std::vector<float>& y){
    std::ifstream f(path);
    if(!f.good()) {std::cerr<<"Failed to open "<<path<<"\n";return false;}
    std::string line; float a,b; char c;
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

bool loadFilmLUT(const std::string& stockFolder, FilmLUT& out){
    std::string base = stockFolder;
    if(base.back()!='/') base += "/";
    std::vector<float> x;
    if(!loadCurve(base+"density_curve_r.csv",x,out.curveR)) return false;
    std::vector<float> xg; if(!loadCurve(base+"density_curve_g.csv",xg,out.curveG)) return false;
    std::vector<float> xb; if(!loadCurve(base+"density_curve_b.csv",xb,out.curveB)) return false;
    out.logE = x;
    return true;
} 