#include "LUTHelper.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <dlfcn.h>

static bool loadCurve(const std::string& path, std::vector<float>& x, std::vector<float>& y) {
    std::ifstream f(path);
    if(!f.good()) return false;
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

extern "C" bool loadFilmLUT(const char* stock, float* logE, float* r, float* g, float* b) {
    std::vector<std::string> roots;
    // caller provided relative default (from Resolve cwd)
    roots.push_back("../data/film/negative/");
    roots.push_back("./data/film/negative/");
    // path relative to library directory
    Dl_info info;
    if(dladdr((void*)&loadFilmLUT,&info) && info.dli_fname){
        std::string libPath(info.dli_fname);
        size_t pos = libPath.find_last_of('/') ;
        if(pos!=std::string::npos){
            std::string dir = libPath.substr(0,pos+1); // dir of .so
            // bundle path: <bundle>/Contents/Linux-x86-64/  -> we need ../../../data/film/negative/
            roots.push_back(dir+"../../../data/film/negative/");
        }
    }

    std::string base="";
    for(const auto& root: roots){
        std::string test = root + stock + "/";
        std::ifstream f(test+"density_curve_r.csv");
        if(f.good()){ base = test; break; }
    }
    if(base.empty()) return false;
    
    std::vector<float> x, yr, yg, yb;
    if(!loadCurve(base+"density_curve_r.csv", x, yr)) return false;
    std::vector<float> xg; if(!loadCurve(base+"density_curve_g.csv", xg, yg)) return false;
    std::vector<float> xb; if(!loadCurve(base+"density_curve_b.csv", xb, yb)) return false;
    
    // Copy to output arrays (assuming 601 samples)
    size_t n = std::min((size_t)601, x.size());
    for(size_t i=0; i<n; i++) {
        logE[i] = x[i];
        r[i] = yr[i];
        g[i] = yg[i];
        b[i] = yb[i];
    }
    // Zero remaining if less than 601
    for(size_t i=n; i<601; i++) {
        logE[i] = r[i] = g[i] = b[i] = 0.0f;
    }
    
    return true;
} 