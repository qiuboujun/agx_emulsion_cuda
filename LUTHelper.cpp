#include "LUTHelper.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <cstdio> // Required for printf

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
    printf("DEBUG LUT: Searching for stock '%s'\n", stock);
    
    std::vector<std::string> roots;
    // caller provided relative default (from Resolve cwd)
    roots.push_back("../data/film/negative/");
    roots.push_back("./data/film/negative/");
    // system-wide OFX plugins directory (Resolve/Natron common install)
    roots.push_back("/usr/OFX/Plugins/data/film/negative/");
    // path relative to library directory
    Dl_info info;
    if(dladdr((void*)&loadFilmLUT,&info) && info.dli_fname){
        std::string libPath(info.dli_fname);
        printf("DEBUG LUT: Library path: %s\n", libPath.c_str());
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
        printf("DEBUG LUT: Trying path: %s\n", test.c_str());
        std::ifstream f(test+"density_curve_r.csv");
        if(!f.good()) f.open((test+"dye_density_y.csv").c_str());
        if(f.good()){ 
            printf("DEBUG LUT: Found data at: %s\n", test.c_str());
            base = test; 
            break; 
        } else {
            printf("DEBUG LUT: Failed to open: %s\n", (test+"density_curve_r.csv").c_str());
        }
    }
    if(base.empty()) {
        printf("DEBUG LUT: No valid path found for stock '%s'\n", stock);
        return false;
    }
    
    std::vector<float> x, yr, yg, yb;
    if(!loadCurve(base+"density_curve_r.csv", x, yr)) {
        printf("DEBUG LUT: Failed to load red curve from %s\n", (base+"density_curve_r.csv").c_str());
        return false;
    }
    std::vector<float> xg; 
    if(!loadCurve(base+"density_curve_g.csv", xg, yg)) {
        printf("DEBUG LUT: Failed to load green curve from %s\n", (base+"density_curve_g.csv").c_str());
        return false;
    }
    std::vector<float> xb; 
    if(!loadCurve(base+"density_curve_b.csv", xb, yb)) {
        printf("DEBUG LUT: Failed to load blue curve from %s\n", (base+"density_curve_b.csv").c_str());
        return false;
    }

    printf("DEBUG LUT: Successfully loaded %zu samples\n", x.size());
    
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

extern "C" bool loadPrintLUT(const char* stock, float* logE, float* r, float* g, float* b) {
    printf("DEBUG LUT PRINT: Searching for print stock '%s'\n", stock);
    std::vector<std::string> roots;
    roots.push_back("../data/paper/");
    roots.push_back("./data/paper/");
    roots.push_back("/usr/OFX/Plugins/data/paper/");
    Dl_info info;
    if(dladdr((void*)&loadPrintLUT,&info) && info.dli_fname){
        std::string libPath(info.dli_fname);
        size_t pos = libPath.find_last_of('/') ;
        if(pos!=std::string::npos){
            std::string dir = libPath.substr(0,pos+1);
            roots.push_back(dir+"../../../data/paper/");
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
    bool ok=false;
    if(loadCurve(base+"density_curve_r.csv", x, yr)){
        std::vector<float> xg; if(!loadCurve(base+"density_curve_g.csv", xg, yg)) return false;
        std::vector<float> xb; if(!loadCurve(base+"density_curve_b.csv", xb, yb)) return false;
        ok=true;
    } else {
        if(loadCurve(base+"dye_density_y.csv", x, yr)){
            std::vector<float> xg; if(!loadCurve(base+"dye_density_m.csv", xg, yg)) return false;
            std::vector<float> xb; if(!loadCurve(base+"dye_density_c.csv", xb, yb)) return false;
            ok=true;
        }
    }
    if(!ok) return false;
    printf("DEBUG PRINT LUT: Loaded %zu samples, logE range [%f, %f]\n", 
           x.size(), x.empty() ? 0.0f : x[0], x.empty() ? 0.0f : x[x.size()-1]);
    printf("DEBUG PRINT LUT: Sample curves R[0]=%f R[last]=%f\n", 
           yr.empty() ? 0.0f : yr[0], yr.empty() ? 0.0f : yr[yr.size()-1]);
    
    size_t n = std::min((size_t)601, x.size());
    for(size_t i=0;i<n;i++){logE[i]=x[i]; r[i]=yr[i]; g[i]=yg[i]; b[i]=yb[i];}
    for(size_t i=n;i<601;i++){logE[i]=r[i]=g[i]=b[i]=0.f;}
    return true;
} 

extern "C" bool loadPaperSpectra(const char* stock,float* c,float* m,float* y,float* dmin,int* count){
    printf("DEBUG SPECTRA: Loading paper spectra '%s'\n",stock);
    std::vector<std::string> roots={"../data/paper/","./data/paper/","/usr/OFX/Plugins/data/paper/"};
    Dl_info info; if(dladdr((void*)&loadPaperSpectra,&info)&&info.dli_fname){std::string libPath(info.dli_fname);size_t pos=libPath.find_last_of('/');if(pos!=std::string::npos){roots.push_back(libPath.substr(0,pos+1)+"../../../data/paper/");}}
    std::string base="";
    for(const auto& r:roots){std::string test=r+stock+"/";std::ifstream f(test+"dye_density_c.csv");if(f.good()){base=test;break;}}
    if(base.empty()) return false;
    auto readVec=[&](const std::string& path,std::vector<float>& vec){
        std::ifstream f(path);
        if(!f.good()) return false;
        std::string line;
        while(std::getline(f,line)){
            if(line.empty()) continue;
            size_t comma = line.find(',');
            if(comma==std::string::npos) continue;
            std::string valStr = line.substr(comma+1);
            float v = std::stof(valStr);
            vec.push_back(v);
        }
        return !vec.empty();
    };
    std::vector<float> vc,vm,vy,vd;
    if(!readVec(base+"dye_density_c.csv",vc)) return false;
    if(!readVec(base+"dye_density_m.csv",vm)) return false;
    if(!readVec(base+"dye_density_y.csv",vy)) return false;
    readVec(base+"dye_density_min.csv",vd);
    int n=vc.size(); if(count) *count=n;
    for(int i=0;i<n;i++){c[i]=vc[i];m[i]=vm[i];y[i]=vy[i]; dmin[i]=(i<vd.size()?vd[i]:0.f);} return true; } 