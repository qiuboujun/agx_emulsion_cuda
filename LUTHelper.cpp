#include "LUTHelper.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>
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

static std::string parseViewingIlluminant(const std::string& stock) {
    printf("DEBUG: parseViewingIlluminant for '%s'\n", stock.c_str());
    
    std::vector<std::string> roots;
    roots.push_back("../data/profiles/");
    roots.push_back("./data/profiles/");
    roots.push_back("/usr/OFX/Plugins/data/profiles/");
    
    Dl_info info;
    if(dladdr((void*)&loadPrintLUT,&info) && info.dli_fname){
        std::string libPath(info.dli_fname);
        size_t pos = libPath.find_last_of('/');
        if(pos!=std::string::npos){
            std::string dir = libPath.substr(0,pos+1);
            roots.push_back(dir+"../../../data/profiles/");
        }
    }

    for(const auto& root: roots){
        std::string jsonPath = root + stock + ".json";
        printf("DEBUG: Trying JSON path: %s\n", jsonPath.c_str());
        
        std::ifstream f(jsonPath);
        if(!f.good()) continue;
        
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        
        // Simple JSON parsing for "viewing_illuminant" field
        size_t pos = content.find("\"viewing_illuminant\"");
        if(pos != std::string::npos) {
            size_t colon = content.find(":", pos);
            if(colon != std::string::npos) {
                size_t quote1 = content.find("\"", colon);
                if(quote1 != std::string::npos) {
                    size_t quote2 = content.find("\"", quote1 + 1);
                    if(quote2 != std::string::npos) {
                        std::string illuminant = content.substr(quote1 + 1, quote2 - quote1 - 1);
                        printf("DEBUG: Found viewing illuminant: '%s'\n", illuminant.c_str());
                        return illuminant;
                    }
                }
            }
        }
    }
    
    printf("DEBUG: No viewing illuminant found, defaulting to D50\n");
    return "D50";
}

extern "C" bool loadPrintLUT(const char* stock, float* logE, float* r, float* g, float* b, char* illuminant_out) {
    printf("DEBUG LUT: Searching for print stock '%s'\n", stock);
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
    
    // Parse viewing illuminant from profile JSON
    std::string illuminant = parseViewingIlluminant(stock);
    strncpy(illuminant_out, illuminant.c_str(), 15);
    illuminant_out[15] = '\0';  // Ensure null termination
    
    return true;
} 

extern "C" bool loadPaperSpectra(const char* stock, float* c, float* m, float* y, float* dmin, int* count) {
    printf("DEBUG: loadPaperSpectra for '%s'\n", stock);
    
    std::vector<std::string> roots;
    roots.push_back("../data/paper/");
    roots.push_back("./data/paper/");
    roots.push_back("/usr/OFX/Plugins/data/paper/");
    
    Dl_info info;
    if(dladdr((void*)&loadPaperSpectra,&info) && info.dli_fname){
        std::string libPath(info.dli_fname);
        size_t pos = libPath.find_last_of('/');
        if(pos!=std::string::npos){
            std::string dir = libPath.substr(0,pos+1);
            roots.push_back(dir+"../../../data/paper/");
        }
    }

    std::string base="";
    for(const auto& root: roots){
        std::string test = root + stock + "/";
        std::ifstream f(test+"dye_density_c.csv");
        if(f.good()){ 
            base = test; 
            break; 
        }
    }
    if(base.empty()) return false;
    
    auto readVec = [](const std::string& path) -> std::vector<float> {
        std::vector<float> vec;
        std::ifstream f(path);
        if(!f.good()) return vec;
        
        std::string line;
        while(std::getline(f,line)){
            if(line.empty() || line[0]=='#') continue;
            size_t comma = line.find(',');
            if(comma==std::string::npos) continue;
            std::string valStr = line.substr(comma+1);
            float v = std::stof(valStr);
            vec.push_back(v);
        }
        return vec;
    };
    
    auto cVec = readVec(base+"dye_density_c.csv");
    auto mVec = readVec(base+"dye_density_m.csv");
    auto yVec = readVec(base+"dye_density_y.csv");
    auto dminVec = readVec(base+"dye_density_min.csv");
    
    if(cVec.empty() || mVec.empty() || yVec.empty()) return false;
    
    int n = std::min({(int)cVec.size(), (int)mVec.size(), (int)yVec.size(), 200});
    for(int i=0; i<n; i++){
        c[i] = cVec[i];
        m[i] = mVec[i];
        y[i] = yVec[i];
        dmin[i] = dminVec.empty() ? 0.0f : (i < (int)dminVec.size() ? dminVec[i] : 0.0f);
    }
    *count = n;
    
    printf("DEBUG: Loaded %d spectral samples\n", n);
    return true;
}

extern "C" bool loadIlluminantSPD(const char* name, float* spd, int* count) {
    printf("DEBUG: loadIlluminantSPD for '%s'\n", name);
    
    // Map illuminant names to pre-computed arrays from CIE1931.cuh
    // These will be populated by including the header and accessing the constants
    if(strcmp(name, "D50") == 0) {
        // Will be filled by caller with c_d50SPD
        *count = 81;
        return true;
    } else if(strcmp(name, "D65") == 0) {
        // Will be filled by caller with c_d65SPD
        *count = 81;
        return true;
    } else if(strcmp(name, "K75P") == 0) {
        // Will be filled by caller with c_k75pSPD
        *count = 81;
        return true;
    } else {
        printf("DEBUG: Unknown illuminant '%s', defaulting to D50\n", name);
        *count = 81;
        return true;  // Caller will use D50 as fallback
    }
} 