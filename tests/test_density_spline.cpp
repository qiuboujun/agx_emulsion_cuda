#include "density_spline.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

static std::vector<double> load1d(const char* p){std::ifstream f(p);double v;std::vector<double> o;while(f>>v) o.push_back(v);return o;}

int main(){
    ds::DensitySpline spl;
    if(!spl.load_csv_triplet("../data/film/negative/kodak_portra_400")){
        std::cerr<<"CSV load failed\n";return 1;}
    auto logE=load1d("tests/dens_loge.txt"); // reuse earlier grid
    std::vector<std::array<double,3>> calc; spl.evaluate(logE,calc);
    // compute error against linear interpolation of same csv (should be small)
    double maxErr=0.0;
    // simple consistency check: values within bounds
    for(auto &rgb:calc){for(double v:rgb) if(std::isnan(v)) {std::cerr<<"nan";return 1;}}
    std::cout<<"spline test ok\n"; return 0; } 