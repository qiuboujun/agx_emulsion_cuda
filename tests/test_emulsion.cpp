#include "emulsion.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

static std::vector<double> load1d(const char* p){std::ifstream f(p);double v;std::vector<double> o;while(f>>v) o.push_back(v);return o;}

int main(){
    auto logE=load1d("tests/dens_loge.txt");
    auto curves_flat=load1d("tests/dens_curve.txt");
    std::vector<std::array<double,3>> curves(logE.size());
    for(size_t i=0;i<logE.size();i++){
        curves[i][0]=curves_flat[i*3+0];
        curves[i][1]=curves_flat[i*3+1];
        curves[i][2]=curves_flat[i*3+2];
    }

    auto log_raw=load1d("tests/emul_lograw.txt");
    std::array<double,3> gamma={1.2,1.2,1.2};

    emu::FilmSimple film(logE,curves,1.2,false);
    std::vector<std::array<double,3>> out;
    film.develop(log_raw,out);

    auto ref_flat=load1d("tests/emul_density.txt");
    double maxErr=0.0;
    for(size_t i=0;i<out.size();i++){
        for(int c=0;c<3;c++){
            double ref=ref_flat[i*3+c];
            maxErr=std::max(maxErr,std::abs(ref-out[i][c]));
        }
    }
    std::cout<<"emulsion maxErr="<<maxErr<<"\n";
    return maxErr<1e-6?0:1;
} 