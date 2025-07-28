#include "emulsion.hpp"
#include "couplers.hpp"
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

    std::array<double,3> amt={0.7,0.7,0.5};
    auto M=cp::compute_dir_couplers_matrix(amt,1.0);

    emu::FilmSimple film(logE,curves,1.2,true,M,0.0);
    std::vector<std::array<double,3>> out;
    film.develop(log_raw,out);
    // ensure output differs from noncoupler
    emu::FilmSimple film0(logE,curves,1.2,false);
    std::vector<std::array<double,3>> base;film0.develop(log_raw,base);
    double diff=0.0;for(size_t i=0;i<out.size();i++) diff+=std::abs(out[i][0]-base[i][0]);
    std::cout<<"coupler effect diff="<<diff<<"\n";
    return diff>0?0:1;
} 