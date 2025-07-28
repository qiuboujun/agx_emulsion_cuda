#include "parametric.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

static std::vector<double> load1d(const char* p){std::ifstream f(p);double v;std::vector<double> o;while(f>>v) o.push_back(v);return o;}

int main(){
    auto logE=load1d("tests/para_loge.txt");
    auto curves_flat=load1d("tests/para_curve.txt");
    std::array<double,3> gamma={0.65,0.7,0.68};
    std::array<double,3> loge0={-1.0,-1.0,-1.0};
    std::array<double,3> dmax={2.0,2.0,2.0};
    std::array<double,3> toe={0.2,0.2,0.2};
    std::array<double,3> shoulder={0.3,0.3,0.3};
    std::vector<std::array<double,3>> calc;
    prm::parametric_density_curves(logE,gamma,loge0,dmax,toe,shoulder,calc);
    double maxErr=0.0;
    size_t n=logE.size();
    for(size_t i=0;i<n;i++){
        for(int c=0;c<3;c++){
            double ref=curves_flat[i*3+c];
            maxErr=std::max(maxErr,std::abs(ref-calc[i][c]));
        }
    }
    std::cout<<"parametric maxErr="<<maxErr<<"\n";
    return maxErr<1e-6?0:1;
} 