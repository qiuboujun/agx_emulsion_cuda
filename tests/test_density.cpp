#include "density_curves.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

static std::vector<double> load1d(const char* path){std::ifstream f(path);double v;std::vector<double> out;while(f>>v) out.push_back(v);return out;}

int main(){
    auto logE=load1d("tests/dens_loge.txt");
    auto curves_flat=load1d("tests/dens_curve.txt");
    size_t n=logE.size();
    std::array<std::array<double,9>,3> params={{{0,1,2, 0.5,0.5,0.5, 0.3,0.5,0.7},
                                               {0,1,2, 0.5,0.5,0.5, 0.3,0.5,0.7},
                                               {0,1,2, 0.5,0.5,0.5, 0.3,0.5,0.7}}};
    std::vector<std::array<double,3>> calc;
    dc::density_curves_rgb_norm(logE,params,dc::CurveType::Negative,calc);
    double maxErr=0.0;
    for(size_t i=0;i<n;i++){
        for(int c=0;c<3;c++){
            double ref=curves_flat[i*3+c];
            maxErr=std::max(maxErr,std::abs(ref-calc[i][c]));
        }
    }
    std::cout<<"density norm maxErr="<<maxErr<<"\n";
    return maxErr<1e-6?0:1;
} 