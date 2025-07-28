#include "couplers.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

static std::vector<double> load1d(const char* p){std::ifstream f(p);double v;std::vector<double> o;while(f>>v) o.push_back(v);return o;}

int main(){
    // load reference data
    auto logE=load1d("tests/dens_loge.txt");
    auto density_flat=load1d("tests/dens_curve.txt");
    size_t n=logE.size();
    std::vector<std::array<double,3>> density_curves(n);
    for(size_t i=0;i<n;i++){
        density_curves[i][0]=density_flat[i*3+0];
        density_curves[i][1]=density_flat[i*3+1];
        density_curves[i][2]=density_flat[i*3+2];
    }
    auto dir_mat_flat=load1d("tests/dir_matrix.txt");
    cp::Matrix3 M;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) M[i][j]=dir_mat_flat[i*3+j];

    std::array<double,3> amount={0.7,0.7,0.5};
    auto Mcalc=cp::compute_dir_couplers_matrix(amount,1.0);
    double errMat=0.0;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) errMat=std::max(errMat,std::abs(M[i][j]-Mcalc[i][j]));

    std::vector<std::array<double,3>> corr_cpp;
    cp::compute_density_curves_before_dir_couplers(density_curves,logE,M,0.0,corr_cpp);
    auto corr_flat=load1d("tests/corr_curve.txt");
    double maxErr=errMat;
    for(size_t i=0;i<n;i++) for(int c=0;c<3;c++)
        maxErr=std::max(maxErr,std::abs(corr_cpp[i][c]-corr_flat[i*3+c]));
    std::cout<<"couplers maxErr="<<maxErr<<"\n";
    return maxErr<1e-6?0:1;
} 