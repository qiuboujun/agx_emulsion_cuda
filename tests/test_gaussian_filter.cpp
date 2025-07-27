#include "kernels.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

static std::vector<double> load_txt(const std::string& path){
    std::ifstream f(path);
    std::vector<double> vals; double v;
    while(f>>v) vals.push_back(v);
    return vals;
}

int main(){
    const double tol = 1e-6;
    const int H = 10, W = 10;
    /* Load input & reference output ---------------------------------- */
    auto in_vals  = load_txt("tests/gaussian_input.txt");
    auto ref_vals = load_txt("tests/gaussian_ref.txt");
    if(in_vals.size()!=H*W || ref_vals.size()!=H*W){
        std::cerr << "Unexpected file size." << std::endl;
        return 1;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> input(H, W), ref(H, W);
    for(int i=0;i<H*W;++i){
        input.data()[i] = in_vals[i];
        ref.data()[i]   = ref_vals[i];
    }

    /* Run GPU Gaussian filter ---------------------------------------- */
    double sigma = 1.2;
    auto out = gaussian_filter(input, sigma);

    /* Compute maximum absolute error --------------------------------- */
    double max_err = 0.0;
    for(int i=0;i<H*W;++i){
        max_err = std::max(max_err, std::abs(out.data()[i]-ref.data()[i]));
    }

    std::cout << "max_err_gaussian=" << max_err << std::endl;
    if(max_err>tol){
        std::cerr << "Gaussian filter test FAILED" << std::endl;
        return 1;
    }
    std::cout << "Gaussian filter test PASSED" << std::endl;
    return 0;
} 