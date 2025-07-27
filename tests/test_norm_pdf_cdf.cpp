#include "gp_numpy.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

static gpnp::hvec load_txt(const std::string& path){
    std::ifstream f(path);
    std::vector<double> vals; double v;
    while(f>>v) vals.push_back(v);
    return gpnp::hvec(vals.begin(), vals.end());
}

int main(){
    const double tol = 1e-6;
    auto x_h   = load_txt("tests/x_vals.txt");
    auto pdf_h = load_txt("tests/pdf_ref.txt");
    auto cdf_h = load_txt("tests/cdf_ref.txt");

    gpnp::dvec x_d(x_h.begin(), x_h.end());
    gpnp::dvec pdf_d, cdf_d;
    gpnp::norm_pdf(x_d, pdf_d);
    gpnp::norm_cdf(x_d, cdf_d);

    gpnp::hvec pdf_gpu(pdf_d.begin(), pdf_d.end());
    gpnp::hvec cdf_gpu(cdf_d.begin(), cdf_d.end());

    double max_err_pdf = 0.0, max_err_cdf = 0.0;
    for(size_t i=0;i<x_h.size();++i){
        max_err_pdf = std::max(max_err_pdf, std::abs(pdf_gpu[i]-pdf_h[i]));
        max_err_cdf = std::max(max_err_cdf, std::abs(cdf_gpu[i]-cdf_h[i]));
    }

    std::cout << "max_err_pdf=" << max_err_pdf << " max_err_cdf=" << max_err_cdf << std::endl;
    if(max_err_pdf > tol || max_err_cdf > tol){
        std::cerr << "Test failed: error above tolerance" << std::endl;
        return 1;
    }
    std::cout << "Test passed" << std::endl;
    return 0;
} 