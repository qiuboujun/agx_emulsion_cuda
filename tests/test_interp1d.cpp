#include "gp_scipy.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

static std::vector<double> load_txt(const std::string& path){
    std::ifstream f(path);
    std::vector<double> vals; double v; while(f>>v) vals.push_back(v);
    return vals;
}

int main(){
    const double tol = 1e-6;
    auto x_h   = load_txt("tests/interp_x.txt");
    auto y_h   = load_txt("tests/interp_y.txt");
    auto xq_h  = load_txt("tests/interp_xq.txt");
    auto y_lin_ref  = load_txt("tests/interp_yq_lin.txt");
    auto y_near_ref = load_txt("tests/interp_yq_near.txt");

    gpnp::dvec x_d(x_h.begin(), x_h.end());
    gpnp::dvec y_d(y_h.begin(), y_h.end());
    gpnp::dvec xq_d(xq_h.begin(), xq_h.end());

    gpnp::dvec y_lin_d, y_near_d;
    gpnp::interp1d(x_d, y_d, xq_d, y_lin_d, "linear");
    gpnp::interp1d(x_d, y_d, xq_d, y_near_d, "nearest");

    gpnp::hvec y_lin_g(y_lin_d.begin(), y_lin_d.end());
    gpnp::hvec y_near_g(y_near_d.begin(), y_near_d.end());

    double max_err_lin = 0.0, max_err_near = 0.0;
    for(size_t i=0;i<xq_h.size();++i){
        max_err_lin  = std::max(max_err_lin,  std::abs(y_lin_g[i]-y_lin_ref[i]));
        max_err_near = std::max(max_err_near, std::abs(y_near_g[i]-y_near_ref[i]));
    }
    std::cout << "interp1d max_err_lin=" << max_err_lin << " max_err_near=" << max_err_near << std::endl;
    if(max_err_lin>tol || max_err_near>tol){
        std::cerr << "interp1d test FAILED" << std::endl; return 1;
    }
    std::cout << "interp1d test PASSED" << std::endl; return 0;
} 