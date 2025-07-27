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
    auto x_h  = load_txt("tests/interp_x.txt");
    auto y_h  = load_txt("tests/interp_y.txt");
    auto xq_h = load_txt("tests/interp_xq.txt");
    auto y_ref = load_txt("tests/interp_yq_spline.txt");

    gpnp::dvec x_d(x_h.begin(), x_h.end());
    gpnp::dvec y_d(y_h.begin(), y_h.end());
    gpnp::dvec xq_d(xq_h.begin(), xq_h.end());

    // Build spline on GPU (coeffs on host)
    auto sp = gpnp::splrep(x_d, y_d);
    std::cout << "GPU spline coefficients:" << std::endl;
    std::cout << "a = ";
    for(auto v : sp.a) std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "b = ";
    for(auto v : sp.b) std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "c = ";
    for(auto v : sp.c) std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "d = ";
    for(auto v : sp.d) std::cout << v << " ";
    std::cout << std::endl;
    gpnp::dvec y_d_gpu;
    gpnp::splev(sp, xq_d, y_d_gpu);

    gpnp::hvec y_gpu(y_d_gpu.begin(), y_d_gpu.end());
    double max_err = 0.0;
    for(size_t i=0;i<y_ref.size();++i)
        max_err = std::max(max_err, std::abs(y_gpu[i]-y_ref[i]));

    std::cout << "spline max_err=" << max_err << std::endl;
    if(max_err>tol){
        std::cerr << "spline test FAILED" << std::endl; return 1;
    }
    std::cout << "spline test PASSED" << std::endl; return 0;
} 