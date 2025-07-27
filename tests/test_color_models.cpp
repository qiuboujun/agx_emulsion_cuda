#include "color_models.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

static std::vector<double> load_txt(const std::string& path){
    std::ifstream f(path); std::vector<double> v; double x; while(f>>x) v.push_back(x); return v;
}

int main(){
    const double tol_xyz = 5e-4;   // 0.05% absolute
    const double tol_xy  = 5e-4;
    auto rgb_ref = load_txt("tests/color_rgb.txt");
    auto xyz_ref = load_txt("tests/color_xyz.txt");
    auto xy_ref  = load_txt("tests/color_xy.txt");

    const size_t M = rgb_ref.size()/3;
    const auto& cs = col::sRGB();

    double max_err_xyz = 0.0, max_err_rgb = 0.0, max_err_xy = 0.0;
    for(size_t i=0;i<M;++i){
        Eigen::Vector3d rgb(rgb_ref[3*i], rgb_ref[3*i+1], rgb_ref[3*i+2]);
        Eigen::Vector3d xyz_gpu = col::RGB_to_XYZ(rgb, cs);
        Eigen::Vector3d xyz_ref_v(xyz_ref[3*i], xyz_ref[3*i+1], xyz_ref[3*i+2]);
        max_err_xyz = std::max(max_err_xyz, (xyz_gpu - xyz_ref_v).cwiseAbs().maxCoeff());

        Eigen::Vector3d rgb_back = col::XYZ_to_RGB(xyz_gpu, cs);
        max_err_rgb = std::max(max_err_rgb, (rgb_back - rgb).cwiseAbs().maxCoeff());

        Eigen::Vector3d xy_gpu = col::XYZ_to_xy(xyz_gpu);
        Eigen::Vector3d xy_ref_v(xy_ref[3*i], xy_ref[3*i+1], 0.0);
        max_err_xy = std::max(max_err_xy, (xy_gpu - xy_ref_v).cwiseAbs().maxCoeff());
    }
    std::cout << "color max_err_xyz=" << max_err_xyz << " max_err_rgb=" << max_err_rgb << " max_err_xy=" << max_err_xy << std::endl;
    if(max_err_xyz>tol_xyz || max_err_rgb>1e-6 || max_err_xy>tol_xy){
        std::cerr << "Color model test FAILED" << std::endl; return 1;
    }
    std::cout << "Color model test PASSED" << std::endl; return 0;
} 