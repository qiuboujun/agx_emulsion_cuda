#include "color_models.hpp"

namespace col {

static const ColorSpace sRGB_cs = [](){
    ColorSpace cs;
    cs.M_RGB_to_XYZ << 0.4124564, 0.3575761, 0.1804375,
                      0.2126729, 0.7151522, 0.0721750,
                      0.0193339, 0.1191920, 0.9503041;
    cs.M_XYZ_to_RGB <<  3.2404542, -1.5371385, -0.4985314,
                      -0.9692660,  1.8760108,  0.0415560,
                       0.0556434, -0.2040259,  1.0572252;
    cs.white_XYZ << 0.95047, 1.0, 1.08883; // D65
    return cs;
}();

const ColorSpace& sRGB(){ return sRGB_cs; }

Eigen::Vector3d XYZ_to_xy(const Eigen::Vector3d& XYZ){
    double sum = XYZ.sum();
    if(sum==0) return Eigen::Vector3d::Zero();
    return {XYZ.x()/sum, XYZ.y()/sum, 0.0};
}

Eigen::Vector3d XYZ_to_RGB(const Eigen::Vector3d& XYZ, const ColorSpace& cs){
    return cs.M_XYZ_to_RGB * XYZ;
}

Eigen::Vector3d RGB_to_XYZ(const Eigen::Vector3d& RGB, const ColorSpace& cs){
    return cs.M_RGB_to_XYZ * RGB;
}

Eigen::Vector3d RGB_to_RGB(const Eigen::Vector3d& RGB,
                           const ColorSpace& src,
                           const ColorSpace& dst){
    Eigen::Vector3d XYZ = src.M_RGB_to_XYZ * RGB;
    return dst.M_XYZ_to_RGB * XYZ;
}

} // namespace col 