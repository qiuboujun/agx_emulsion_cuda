#pragma once
#include <Eigen/Dense>
#include <string>

namespace col {

struct ColorSpace {
    Eigen::Matrix3d M_RGB_to_XYZ;
    Eigen::Matrix3d M_XYZ_to_RGB;
    Eigen::Vector3d white_XYZ;   // normalized Y=1
};

// Predefined sRGB (D65)
const ColorSpace& sRGB();

// Compute xy chromaticity from XYZ (returns x,y,0)
Eigen::Vector3d XYZ_to_xy(const Eigen::Vector3d& XYZ);

// Convert XYZ to RGB in given colour-space (linear, no gamma)
Eigen::Vector3d XYZ_to_RGB(const Eigen::Vector3d& XYZ, const ColorSpace& cs);

// Convert RGB (linear) to XYZ in given colour-space
Eigen::Vector3d RGB_to_XYZ(const Eigen::Vector3d& RGB, const ColorSpace& cs);

// Convert RGB from one colour-space to another (linear)
Eigen::Vector3d RGB_to_RGB(const Eigen::Vector3d& RGB,
                           const ColorSpace& src,
                           const ColorSpace& dst);

} // namespace col 