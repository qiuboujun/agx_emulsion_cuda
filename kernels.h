#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#include <string>
#include <utility>
#include <Eigen/Dense>

// Define a type alias for our matrix, which makes the code cleaner.
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Multidimensional Gaussian blur (2-D matrix version).
 *
 * This is the public API declaration for the 2D filter.
 * The implementation is in the .cu file.
 */
Matrix gaussian_filter(const Matrix& input,
                       double sigma,
                       const std::string& mode = "reflect",
                       double truncate = 4.0,
                       int radius = -1);

// Overload for separate sigma values per axis
Matrix gaussian_filter(const Matrix& input,
                       std::pair<double, double> sigma,
                       const std::string& mode = "reflect",
                       double truncate = 4.0,
                       std::pair<int, int> radius = {-1, -1});

/**
 * @brief 1-D Gaussian filter.
 *
 * This is the public API declaration for the 1D filter.
 * The implementation is in the .cu file.
 */
Matrix gaussian_filter1d(const Matrix& input,
                         double sigma,
                         int axis = -1,
                         int order = 0,
                         const std::string& mode = "reflect",
                         double cval = 0.0,
                         double truncate = 4.0,
                         int radius = -1);

#endif // GAUSSIAN_FILTER_H

