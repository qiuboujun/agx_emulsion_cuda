#include "measure.hpp"
#include <iostream>
#include <array>

int main()
{
    using gpnp::dvec;

    /* monotone density curves ------------------------------------------- */
    const int N = 12;
    double logE_host[N] = { -3,-2.5,-2,-1.5,-1,-0.5,
                             0, 0.5, 1, 1.5, 2, 2.5 };

    double dens_host[3][N];
    for (int i = 0; i < N; ++i) {
        double x = logE_host[i] + 3.0;
        dens_host[0][i] = 0.20 + 0.15 * x;   // R
        dens_host[1][i] = 0.25 + 0.14 * x;   // G
        dens_host[2][i] = 0.22 + 0.16 * x;   // B
    }
    /* no NaNs now — each curve fully covers ≥0.25 */

    dvec logE_d(logE_host, logE_host + N);
    std::array<dvec,3> dens_d;
    for (int ch = 0; ch < 3; ++ch)
        dens_d[ch] = dvec(dens_host[ch], dens_host[ch] + N);

    auto gamma = measure_gamma(logE_d, dens_d);
    auto slope = measure_slopes_at_exposure(logE_d, dens_d);

    std::cout.setf(std::ios::fixed); std::cout.precision(6);
    std::cout << "gamma  = [" << gamma[0] << ", "
                               << gamma[1] << ", "
                               << gamma[2] << "]\n";
    std::cout << "slope  = [" << slope[0] << ", "
                               << slope[1] << ", "
                               << slope[2] << "]\n";
    return 0;
}

