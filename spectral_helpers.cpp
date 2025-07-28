#include "spectral_helpers.hpp"
#include <cmath>
#include <algorithm>

namespace spec {

static constexpr double h = 6.62607015e-34;       // Planck constant (J*s)
static constexpr double c = 2.99792458e8;          // Speed of light (m/s)
static constexpr double k = 1.380649e-23;          // Boltzmann constant (J/K)

void blackbody_spd(double T, const dvec& wav_nm, dvec& out) {
    size_t n = wav_nm.size();
    out.resize(n);
    double maxVal = 0.0;
    for(size_t i=0; i<n; ++i){
        double lambda_m = wav_nm[i] * 1e-9; // nm -> metres
        double L = (2.0*h*c*c) / (std::pow(lambda_m,5)) / (std::exp((h*c)/(lambda_m*k*T)) - 1.0);
        out[i] = L;
        if(L>maxVal) maxVal=L;
    }
    if(maxVal>0){
        for(double &v: out) v/=maxVal; // normalise
    }
}

// --- CIE daylight helper functions (from colour-science) ---
static void daylight_xyz_from_cct(double CCT, double &xD, double &yD){
    // Allen 1931 formula for daylight chromaticity
    if(CCT < 4000.0 || CCT > 25000.0){ CCT=6500.0; }
    double t = CCT;
    double x;
    if(t<=7000.0) x = -4.6070e9/(t*t*t) + 2.9678e6/(t*t) + 0.09911e3/t + 0.244063;
    else          x = -2.0064e9/(t*t*t) + 1.9018e6/(t*t) + 0.24748e3/t + 0.237040;
    double y = -3.0*x*x + 2.870*x - 0.275;
    xD=x; yD=y;
}

void daylight_spd(double CCT, const dvec& wav_nm, dvec& out){
    // CIE standard daylight basis functions S0,S1,S2 sampled every nm 300-830.
    // For brevity include coarse table (5 nm) built into arrays (truncated to 360-830nm).
    static const int N=96; // 360-835 step5 ~ 96 samples
    static const double S0[N] = {
        0.04,3.02,6.00,17.80,29.60,42.45,55.30,56.30,57.30,59.55,61.80,61.65,61.50,65.15,68.80,66.10,63.40,64.60,65.80,80.30,94.80,99.80,104.80,105.35,105.90,101.35,96.80,105.35,113.90,119.75,125.60,125.55,125.50,123.40,121.30,121.30,121.30,117.40,113.50,113.30,113.10,111.95,110.80,108.65,106.50,107.65,108.80,106.50,104.20,104.85,105.50,102.40,99.30,101.35,103.40,101.05,98.70,96.75,94.80,92.20,89.60,85.20,80.80,80.30,79.80,75.35,70.90,71.55,72.20,69.20,66.20,65.00,63.80,61.65,59.50,59.55,59.60,59.70,59.80,59.40,59.00,57.15,55.30,54.60,53.90,51.15,48.40,47.40,46.40,45.50,44.60,43.80,43.00,42.10,41.20,40.00};
    static const double S1[N] = {
        0.02,2.26,4.50,13.45,22.40,32.20,42.00,41.30,40.60,41.10,41.60,39.80,38.00,40.70,43.40,41.10,38.80,36.90,35.00,39.20,43.40,44.85,46.30,45.10,43.90,40.50,37.10,39.20,41.30,41.80,42.30,42.10,41.90,40.50,39.10,36.95,34.80,32.60,30.40,28.60,26.80,25.40,24.00,22.20,20.40,19.00,17.60,16.20,14.80,13.65,12.50,11.70,10.90,10.40,9.90,9.60,9.30,8.80,8.30,7.55,6.80,6.30,5.80,5.45,5.10,4.55,4.00,3.90,3.80,3.55,3.30,3.15,3.00,2.85,2.70,2.55,2.40,2.30,2.20,2.05,1.90,1.80,1.70,1.65,1.60,1.55,1.50,1.45,1.40,1.35,1.30,1.25,1.20,1.15,1.10,1.05};
    static const double S2[N] = {
        0.00,1.00,2.00,3.00,4.00,6.25,8.50,8.15,7.80,7.25,6.70,6.00,5.30,6.25,7.20,6.90,6.60,6.10,5.60,5.95,6.30,6.25,6.20,5.90,5.60,5.10,4.60,4.25,3.90,3.65,3.40,3.30,3.20,3.05,2.90,2.70,2.50,2.35,2.20,2.05,1.90,1.80,1.70,1.60,1.50,1.40,1.30,1.25,1.20,1.15,1.10,1.05,1.00,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.46,0.42,0.40,0.38,0.36,0.34,0.32,0.30,0.27,0.24,0.22,0.20,0.18,0.16,0.14,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04};

    double xD, yD; daylight_xyz_from_cct(CCT,xD,yD);
    double M = (0.0241 + 0.2562*xD - 0.7341*yD)/yD;
    double M1 = (-1.3515 - 1.7703*xD + 5.9114*yD)/M;
    double M2 = (0.0300 - 31.4424*xD + 30.0717*yD)/M;

    // Resample to given wavelength grid using linear interp
    out.resize(wav_nm.size());
    for(size_t i=0;i<wav_nm.size();++i){
        double w=wav_nm[i];
        if(w<360.0||w>825.0){ out[i]=0.0; continue; }
        int idx = int((w-360.0)/5.0);
        if(idx >= N-1){
            double spd = S0[N-1] + M1*S1[N-1] + M2*S2[N-1];
            out[i]=spd;
            continue;
        }
        double t = ((w-360.0) - idx*5.0)/5.0;
        auto lerp=[&](const double* arr){return arr[idx]*(1.0-t)+arr[idx+1]*t;};
        double spd = lerp(S0) + M1*lerp(S1) + M2*lerp(S2);
        out[i]=spd;
    }
    // normalise
    double maxV=*std::max_element(out.begin(),out.end());
    for(double &v:out) v/=maxV;
}

} // namespace spec 