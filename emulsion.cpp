#include "emulsion.hpp"

namespace emu {
FilmSimple::FilmSimple(const interp::dvec& log_exposure,
                       const std::vector<std::array<double,3>>& density_curves,
                       double gamma_factor,
                       bool use_couplers,
                       const cp::Matrix3& dir_matrix,
                       double high_exposure_shift)
: m_logExp(log_exposure), m_densityCurves(density_curves), m_useCouplers(use_couplers), m_dirM(dir_matrix), m_shift(high_exposure_shift) {
    m_gamma = {gamma_factor,gamma_factor,gamma_factor};
}

void FilmSimple::develop(const interp::dvec& log_raw,
                         std::vector<std::array<double,3>>& density_out) const{
    std::vector<std::array<double,3>> density_tmp;
    interp::exposure_to_density(log_raw,m_densityCurves,m_logExp,m_gamma,density_tmp);
    if(!m_useCouplers){
        density_out=density_tmp;return;
    }
    // single-pass correction
    std::array<double,3> dmax={0,0,0};
    for(const auto& row: m_densityCurves){for(int c=0;c<3;c++) dmax[c]=std::max(dmax[c],row[c]);}
    // Prepare per-pixel per-channel input vector
    std::vector<std::array<double,3>> log_raw_vec(log_raw.size());
    for(size_t i=0;i<log_raw.size();i++) for(int c=0;c<3;c++) log_raw_vec[i][c]=log_raw[i];
    std::vector<std::array<double,3>> log_raw_corr;
    cp::compute_exposure_correction_dir_couplers(log_raw_vec,density_tmp,dmax,m_dirM,m_shift,log_raw_corr);
    // re-interpolate channel-wise
    size_t n=log_raw.size();
    density_out.resize(n);
    for(int c=0;c<3;c++){
        interp::dvec log_chan(n);
        for(size_t i=0;i<n;i++) log_chan[i]=log_raw_corr[i][c];
        std::vector<std::array<double,3>> dens_tmp;
        std::array<double,3> gamma_ch = {m_gamma[c],m_gamma[c],m_gamma[c]};
        interp::exposure_to_density(log_chan,m_densityCurves,m_logExp,{m_gamma[c],m_gamma[c],m_gamma[c]},dens_tmp);
        for(size_t i=0;i<n;i++) density_out[i][c]=dens_tmp[i][0]; // dens_tmp saves same value for all channels
    }
}
} 