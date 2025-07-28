#include "couplers.hpp"
#include <algorithm>
#include <numeric>

namespace cp {

static double gauss1d(int dist, double sigma){
    double s2=sigma*sigma;
    return std::exp(-0.5*dist*dist/s2);
}

Matrix3 compute_dir_couplers_matrix(const std::array<double,3>& amount_rgb, double layer_diffusion){
    Matrix3 mat{};
    for(int i=0;i<3;i++){
        // compute gaussian weights across columns j for row i
        double sum=0.0;
        for(int j=0;j<3;j++){
            int dist=j-i;
            double w=gauss1d(dist,layer_diffusion);
            mat[i][j]=w;
            sum+=w;
        }
        // normalize row
        for(int j=0;j<3;j++) mat[i][j]= (mat[i][j]/sum) * amount_rgb[i];
    }
    return mat;
}

static double interp_linear(const dvec& x,const dvec& y,double xv){
    if(xv<=x.front()) return y.front();
    if(xv>=x.back()) return y.back();
    auto it=std::lower_bound(x.begin(),x.end(),xv);
    size_t idx=std::distance(x.begin(),it);
    double x0=x[idx-1], x1=x[idx];
    double t=(xv-x0)/(x1-x0);
    return y[idx-1]*(1-t)+y[idx]*t;
}

void compute_density_curves_before_dir_couplers(const std::vector<std::array<double,3>>& density_curves,
                                                const dvec& log_exposure,
                                                const Matrix3& dir_couplers_matrix,
                                                double high_exposure_couplers_shift,
                                                std::vector<std::array<double,3>>& out){
    size_t n=log_exposure.size();
    out.resize(n);

    // compute d_max per channel
    std::array<double,3> dmax={0,0,0};
    for(const auto& row: density_curves){
        for(int c=0;c<3;c++) dmax[c]=std::max(dmax[c],row[c]);
    }

    // for each row, compute dc_norm_shift and couplers_amount
    for(size_t i=0;i<n;i++){
        std::array<double,3> norm_shift;
        for(int c=0;c<3;c++){
            double v=density_curves[i][c]/dmax[c];
            v+=high_exposure_couplers_shift*v*v;
            norm_shift[c]=v;
        }
        std::array<double,3> couplers_amount={0,0,0};
        // matrix multiply row vector (norm_shift) with dir_couplers_matrix
        for(int j=0;j<3;j++){
            double sum=0.0;
            for(int k=0;k<3;k++) sum+=norm_shift[k]*dir_couplers_matrix[k][j];
            couplers_amount[j]=sum;
        }
        // compute x0 = log_exposure[i] - couplers_amount[channel]
        for(int c=0;c<3;c++){
            double target_x = log_exposure[i];
            double shifted_x = target_x - couplers_amount[c];
            // interpolate density_curves[:,c] at x=shifted_x
            // We need x-vector (log_exposure) and y-vector for channel
            // gather channel column y vector once ( ineff ) but fine for n small
        }
    }
    // Precompute channel x0 arrays
    dvec x0_R(n), x0_G(n), x0_B(n);
    dvec yR(n),yG(n),yB(n);
    for(size_t i=0;i<n;i++){
        // recompute norm_shift and couplers amount
        std::array<double,3> norm_shift;
        for(int c=0;c<3;c++){
            double v=density_curves[i][c]/dmax[c];
            v+=high_exposure_couplers_shift*v*v;
            norm_shift[c]=v;
        }
        std::array<double,3> couplers_amount;
        for(int j=0;j<3;j++){
            double sum=0.0;for(int k=0;k<3;k++) sum+=norm_shift[k]*dir_couplers_matrix[k][j];
            couplers_amount[j]=sum;
        }
        double le=log_exposure[i];
        x0_R[i]=le - couplers_amount[0];
        x0_G[i]=le - couplers_amount[1];
        x0_B[i]=le - couplers_amount[2];
        yR[i]=density_curves[i][0];
        yG[i]=density_curves[i][1];
        yB[i]=density_curves[i][2];
    }
    // Now perform interpolation for each target le
    for(size_t i=0;i<n;i++){
        double le=log_exposure[i];
        out[i][0]=interp_linear(x0_R,yR, le);
        out[i][1]=interp_linear(x0_G,yG, le);
        out[i][2]=interp_linear(x0_B,yB, le);
    }
}

void compute_exposure_correction_dir_couplers(const std::vector<std::array<double,3>>& log_raw,
                                              const std::vector<std::array<double,3>>& density_cmy,
                                              const std::array<double,3>& density_max,
                                              const Matrix3& dir_couplers_matrix,
                                              double high_exposure_shift,
                                              std::vector<std::array<double,3>>& log_raw_corrected){
    size_t n=log_raw.size();
    log_raw_corrected.resize(n);
    for(size_t i=0;i<n;i++){
        // norm density
        std::array<double,3> norm_density;
        for(int c=0;c<3;c++){
            double v=density_cmy[i][c]/density_max[c];
            v+=high_exposure_shift*v*v;
            norm_density[c]=v;
        }
        // compute correction per channel via mat multiply
        std::array<double,3> corr={0,0,0};
        for(int j=0;j<3;j++){
            double sum=0.0; for(int k=0;k<3;k++) sum+=norm_density[k]*dir_couplers_matrix[k][j];
            corr[j]=sum;
        }
        // subtract
        for(int c=0;c<3;c++){
            log_raw_corrected[i][c]=log_raw[i][c]-corr[c];
        }
    }
}

} 