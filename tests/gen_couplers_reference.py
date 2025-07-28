import numpy as np
import math

# helper for Gaussian
def compute_dir_couplers_matrix(amount_rgb, layer_diffusion):
    M = np.eye(3)
    rows=[]
    for i in range(3):
        w=[]
        for j in range(3):
            dist=j-i
            w.append(math.exp(-0.5*dist*dist/(layer_diffusion**2)))
        w=np.array(w)
        w=w/np.sum(w)*amount_rgb[i]
        rows.append(w)
    return np.vstack(rows)

def compute_density_curves_before_dir_couplers(density_curves, log_exposure, dir_couplers_matrix, high_exposure_couplers_shift=0.0):
    d_max = np.nanmax(density_curves, axis=0)
    dc_norm = density_curves/d_max
    dc_norm_shift = dc_norm + high_exposure_couplers_shift*dc_norm**2
    couplers_amount_curves = np.dot(dc_norm_shift, dir_couplers_matrix)
    x0 = log_exposure[:,None] - couplers_amount_curves
    density_curves_corrected = np.zeros_like(density_curves)
    for i in range(3):
        density_curves_corrected[:,i] = np.interp(log_exposure, x0[:,i], density_curves[:,i])
    return density_curves_corrected

# Load previous density curves and logE
logE = np.loadtxt('tests/dens_loge.txt')
curves_flat = np.loadtxt('tests/dens_curve.txt')
curves = curves_flat.reshape(-1,3)

amount_rgb=[0.7,0.7,0.5]
layer_diff=1
shift=0.0
M = compute_dir_couplers_matrix(amount_rgb, layer_diff)
np.savetxt('tests/dir_matrix.txt', M.reshape(-1), fmt='%.8e')

corr = compute_density_curves_before_dir_couplers(curves, logE, M, shift)
np.savetxt('tests/corr_curve.txt', corr.reshape(-1), fmt='%.8e') 