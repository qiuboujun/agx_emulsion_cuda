import numpy as np

def parametric_density_curves_model(log_exposure, gamma, log_exposure_0, density_max, toe_size, shoulder_size):
    density_curves = np.zeros((np.size(log_exposure), 3))
    for i, g, loge0, dmax, ts, ss in zip(np.arange(3),
                                            gamma, log_exposure_0, density_max, toe_size, shoulder_size):
        density_curves[:,i] = (  
              g*ts * np.log10(1 + 10**( (log_exposure - loge0         )/ts ))
            - g*ss * np.log10(1 + 10**( (log_exposure - loge0 - dmax/g)/ss ))
        )
    return density_curves

logE=np.linspace(-3,3,601)

gamma=[0.65,0.7,0.68]
loge0=[-1.0,-1.0,-1.0]
dmax=[2.0,2.0,2.0]
toe=[0.2,0.2,0.2]
shoulder=[0.3,0.3,0.3]

curves=parametric_density_curves_model(logE,gamma,loge0,dmax,toe,shoulder)

np.savetxt('tests/para_loge.txt',logE,fmt='%.8e')
np.savetxt('tests/para_curve.txt',curves.reshape(-1),fmt='%.8e') 