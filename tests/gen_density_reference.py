import numpy as np, math

# logE grid
logE = np.linspace(-3,3,601)

# default parameters same as python density_curves.py
params_norm = [
    [0,1,2, 0.5,0.5,0.5, 0.3,0.5,0.7],
    [0,1,2, 0.5,0.5,0.5, 0.3,0.5,0.7],
    [0,1,2, 0.5,0.5,0.5, 0.3,0.5,0.7],
]

def norm_cdf(x):
    return 0.5*(1+np.vectorize(lambda t: math.erf(t/np.sqrt(2)))(x))

def curve_norm(logE,p,type='negative'):
    centers=p[:3]; amps=p[3:6]; sigmas=p[6:]
    out=np.zeros_like(logE)
    for c,a,s in zip(centers,amps,sigmas):
        if type=='positive':
            out+=norm_cdf(-(logE-c)/s)*a
        else:
            out+=norm_cdf( (logE-c)/s)*a
    return out

all_curves=[]
for ch in range(3):
    all_curves.append(curve_norm(logE,params_norm[ch],'negative'))
curves=np.stack(all_curves,axis=1)

np.savetxt('tests/dens_loge.txt',logE,fmt='%.8e')
np.savetxt('tests/dens_curve.txt',curves.reshape(-1),fmt='%.8e') 