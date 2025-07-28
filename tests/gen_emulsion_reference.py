import numpy as np

logE=np.linspace(-3,3,601)
curves_flat=np.loadtxt('tests/dens_curve.txt')
curves=curves_flat.reshape(-1,3)

# choose log_raw grid offset by 0.123
log_raw=logE+0.123

gamma=1.2

out=np.zeros((len(log_raw),3))
for ch in range(3):
    out[:,ch]=np.interp(log_raw,logE/gamma,curves[:,ch])

np.savetxt('tests/emul_lograw.txt',log_raw,fmt='%.8e')
np.savetxt('tests/emul_density.txt',out.reshape(-1),fmt='%.8e') 