import numpy as np
from scipy.interpolate import interp1d, CubicSpline

np.random.seed(1)
N = 100
x = np.sort(np.random.rand(N) * 10)
y = np.sin(x) + 0.1 * np.random.randn(N)

xq = np.linspace(0, 10, 201)

# Linear and nearest interpolation
f_lin = interp1d(x, y, kind='linear', fill_value='extrapolate')
f_near = interp1d(x, y, kind='nearest', fill_value='extrapolate')

yq_lin  = f_lin(xq)
yq_near = f_near(xq)

# Cubic spline (not-a-knot, SciPy default)
cs = CubicSpline(x, y, bc_type='natural')
yq_spline = cs(xq)


np.savetxt('tests/interp_x.txt', x, fmt='%.17e')
np.savetxt('tests/interp_y.txt', y, fmt='%.17e')
np.savetxt('tests/interp_xq.txt', xq, fmt='%.17e')
np.savetxt('tests/interp_yq_lin.txt', yq_lin, fmt='%.17e')
np.savetxt('tests/interp_yq_near.txt', yq_near, fmt='%.17e')
np.savetxt('tests/interp_yq_spline.txt', yq_spline, fmt='%.17e')
print('Interpolation reference data written to tests/*.txt') 