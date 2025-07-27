import numpy as np
from scipy.ndimage import gaussian_filter

np.random.seed(0)
img = np.random.rand(10, 10)

sigma = 1.2
out = gaussian_filter(img, sigma=sigma, mode='reflect')

np.savetxt('tests/gaussian_input.txt', img.reshape(-1), fmt='%.17e')
np.savetxt('tests/gaussian_ref.txt', out.reshape(-1), fmt='%.17e')
print('Gaussian reference data written to tests/*.txt') 