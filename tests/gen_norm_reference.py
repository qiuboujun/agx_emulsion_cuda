import numpy as np

# Generate input values
x = np.linspace(-5, 5, 201)

# Reference computations using NumPy
pdf = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
cdf = 0.5 * (1 + np.erf(x / np.sqrt(2)))

np.savetxt('tests/x_vals.txt', x, fmt='%.17e')
np.savetxt('tests/pdf_ref.txt', pdf, fmt='%.17e')
np.savetxt('tests/cdf_ref.txt', cdf, fmt='%.17e')
print('Reference data written to tests/*.txt') 