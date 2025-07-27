import numpy as np
import colour

np.random.seed(2)
M = 50
rgb = np.random.rand(M, 3)

xyz = colour.RGB_to_XYZ(rgb, colourspace='sRGB', apply_cctf_decoding=False)
xy  = colour.XYZ_to_xy(xyz)
# pad with zero to get (x, y, 0) per sample for easier 3-vector comparison
xy_pad = np.column_stack([xy, np.zeros(len(xy))])

np.savetxt('tests/color_rgb.txt', rgb.reshape(-1), fmt='%.17e')
np.savetxt('tests/color_xyz.txt', xyz.reshape(-1), fmt='%.17e')
np.savetxt('tests/color_xy.txt', xy_pad.reshape(-1), fmt='%.17e')
print('Colour reference data written to tests/*.txt') 