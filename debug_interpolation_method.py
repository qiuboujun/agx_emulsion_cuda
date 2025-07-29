#!/usr/bin/env python3

import numpy as np
import sys
import os
import scipy.interpolate

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.utils.spectral_upsampling import HANATOS2025_SPECTRA_LUT, rgb_to_tc_b

def debug_interpolation_method():
    """Compare scipy RegularGridInterpolator vs manual bilinear"""
    
    print("=== Debug Interpolation Method ===")
    
    # Exact input from test
    input_rgb = np.array([[[0.176700, 0.036017, 0.007001]]])
    
    # Get Python's tc coordinates  
    tc, b = rgb_to_tc_b(input_rgb, color_space='ACES2065-1', apply_cctf_decoding=False, reference_illuminant='D55')
    tc_flat = tc.flatten()
    print(f"tc coordinates: {tc_flat}")
    print(f"brightness: {b.flatten()[0]:.6f}")
    
    # Method 1: scipy.interpolate.RegularGridInterpolator (Python's method)
    v = np.linspace(0, 1, HANATOS2025_SPECTRA_LUT.shape[0])  # [0, 1] with 192 points
    interpolator = scipy.interpolate.RegularGridInterpolator((v, v), HANATOS2025_SPECTRA_LUT)
    spectrum_scipy = interpolator(tc)
    spectrum_scipy = spectrum_scipy.flatten()  # Flatten to 1D
    spectrum_scipy_scaled = spectrum_scipy * b.flatten()[0]
    
    print(f"\nMethod 1 - scipy RegularGridInterpolator:")
    print(f"  Raw spectrum[0]: {spectrum_scipy[0]:.6f}, spectrum[40]: {spectrum_scipy[40]:.6f}")
    print(f"  Scaled spectrum[0]: {spectrum_scipy_scaled[0]:.6f}")
    
    # Method 2: Manual bilinear interpolation (our CUDA method)
    lut_size = 192
    x_float = tc_flat[0] * (lut_size - 1)  
    y_float = tc_flat[1] * (lut_size - 1)
    
    x = int(x_float)
    y = int(y_float)
    fx = x_float - x
    fy = y_float - y
    
    x1 = min(x + 1, lut_size - 1)
    y1 = min(y + 1, lut_size - 1)
    
    # Get 4 corner spectra
    s00 = HANATOS2025_SPECTRA_LUT[y, x, :]
    s10 = HANATOS2025_SPECTRA_LUT[y, x1, :]  
    s01 = HANATOS2025_SPECTRA_LUT[y1, x, :]
    s11 = HANATOS2025_SPECTRA_LUT[y1, x1, :]
    
    # Manual bilinear interpolation  
    s0 = s00 + (s10 - s00) * fx
    s1 = s01 + (s11 - s01) * fx
    spectrum_manual = s0 + (s1 - s0) * fy
    spectrum_manual_scaled = spectrum_manual * b.flatten()[0]
    
    print(f"\nMethod 2 - Manual bilinear interpolation:")
    print(f"  Raw spectrum[0]: {spectrum_manual[0]:.6f}, spectrum[40]: {spectrum_manual[40]:.6f}")
    print(f"  Scaled spectrum[0]: {spectrum_manual_scaled[0]:.6f}")
    
    # Comparison
    print(f"\nComparison:")
    print(f"  Ratio (scipy/manual): {spectrum_scipy[0] / spectrum_manual[0]:.2f}")
    print(f"  CUDA debug showed: {0.000546:.6f} (matches manual bilinear)")
    print(f"  Python debug showed: {0.00499:.6f} (matches scipy interpolator)")
    
    # Test different interpolation methods
    print(f"\nTesting different scipy interpolation methods:")
    for method in ['linear', 'nearest', 'cubic']:
        try:
            interp = scipy.interpolate.RegularGridInterpolator((v, v), HANATOS2025_SPECTRA_LUT, method=method)
            result = interp(tc)
            result = result.flatten()
            scaled = result * b.flatten()[0]
            print(f"  {method}: raw={result[0]:.6f}, scaled={scaled[0]:.6f}")
        except Exception as e:
            print(f"  {method}: Error - {e}")

if __name__ == "__main__":
    debug_interpolation_method() 