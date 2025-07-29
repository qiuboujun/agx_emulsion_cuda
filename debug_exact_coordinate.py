#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.utils.spectral_upsampling import HANATOS2025_SPECTRA_LUT, rgb_to_tc_b, rgb_to_spectrum
import scipy.interpolate

def debug_exact_coordinate():
    """Debug exact spectral values at Python-computed coordinates"""
    
    print("=== Debug Exact Coordinate Match ===")
    
    # Exact input from test
    input_rgb = np.array([[[0.176700, 0.036017, 0.007001]]])
    
    # Get Python's tc coordinates  
    tc, b = rgb_to_tc_b(input_rgb, color_space='ACES2065-1', apply_cctf_decoding=False, reference_illuminant='D55')
    print(f"Python tc: {tc.flatten()}")
    print(f"Python brightness: {b.flatten()}")
    
    # Get spectrum using Python's rgb_to_spectrum
    spectrum_python = rgb_to_spectrum(input_rgb, color_space='ACES2065-1', apply_cctf_decoding=False, reference_illuminant='D55')
    print(f"Python spectrum shape: {spectrum_python.shape}")
    print(f"Python spectrum[0]: {spectrum_python[0]:.6f}, spectrum[40]: {spectrum_python[40]:.6f}, spectrum[80]: {spectrum_python[80]:.6f}")
    
    # Now manually fetch from the LUT at the same tc coordinates
    lut_size = 192
    tc_flat = tc.flatten()
    
    # Manual bilinear interpolation like CUDA should do
    x_float = tc_flat[0] * (lut_size - 1)  
    y_float = tc_flat[1] * (lut_size - 1)
    
    x = int(x_float)
    y = int(y_float)
    fx = x_float - x
    fy = y_float - y
    
    x1 = min(x + 1, lut_size - 1)
    y1 = min(y + 1, lut_size - 1)
    
    print(f"Manual interpolation:")
    print(f"  tc coordinates: ({tc_flat[0]:.6f}, {tc_flat[1]:.6f})")
    print(f"  float indices: ({x_float:.3f}, {y_float:.3f})")
    print(f"  integer indices: ({x}, {y}) to ({x1}, {y1})")
    print(f"  fractional parts: ({fx:.3f}, {fy:.3f})")
    
    # Get 4 corner spectra
    s00 = HANATOS2025_SPECTRA_LUT[y, x, :]
    s10 = HANATOS2025_SPECTRA_LUT[y, x1, :]  
    s01 = HANATOS2025_SPECTRA_LUT[y1, x, :]
    s11 = HANATOS2025_SPECTRA_LUT[y1, x1, :]
    
    print(f"Corner spectra[0]: s00={s00[0]:.6f}, s10={s10[0]:.6f}, s01={s01[0]:.6f}, s11={s11[0]:.6f}")
    
    # Manual bilinear interpolation  
    s0 = s00 + (s10 - s00) * fx
    s1 = s01 + (s11 - s01) * fx
    spectrum_manual = s0 + (s1 - s0) * fy
    
    print(f"Manual bilinear result:")
    print(f"  spectrum[0]: {spectrum_manual[0]:.6f}, spectrum[40]: {spectrum_manual[40]:.6f}, spectrum[80]: {spectrum_manual[80]:.6f}")
    
    # Compare with Python's rgb_to_spectrum
    print(f"\nComparison:")
    print(f"  Python rgb_to_spectrum[0]: {spectrum_python[0]:.6f}")
    print(f"  Manual bilinear[0]: {spectrum_manual[0]:.6f}")
    print(f"  Difference: {abs(spectrum_python[0] - spectrum_manual[0]):.6f}")
    
    # Scale by brightness like Python does
    spectrum_scaled = spectrum_manual * b.flatten()[0]
    print(f"\nAfter brightness scaling:")
    print(f"  Python brightness: {b.flatten()[0]:.6f}")
    print(f"  Scaled spectrum[0]: {spectrum_scaled[0]:.6f}")
    print(f"  Expected from Python debug: ~0.00499 (but this would give ~{spectrum_manual[0] * b.flatten()[0]:.6f})")

if __name__ == "__main__":
    debug_exact_coordinate() 