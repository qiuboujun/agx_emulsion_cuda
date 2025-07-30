#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.utils.spectral_upsampling import rgb_to_tc_b, tri2quad
import colour

def test_step_by_step():
    """Trace step by step what rgb_to_tc_b does vs manual calculation"""
    
    print("=== Step-by-Step Debug of rgb_to_tc_b ===")
    
    # Input
    input_rgb = np.array([[[0.176700, 0.036017, 0.007001]]])
    print(f"Input RGB: {input_rgb.flatten()}")
    
    # What rgb_to_tc_b does (line 124-134):
    print("\n--- Python rgb_to_tc_b Steps ---")
    
    # Step 1: Get illuminant
    from agx_emulsion.utils.spectral_upsampling import illuminant_to_xy
    illu_xy = illuminant_to_xy('D55')
    print(f"Illuminant D55 xy: {illu_xy}")
    
    # Step 2: RGB to XYZ using colour library
    xyz_colour = colour.RGB_to_XYZ(input_rgb, colourspace='ITU-R BT.2020',
                                   apply_cctf_decoding=False,
                                   illuminant=illu_xy,
                                   chromatic_adaptation_transform='CAT02')
    print(f"Python colour.RGB_to_XYZ result: {xyz_colour.flatten()}")
    
    # Step 3: Brightness and xy
    b_colour = np.sum(xyz_colour, axis=-1)
    xy_colour = xyz_colour[...,0:2] / np.fmax(b_colour[...,None], 1e-10)
    xy_colour = np.clip(xy_colour, 0, 1)
    print(f"Python b (brightness): {b_colour.flatten()}")
    print(f"Python xy: {xy_colour.flatten()}")
    
    # Step 4: tri2quad
    tc_colour = tri2quad(xy_colour)
    print(f"Python tc from colour: {tc_colour.flatten()}")
    
    # Compare with rgb_to_tc_b result
    tc_direct, b_direct = rgb_to_tc_b(input_rgb)
    print(f"Direct rgb_to_tc_b tc: {tc_direct.flatten()}")
    print(f"Direct rgb_to_tc_b b: {b_direct.flatten()}")
    
    print("\n--- Manual ACES2065-1 Calculation ---")
    
    # Our manual calculation using ACES2065-1 matrix
    aces_to_xyz = np.array([
        [0.9525523959, 0.0000000000, 0.0000936786],
        [0.3439664498, 0.7281660966, -0.0721325464],
        [0.0000000000, 0.0000000000, 1.0088251844]
    ])
    
    rgb_flat = input_rgb.reshape(-1, 3)
    xyz_manual = np.dot(rgb_flat, aces_to_xyz.T)
    b_manual = np.sum(xyz_manual, axis=1)
    xy_manual = xyz_manual[:, :2] / b_manual[:, None]
    tc_manual = tri2quad(xy_manual)
    
    print(f"Manual XYZ: {xyz_manual.flatten()}")
    print(f"Manual b: {b_manual.flatten()}")
    print(f"Manual xy: {xy_manual.flatten()}")
    print(f"Manual tc: {tc_manual.flatten()}")
    
    print("\n--- Comparison ---")
    print(f"XYZ difference: {xyz_colour.flatten() - xyz_manual.flatten()}")
    print(f"tc difference: {tc_direct.flatten() - tc_manual.flatten()}")
    
    # Key insight: Are we using different color spaces or illuminants?
    print(f"\nITU-R BT.2020 vs ACES2065-1 difference in tc: {tc_colour.flatten() - tc_manual.flatten()}")

if __name__ == "__main__":
    test_step_by_step() 