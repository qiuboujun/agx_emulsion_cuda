#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025, rgb_to_tc_b, tri2quad, quad2tri
from agx_emulsion.profiles.io import load_profile

def test_fixed_coordinates():
    """Test that the fixed CUDA coordinate calculation should now match Python"""
    
    print("=== Testing Fixed Coordinate Transformation ===")
    
    # Exact input from CUDA debug
    input_rgb = np.array([[[0.176700, 0.036017, 0.007001]]])
    print(f"Input RGB: {input_rgb.flatten()}")
    
    # Step 1: RGB to XYZ (identical between Python/CUDA)
    aces_to_xyz = np.array([
        [0.9525523959, 0.0000000000, 0.0000936786],
        [0.3439664498, 0.7281660966, -0.0721325464],
        [0.0000000000, 0.0000000000, 1.0088251844]
    ])
    
    rgb_flat = input_rgb.reshape(-1, 3)
    xyz = np.dot(rgb_flat, aces_to_xyz.T)
    brightness = np.sum(xyz, axis=1)
    xy = xyz[:, :2] / brightness[:, None]
    
    print(f"XYZ: {xyz[0]}")
    print(f"xy chromaticity: {xy[0]}")
    
    # Step 2: Apply the CORRECTED coordinate transformation
    # This is what the fixed CUDA code should now do:
    triangular = quad2tri(xy)  # xy -> triangular coordinates
    tc_corrected = tri2quad(triangular)  # triangular -> square coordinates
    
    print(f"Triangular coordinates: {triangular[0]}")
    print(f"Corrected tc coordinates: {tc_corrected[0]}")
    
    # Step 3: Compare with what Python actually produces
    tc_python, b = rgb_to_tc_b(input_rgb)
    print(f"Python tc coordinates: {tc_python[0,0]}")
    
    # Calculate difference
    diff = tc_corrected[0] - tc_python[0,0]
    print(f"Difference (corrected - python): {diff}")
    print(f"Difference magnitude: {np.linalg.norm(diff)}")
    
    # Convert to pixel coordinates for 192x192 LUT
    corrected_pixel = tc_corrected[0] * (192 - 1)
    python_pixel = tc_python[0,0] * (192 - 1)
    
    print(f"Corrected pixel coords: {corrected_pixel}")
    print(f"Python pixel coords: {python_pixel}")
    print(f"Pixel difference: {corrected_pixel - python_pixel}")
    
    # Expected result after fix
    print("\n=== Expected CUDA Debug Output After Fix ===")
    print(f"Expected CUDA tc: [{tc_corrected[0,0]:.6f}, {tc_corrected[0,1]:.6f}]")
    print(f"Expected pixel coords: ({corrected_pixel[0]:.6f}, {corrected_pixel[1]:.6f})")
    
    if np.linalg.norm(diff) < 0.001:
        print("✅ COORDINATE TRANSFORMATION SHOULD NOW BE CORRECT!")
    else:
        print("❌ Still a discrepancy - need further investigation")

if __name__ == "__main__":
    test_fixed_coordinates() 