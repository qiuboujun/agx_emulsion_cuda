#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025, rgb_to_tc_b
from agx_emulsion.profiles.io import load_profile

def test_coordinate_precision():
    """Test exact coordinate calculations step by step"""
    
    print("=== Coordinate Precision Debug ===")
    
    # Exact input from CUDA debug
    input_rgb = np.array([[[0.176700, 0.036017, 0.007001]]])
    print(f"Input RGB: {input_rgb.flatten()}")
    
    # Load film profile for sensitivity
    profile = load_profile('kodak_portra_400')
    film = profile['data']
    log_sensitivity = film['log_sensitivity']
    sensitivity = 10.0 ** log_sensitivity  # Convert to linear
    
    # Step 1: RGB to XYZ (ACES2065-1 matrix from the CUDA code)
    # From DynamicSpectralUpsampling.cu lines 26-35
    aces_to_xyz = np.array([
        [0.9525523959, 0.0000000000, 0.0000936786],
        [0.3439664498, 0.7281660966, -0.0721325464],
        [0.0000000000, 0.0000000000, 1.0088251844]
    ])
    
    rgb_flat = input_rgb.reshape(-1, 3)
    xyz = np.dot(rgb_flat, aces_to_xyz.T)
    print(f"Python XYZ: {xyz[0]}")
    print(f"CUDA XYZ: [0.168316, 0.086500, 0.007062]")
    
    brightness = np.sum(xyz, axis=1)
    print(f"Python brightness: {brightness[0]}")
    print(f"CUDA brightness: 0.261879")
    
    # Step 2: XYZ to xy chromaticity
    xy = xyz[:, :2] / brightness[:, None]
    print(f"Python xy: {xy[0]}")
    print(f"CUDA xy: [0.642725, 0.330307]")
    
    # Step 3: xy to tc using tri2quad
    tc_raw, b = rgb_to_tc_b(input_rgb)
    
    print(f"Python tc: {tc_raw[0,0]}")
    print(f"CUDA tc: [0.127645, 0.924517]")
    
    # Calculate pixel coordinates for 192x192 LUT
    python_pixel = tc_raw[0,0] * (192 - 1)
    cuda_pixel = np.array([0.127645, 0.924517]) * (192 - 1)
    
    print(f"Python pixel coords: {python_pixel}")
    print(f"CUDA pixel coords: {cuda_pixel}")
    print(f"Pixel difference: {python_pixel - cuda_pixel}")
    
    # Show what this maps to in a 192x192 grid
    print(f"Python grid index: ({int(python_pixel[0])}, {int(python_pixel[1])})")
    print(f"CUDA grid index: ({int(cuda_pixel[0])}, {int(cuda_pixel[1])})")

if __name__ == "__main__":
    test_coordinate_precision() 