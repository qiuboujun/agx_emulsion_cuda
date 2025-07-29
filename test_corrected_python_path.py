#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025, rgb_to_tc_b
from agx_emulsion.profiles.io import load_profile

def test_corrected_python_path():
    """Test using the correct Python execution path (multi-pixel) to match CUDA"""
    
    print("=== Testing Corrected Python Path (Multi-Pixel) ===")
    
    # Create multi-pixel input to force the pre-multiplied LUT path
    # Input shape: (1, 2, 3) - this forces shape[1] > 1, so it uses the LUT path
    input_rgb = np.array([[[0.176700, 0.036017, 0.007001],
                           [0.176700, 0.036017, 0.007001]]])  # Duplicate pixel to force multi-pixel path
    print(f"Input RGB shape: {input_rgb.shape}")
    print(f"First pixel: {input_rgb[0,0]}")
    
    # Load film profile
    profile = load_profile('kodak_portra_400')
    film = profile['data']
    
    # Get sensitivity curves (linear, not log)
    log_sensitivity = film['log_sensitivity']
    sensitivity = 10**np.array(log_sensitivity)
    sensitivity = np.nan_to_num(sensitivity)  # Replace NaN with 0
    
    print(f"Sensitivity shape: {sensitivity.shape}")
    print(f"Sensitivity R[0]={sensitivity[0,0]:.6f}, G[0]={sensitivity[0,1]:.6f}, B[0]={sensitivity[0,2]:.6f}")
    print(f"Sensitivity R[40]={sensitivity[40,0]:.6f}, G[40]={sensitivity[40,1]:.6f}, B[40]={sensitivity[40,2]:.6f}")
    
    # Get Python's tc coordinates (for first pixel)
    tc, b = rgb_to_tc_b(input_rgb[:,:1,:], color_space='ACES2065-1', apply_cctf_decoding=False, reference_illuminant='D55')
    print(f"\nPython tc coordinates: {tc.flatten()}")
    print(f"Python brightness: {b.flatten()}")
    
    # Run Python spectral upsampling (multi-pixel path)
    python_result = rgb_to_raw_hanatos2025(
        input_rgb, 
        sensitivity,
        color_space='ACES2065-1',
        apply_cctf_decoding=False,
        reference_illuminant='D55'
    )
    
    print(f"\nPython Multi-Pixel Path Result:")
    print(f"CMY: [{python_result[0,0,0]:.6f}, {python_result[0,0,1]:.6f}, {python_result[0,0,2]:.6f}]")
    
    # Also test single pixel path for comparison
    single_pixel_input = input_rgb[:,:1,:]  # Just first pixel
    single_pixel_result = rgb_to_raw_hanatos2025(
        single_pixel_input, 
        sensitivity,
        color_space='ACES2065-1',
        apply_cctf_decoding=False,
        reference_illuminant='D55'
    )
    
    print(f"\nPython Single-Pixel Path Result:")
    print(f"CMY: [{single_pixel_result[0,0,0]:.6f}, {single_pixel_result[0,0,1]:.6f}, {single_pixel_result[0,0,2]:.6f}]")
    
    print(f"\nDifference (Multi-Pixel - Single-Pixel):")
    diff = python_result[0,0] - single_pixel_result[0,0]
    print(f"Diff: [{diff[0]:.6f}, {diff[1]:.6f}, {diff[2]:.6f}]")
    
    print(f"\nExpected CUDA Result:")
    print(f"CUDA should match the Multi-Pixel path: [{python_result[0,0,0]:.6f}, {python_result[0,0,1]:.6f}, {python_result[0,0,2]:.6f}]")
    
    return python_result[0,0]  # Return the multi-pixel result

if __name__ == "__main__":
    result = test_corrected_python_path() 