#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025, rgb_to_tc_b
from agx_emulsion.profiles.io import load_profile

def test_cubic_interpolation_fix():
    """Test that cubic interpolation should match Python reference"""
    
    print("=== Testing Cubic Interpolation Fix ===")
    
    # Exact input from DaVinci debug
    input_rgb = np.array([[[0.176700, 0.036017, 0.007001]]])
    print(f"Input RGB: {input_rgb.flatten()}")
    
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
    
    # Get Python's tc coordinates
    tc, b = rgb_to_tc_b(input_rgb, color_space='ACES2065-1', apply_cctf_decoding=False, reference_illuminant='D55')
    print(f"\nPython tc coordinates: {tc.flatten()}")
    print(f"Python brightness: {b.flatten()}")
    
    # Run Python spectral upsampling
    python_result = rgb_to_raw_hanatos2025(
        input_rgb, 
        sensitivity,
        color_space='ACES2065-1',
        apply_cctf_decoding=False,
        reference_illuminant='D55'
    )
    
    print(f"\nPython Reference Result:")
    print(f"CMY: [{python_result[0,0,0]:.6f}, {python_result[0,0,1]:.6f}, {python_result[0,0,2]:.6f}]")
    
    print(f"\nExpected CUDA Result (with cubic interpolation):")
    print(f"CMY: Should be much closer to Python values now!")
    print(f"Previous CUDA (bilinear): [0.004598, 0.416816, 0.662725]")
    print(f"Python (cubic): [{python_result[0,0,0]:.6f}, {python_result[0,0,1]:.6f}, {python_result[0,0,2]:.6f}]")
    
    print(f"\nExpected improvement:")
    print(f"- tc coordinates should be closer to Python: {tc.flatten()}")
    print(f"- Final CMY values should be much closer to Python reference")
    print(f"- The cubic interpolation should provide smoother, more accurate results")

if __name__ == "__main__":
    test_cubic_interpolation_fix() 