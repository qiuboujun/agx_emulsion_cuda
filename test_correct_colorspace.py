#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025
from agx_emulsion.profiles.io import load_profile

def test_correct_colorspace():
    """Test Python implementation with ACES2065-1 to match CUDA"""
    
    print("=== Testing with Correct Color Space (ACES2065-1) ===")
    
    # Multi-pixel input to force the LUT path
    input_rgb = np.array([[[0.176700, 0.036017, 0.007001],
                           [0.176700, 0.036017, 0.007001]]])
    print(f"Input RGB: {input_rgb[0,0]}")
    
    # Load sensitivity
    profile = load_profile('kodak_portra_400')
    film = profile['data']
    log_sensitivity = film['log_sensitivity']
    sensitivity = 10.0 ** log_sensitivity
    
    print("Testing different color spaces:")
    
    # Test 1: ITU-R BT.2020 (what we were using by default)
    try:
        cmy_bt2020 = rgb_to_raw_hanatos2025(input_rgb, sensitivity, 
                                           color_space='ITU-R BT.2020',
                                           apply_cctf_decoding=False, 
                                           reference_illuminant='D55')
        print(f"Python (ITU-R BT.2020): CMY = {cmy_bt2020[0,0]}")
    except Exception as e:
        print(f"ITU-R BT.2020 failed: {e}")
    
    # Test 2: ACES2065-1 (what CUDA uses)
    try:
        cmy_aces = rgb_to_raw_hanatos2025(input_rgb, sensitivity,
                                         color_space='ACES2065-1',
                                         apply_cctf_decoding=False,
                                         reference_illuminant='D55')
        print(f"Python (ACES2065-1): CMY = {cmy_aces[0,0]}")
    except Exception as e:
        print(f"ACES2065-1 failed: {e}")
        
    # Test 3: Try without specifying color space (use default)
    try:
        cmy_default = rgb_to_raw_hanatos2025(input_rgb, sensitivity)
        print(f"Python (default): CMY = {cmy_default[0,0]}")
    except Exception as e:
        print(f"Default failed: {e}")
    
    print(f"\nCUDA Result: CMY = [0.003912, 0.416681, 0.567753]")
    
    # Calculate differences
    if 'cmy_aces' in locals():
        diff_aces = cmy_aces[0,0] - np.array([0.003912, 0.416681, 0.567753])
        print(f"Difference (ACES2065-1): {diff_aces}")
        print(f"Ratio (ACES2065-1): {cmy_aces[0,0] / np.array([0.003912, 0.416681, 0.567753])}")
    
    if 'cmy_bt2020' in locals():
        diff_bt2020 = cmy_bt2020[0,0] - np.array([0.003912, 0.416681, 0.567753])
        print(f"Difference (BT.2020): {diff_bt2020}")

if __name__ == "__main__":
    test_correct_colorspace() 