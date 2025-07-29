#!/usr/bin/env python3

import sys
import os
sys.path.append('ref')

import numpy as np
from agx_emulsion.profiles.factory import load_profile
from agx_emulsion.model.stocks import Film
from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025

def test_single_pixel_spectral():
    print("=== Single Pixel Spectral Upsampling Test ===")
    
    # Test with exact ACES input from debug log
    input_rgb = np.array([[[0.1767, 0.036017, 0.007001]]])
    print(f"Input RGB: {input_rgb[0,0]}")
    
    # Load the film profile
    profile = load_profile('kodak_portra_400')
    film = Film(profile)
    
    # Get camera sensitivity (linear, not log)
    camera_sens = film.sensitivity
    print(f"Camera sensitivity shape: {camera_sens.shape}")
    print(f"Camera sensitivity range R: [{np.min(camera_sens[0]):.6f}, {np.max(camera_sens[0]):.6f}]")
    print(f"Camera sensitivity range G: [{np.min(camera_sens[1]):.6f}, {np.max(camera_sens[1]):.6f}]")
    print(f"Camera sensitivity range B: [{np.min(camera_sens[2]):.6f}, {np.max(camera_sens[2]):.6f}]")
    
    # Perform spectral upsampling
    try:
        cmy_result = rgb_to_raw_hanatos2025(
            input_rgb, 
            camera_sens,
            color_space='ACES2065-1',
            apply_cctf_decoding=False,
            reference_illuminant='D55'
        )
        
        print(f"\nPython Spectral Upsampling Result:")
        print(f"CMY: [{cmy_result[0,0,0]:.6f}, {cmy_result[0,0,1]:.6f}, {cmy_result[0,0,2]:.6f}]")
        
        print(f"\nCUDA Debug Log Shows:")
        print(f"CMY: [0.176700, 0.036017, 0.007001]")
        
        print(f"\nDifference (Python - CUDA):")
        cuda_cmy = np.array([0.176700, 0.036017, 0.007001])
        diff = cmy_result[0,0] - cuda_cmy
        print(f"Diff: [{diff[0]:.6f}, {diff[1]:.6f}, {diff[2]:.6f}]")
        
        print(f"\nRatio (Python / CUDA):")
        ratio = cmy_result[0,0] / cuda_cmy
        print(f"Ratio: [{ratio[0]:.2f}x, {ratio[1]:.2f}x, {ratio[2]:.2f}x]")
        
        # Check if CUDA is just passing through input
        print(f"\nInput RGB vs CUDA CMY:")
        print(f"Input:  [{input_rgb[0,0,0]:.6f}, {input_rgb[0,0,1]:.6f}, {input_rgb[0,0,2]:.6f}]")
        print(f"CUDA:   [0.176700, 0.036017, 0.007001]")
        print(f"CUDA seems to be passing through input RGB unchanged!")
        
    except Exception as e:
        print(f"Error in spectral upsampling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_pixel_spectral() 