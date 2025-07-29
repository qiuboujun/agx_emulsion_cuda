#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the ref directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ref'))

from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025, compute_band_pass_filter
from agx_emulsion.profiles.io import load_profile

def test_spectral_direct():
    """Test spectral upsampling directly with the exact ACES values"""
    
    print("=== Direct Spectral Upsampling Test ===")
    
    # Exact ACES AP0 linear values from user
    input_rgb = np.array([[[0.1767578125, 0.0361328125, 0.0068359375]]])
    print(f"Input ACES AP0 Linear RGB: {input_rgb[0,0]}")
    
    # Load film profile to get sensitivity curves
    print("\nLoading kodak_portra_400 film profile...")
    profile = load_profile('kodak_portra_400')
    
    # Extract log sensitivity and convert to linear
    log_sensitivity = profile['data']['log_sensitivity']
    print(f"Log sensitivity shape: {np.array(log_sensitivity).shape}")
    
    # Convert to numpy array and handle NaN values (same as Python implementation)
    log_sens_array = np.array(log_sensitivity)
    sensitivity = 10**log_sens_array
    sensitivity = np.nan_to_num(sensitivity)  # Replace NaN with 0
    
    print(f"Sensitivity R[0]={sensitivity[0,0]:.6f}, G[0]={sensitivity[0,1]:.6f}, B[0]={sensitivity[0,2]:.6f}")
    print(f"Sensitivity R[40]={sensitivity[40,0]:.6f}, G[40]={sensitivity[40,1]:.6f}, B[40]={sensitivity[40,2]:.6f}")
    print(f"Sensitivity R[80]={sensitivity[80,0]:.6f}, G[80]={sensitivity[80,1]:.6f}, B[80]={sensitivity[80,2]:.6f}")
    
    # Apply band pass filter (default from the CUDA implementation)
    filter_uv = [0, 410, 8]
    filter_ir = [0, 675, 15]
    
    if filter_uv[0] > 0 or filter_ir[0] > 0:
        band_pass_filter = compute_band_pass_filter(filter_uv, filter_ir)
        sensitivity *= band_pass_filter[:,None]
        print("\nApplied band pass filter")
        print(f"Filtered sensitivity R[0]={sensitivity[0,0]:.6f}, G[0]={sensitivity[0,1]:.6f}, B[0]={sensitivity[0,2]:.6f}")
        print(f"Filtered sensitivity R[40]={sensitivity[40,0]:.6f}, G[40]={sensitivity[40,1]:.6f}, B[40]={sensitivity[40,2]:.6f}")
        print(f"Filtered sensitivity R[80]={sensitivity[80,0]:.6f}, G[80]={sensitivity[80,1]:.6f}, B[80]={sensitivity[80,2]:.6f}")
    
    # Run spectral upsampling (exact Python reference)
    print("\nRunning spectral upsampling...")
    raw_cmy = rgb_to_raw_hanatos2025(
        input_rgb, 
        sensitivity,
        color_space='ACES2065-1',
        apply_cctf_decoding=False,
        reference_illuminant='D55'
    )
    
    print(f"\nPython Spectral Upsampling Result:")
    print(f"Raw CMY: [{raw_cmy[0,0,0]:.6f}, {raw_cmy[0,0,1]:.6f}, {raw_cmy[0,0,2]:.6f}]")
    
    print(f"\nFor comparison with CUDA debug output:")
    print(f"DEBUG_SPECTRAL: Input RGB = [{input_rgb[0,0,0]:.6f}, {input_rgb[0,0,1]:.6f}, {input_rgb[0,0,2]:.6f}]")
    print(f"DEBUG_SPECTRAL: Final CMY = [{raw_cmy[0,0,0]:.6f}, {raw_cmy[0,0,1]:.6f}, {raw_cmy[0,0,2]:.6f}]")
    
    return raw_cmy[0,0]

if __name__ == "__main__":
    result = test_spectral_direct()
    print(f"\nFinal Python Result: {result}") 