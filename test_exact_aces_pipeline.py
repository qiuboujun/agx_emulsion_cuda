#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the ref directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ref'))

from agx_emulsion.model.process import AgXPhoto
from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025

def run_exact_aces_test():
    print("Testing exact ACES AP0 linear pipeline matching...")
    
    # Test with exact ACES AP0 linear values from user
    input_rgb = np.array([[[0.1767578125, 0.0361328125, 0.0068359375]]])
    print(f"Input ACES AP0 Linear RGB: {input_rgb[0,0]}")
    
    # Load default film profile (kodak_portra_400)
    params = {
        'negative': {
            'film': 'kodak_portra_400'
        },
        'enlarger': {
            'dichroic_filters': 'edmund_optics'
        },
        'enlarger_illuminant': 'D55',
        'settings': {
            'rgb_to_raw_method': 'hanatos2025'
        },
        'camera': {
            'filter_uv': [0, 410, 8],
            'filter_ir': [0, 675, 15] 
        }
    }
    
    # Create AgXPhoto instance
    agx = AgXPhoto(params)
    
    print(f"\nLoaded film: {agx.negative.info.film_name}")
    print(f"Sensitivity shape: {agx.negative.data.log_sensitivity.shape}")
    
    # Test the spectral upsampling step directly
    print("\n=== Testing Camera LUT / Spectral Upsampling Stage ===")
    
    # Get sensitivity curves
    sensitivity = 10**agx.negative.data.log_sensitivity
    sensitivity = np.nan_to_num(sensitivity) # replace nans with zeros
    
    print(f"Sensitivity R[0]={sensitivity[0,0]:.6f}, G[0]={sensitivity[0,1]:.6f}, B[0]={sensitivity[0,2]:.6f}")
    print(f"Sensitivity R[40]={sensitivity[40,0]:.6f}, G[40]={sensitivity[40,1]:.6f}, B[40]={sensitivity[40,2]:.6f}")
    print(f"Sensitivity R[80]={sensitivity[80,0]:.6f}, G[80]={sensitivity[80,1]:.6f}, B[80]={sensitivity[80,2]:.6f}")
    
    # Apply band pass filter (same as Python)
    from agx_emulsion.utils.spectral_upsampling import compute_band_pass_filter
    if agx.camera.filter_uv[0]>0 or agx.camera.filter_ir[0]>0:
        band_pass_filter = compute_band_pass_filter(agx.camera.filter_uv, agx.camera.filter_ir)
        sensitivity *= band_pass_filter[:,None]
        print("Applied band pass filter")
        print(f"Filtered sensitivity R[0]={sensitivity[0,0]:.6f}, G[0]={sensitivity[0,1]:.6f}, B[0]={sensitivity[0,2]:.6f}")
        print(f"Filtered sensitivity R[40]={sensitivity[40,0]:.6f}, G[40]={sensitivity[40,1]:.6f}, B[40]={sensitivity[40,2]:.6f}")
        print(f"Filtered sensitivity R[80]={sensitivity[80,0]:.6f}, G[80]={sensitivity[80,1]:.6f}, B[80]={sensitivity[80,2]:.6f}")
    
    # Use the exact Python spectral upsampling function
    raw_cmy = rgb_to_raw_hanatos2025(
        input_rgb, 
        sensitivity,
        color_space='ACES2065-1',
        apply_cctf_decoding=False,
        reference_illuminant='D55'
    )
    
    print(f"Python raw CMY: {raw_cmy[0,0]}")
    
    # Compare with expected CUDA values
    print(f"\nExpected CUDA Debug Output:")
    print(f"DEBUG_SPECTRAL: Input RGB = [0.176758, 0.036133, 0.006836]")
    print(f"DEBUG_SPECTRAL: Final CMY = [should match above raw_cmy values]")
    
    return raw_cmy[0,0]

if __name__ == "__main__":
    result = run_exact_aces_test()
    print(f"\nFinal Result: {result}") 