#!/usr/bin/env python3
"""
Debug script to investigate Camera LUT differences
"""

import numpy as np
import sys
sys.path.insert(0, 'ref')

def debug_camera_lut():
    """Debug the Camera LUT stage differences"""
    print("="*80)
    print("CAMERA LUT DEBUG")
    print("="*80)
    
    # Test input (same as OFX debug)
    test_rgb = np.array([0.292817, 0.267206, 0.173047], dtype=np.float32)
    print(f"Input RGB: {test_rgb}")
    
    try:
        from agx_emulsion.profiles.io import load_profile
        from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025
        from agx_emulsion.model.color_filters import compute_band_pass_filter
        
        profile = load_profile('kodak_portra_400')
        
        print(f"\nFilm Profile Info:")
        print(f"  Stock: {profile.info.stock}")
        print(f"  Reference Illuminant: {profile.info.reference_illuminant}")
        print(f"  Log sensitivity shape: {profile.data.log_sensitivity.shape}")
        
        # Check sensitivity values
        sensitivity = 10**profile.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)
        
        print(f"\nSensitivity Info:")
        print(f"  Shape: {sensitivity.shape}")
        print(f"  Range: [{sensitivity.min():.6f}, {sensitivity.max():.6f}]")
        print(f"  First 5 values: {sensitivity[:5].flatten()}")
        print(f"  Last 5 values: {sensitivity[-5:].flatten()}")
        
        # Check band pass filter
        filter_uv = (1, 410, 8)
        filter_ir = (1, 675, 15)
        
        print(f"\nBand Pass Filter:")
        print(f"  UV filter: {filter_uv}")
        print(f"  IR filter: {filter_ir}")
        
        if filter_uv[0] > 0 or filter_ir[0] > 0:
            band_pass_filter = compute_band_pass_filter(filter_uv, filter_ir)
            print(f"  Band pass filter shape: {band_pass_filter.shape}")
            print(f"  Band pass filter range: [{band_pass_filter.min():.6f}, {band_pass_filter.max():.6f}]")
            sensitivity *= band_pass_filter[:, None]
            print(f"  After band pass filter, sensitivity range: [{sensitivity.min():.6f}, {sensitivity.max():.6f}]")
        
        # Test the spectral upsampling
        print(f"\nTesting rgb_to_raw_hanatos2025:")
        print(f"  Input RGB shape: {test_rgb.reshape(1, 1, 3).shape}")
        print(f"  Sensitivity shape: {sensitivity.shape}")
        
        python_cmy = rgb_to_raw_hanatos2025(
            test_rgb.reshape(1, 1, 3),
            sensitivity,
            color_space='ACES2065-1',
            apply_cctf_decoding=False,
            reference_illuminant=profile.info.reference_illuminant
        )
        python_cmy = python_cmy.flatten()
        
        print(f"  Python CMY output: {python_cmy}")
        
        # Compare with OFX
        ofx_cmy = np.array([1.001534, 0.429265, 1.978867])
        print(f"  OFX CMY output: {ofx_cmy}")
        
        diff = np.abs(python_cmy - ofx_cmy)
        print(f"  Difference: {diff}")
        print(f"  Max difference: {np.max(diff):.6f}")
        print(f"  Mean difference: {np.mean(diff):.6f}")
        
        # Check if the issue is with the LUT itself
        print(f"\n" + "="*80)
        print("LUT INVESTIGATION")
        print("="*80)
        
        # The OFX is using a pre-computed LUT from exposure_lut.bin
        # Let's check what parameters the Python function uses internally
        
        print("Python rgb_to_raw_hanatos2025 parameters:")
        print("  - color_space: 'ACES2065-1'")
        print("  - apply_cctf_decoding: False")
        print("  - reference_illuminant: 'D55' (from profile)")
        print("  - sensitivity: from film profile")
        print("  - band_pass_filter: applied")
        
        print("\nOFX Camera LUT parameters:")
        print("  - Uses pre-computed exposure_lut.bin")
        print("  - RGB to xy conversion")
        print("  - Texture lookup")
        
        # Let's test with different parameters to see if we can match OFX
        print(f"\n" + "="*80)
        print("PARAMETER TESTING")
        print("="*80)
        
        # Test without band pass filter
        sensitivity_no_filter = 10**profile.data.log_sensitivity
        sensitivity_no_filter = np.nan_to_num(sensitivity_no_filter)
        
        python_cmy_no_filter = rgb_to_raw_hanatos2025(
            test_rgb.reshape(1, 1, 3),
            sensitivity_no_filter,
            color_space='ACES2065-1',
            apply_cctf_decoding=False,
            reference_illuminant=profile.info.reference_illuminant
        )
        python_cmy_no_filter = python_cmy_no_filter.flatten()
        
        print(f"Without band pass filter:")
        print(f"  Python CMY: {python_cmy_no_filter}")
        diff_no_filter = np.abs(python_cmy_no_filter - ofx_cmy)
        print(f"  Difference: {diff_no_filter}")
        print(f"  Max difference: {np.max(diff_no_filter):.6f}")
        
        # Test with different color space
        python_cmy_srgb = rgb_to_raw_hanatos2025(
            test_rgb.reshape(1, 1, 3),
            sensitivity,
            color_space='sRGB',
            apply_cctf_decoding=False,
            reference_illuminant=profile.info.reference_illuminant
        )
        python_cmy_srgb = python_cmy_srgb.flatten()
        
        print(f"\nWith sRGB color space:")
        print(f"  Python CMY: {python_cmy_srgb}")
        diff_srgb = np.abs(python_cmy_srgb - ofx_cmy)
        print(f"  Difference: {diff_srgb}")
        print(f"  Max difference: {np.max(diff_srgb):.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_camera_lut() 