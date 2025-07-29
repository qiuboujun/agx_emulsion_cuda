#!/usr/bin/env python3
"""
Test script to verify the fixed Camera LUT values match the Python reference
"""

import numpy as np
import sys
sys.path.insert(0, 'ref')

def test_camera_lut_fixed():
    """Test if the fixed Camera LUT matches Python reference"""
    print("="*80)
    print("CAMERA LUT FIXED TEST")
    print("="*80)
    
    # Test input (same as OFX debug)
    test_rgb = np.array([0.292817, 0.267206, 0.173047], dtype=np.float32)
    print(f"Input RGB: {test_rgb}")
    
    try:
        from agx_emulsion.profiles.io import load_profile
        from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025
        from agx_emulsion.model.color_filters import compute_band_pass_filter
        
        profile = load_profile('kodak_portra_400')
        
        # Get sensitivity (same as fixed LUT generation)
        sensitivity = 10**profile.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)
        
        # Apply band pass filter (same as fixed LUT generation)
        filter_uv = (1, 410, 8)
        filter_ir = (1, 675, 15)
        
        if filter_uv[0] > 0 or filter_ir[0] > 0:
            band_pass_filter = compute_band_pass_filter(filter_uv, filter_ir)
            sensitivity *= band_pass_filter[:, None]
        
        # Call the EXACT same function as the fixed LUT generation
        python_cmy = rgb_to_raw_hanatos2025(
            test_rgb.reshape(1, 1, 3),
            sensitivity,
            color_space='ACES2065-1',
            apply_cctf_decoding=False,
            reference_illuminant=profile.info.reference_illuminant
        )
        python_cmy = python_cmy.flatten()
        
        # Apply exposure adjustment (same as fixed LUT generation)
        exposure_ev = 0.0
        python_cmy *= 2**exposure_ev
        
        print(f"Python CMY output (fixed parameters): {python_cmy}")
        print(f"OFX CMY output (should match now): [1.001534, 0.429265, 1.978867]")
        
        # Compare with OFX (this should be much closer now)
        ofx_cmy = np.array([1.001534, 0.429265, 1.978867])
        diff = np.abs(python_cmy - ofx_cmy)
        print(f"Difference: {diff}")
        print(f"Max difference: {np.max(diff):.6f}")
        print(f"Mean difference: {np.mean(diff):.6f}")
        
        # Check if the difference is acceptable
        max_acceptable_diff = 0.1  # Allow some tolerance for float16 precision
        if np.max(diff) < max_acceptable_diff:
            print(f"✅ SUCCESS: Camera LUT values now match within tolerance ({max_acceptable_diff})")
        else:
            print(f"❌ FAILED: Camera LUT values still don't match (max diff: {np.max(diff):.6f})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_camera_lut_fixed() 