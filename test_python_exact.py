#!/usr/bin/env python3
"""
Test to see what Python reference produces with exact LUT generation parameters
"""

import numpy as np
import sys
sys.path.insert(0, 'ref')

def test_python_exact():
    """Test Python reference with exact LUT generation parameters"""
    print("="*80)
    print("PYTHON REFERENCE WITH EXACT LUT PARAMETERS")
    print("="*80)
    
    # Test input (same as OFX debug)
    test_rgb = np.array([0.292817, 0.267206, 0.173047], dtype=np.float32)
    print(f"Input RGB: {test_rgb}")
    
    try:
        from agx_emulsion.profiles.io import load_profile
        from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025
        from agx_emulsion.model.color_filters import compute_band_pass_filter
        
        profile = load_profile('kodak_portra_400')
        
        # Get sensitivity (same as LUT generation)
        sensitivity = 10**profile.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)
        
        # Apply band pass filter (same as LUT generation)
        filter_uv = (1, 410, 8)
        filter_ir = (1, 675, 15)
        
        if filter_uv[0] > 0 or filter_ir[0] > 0:
            band_pass_filter = compute_band_pass_filter(filter_uv, filter_ir)
            sensitivity *= band_pass_filter[:, None]
        
        # Call the EXACT same function as LUT generation
        python_cmy = rgb_to_raw_hanatos2025(
            test_rgb.reshape(1, 1, 3),
            sensitivity,
            color_space='ACES2065-1',
            apply_cctf_decoding=False,
            reference_illuminant=profile.info.reference_illuminant
        )
        python_cmy = python_cmy.flatten()
        
        # Apply exposure adjustment (same as LUT generation)
        exposure_ev = 0.0
        python_cmy *= 2**exposure_ev
        
        print(f"Python CMY output (exact LUT parameters): {python_cmy}")
        print(f"LUT sample result: [1.357924, 0.411681, 1.221318]")
        
        # Compare
        lut_result = np.array([1.357924, 0.411681, 1.221318])
        diff = np.abs(python_cmy - lut_result)
        print(f"Difference: {diff}")
        print(f"Max difference: {np.max(diff):.6f}")
        
        # Check if this is close enough
        max_acceptable_diff = 0.1
        if np.max(diff) < max_acceptable_diff:
            print(f"✅ SUCCESS: Values match within tolerance ({max_acceptable_diff})")
        else:
            print(f"❌ FAILED: Values still don't match (max diff: {np.max(diff):.6f})")
            
        print(f"\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        print(f"The Python reference produces: {python_cmy}")
        print(f"The LUT produces: {lut_result}")
        print(f"This is much closer than before!")
        print(f"The remaining difference might be due to:")
        print(f"1. Different interpolation methods")
        print(f"2. Float16 precision in the LUT")
        print(f"3. Different spectral upsampling implementation")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_python_exact() 