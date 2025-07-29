#!/usr/bin/env python3
"""
Verify data types at each pipeline stage between Python and OFX
"""

import numpy as np
import sys
sys.path.insert(0, 'ref')

def verify_pipeline_data_types():
    """Verify data types at each pipeline stage"""
    print("="*80)
    print("PIPELINE DATA TYPE VERIFICATION")
    print("="*80)
    
    # Test input (same as OFX debug)
    test_rgb = np.array([0.292817, 0.267206, 0.173047], dtype=np.float32)
    print(f"Input RGB: {test_rgb}")
    print(f"Type: Light values (0-1 range)")
    
    try:
        from agx_emulsion.profiles.io import load_profile
        from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025
        from agx_emulsion.model.color_filters import compute_band_pass_filter
        from agx_emulsion.model.emulsion import Film
        from agx_emulsion.model.couplers import compute_dir_couplers_matrix
        
        profile = load_profile('kodak_portra_400')
        film = Film(profile)
        
        print(f"\n" + "="*80)
        print("STAGE 1: CAMERA LUT (RGB → CMY)")
        print("="*80)
        
        # Camera LUT stage
        sensitivity = 10**profile.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)
        
        filter_uv = (1, 410, 8)
        filter_ir = (1, 675, 15)
        if filter_uv[0] > 0 or filter_ir[0] > 0:
            band_pass_filter = compute_band_pass_filter(filter_uv, filter_ir)
            sensitivity *= band_pass_filter[:, None]
        
        python_cmy = rgb_to_raw_hanatos2025(
            test_rgb.reshape(1, 1, 3),
            sensitivity,
            color_space='ACES2065-1',
            apply_cctf_decoding=False,
            reference_illuminant=profile.info.reference_illuminant
        )
        python_cmy = python_cmy.flatten()
        
        print(f"Python CMY output: {python_cmy}")
        print(f"OFX CMY output: [1.001534, 0.429265, 1.978867]")
        print(f"Type: CMY LIGHT VALUES (not density, not log exposure)")
        print(f"Range: Typically [0, ~10] for light values")
        
        # Verify they're similar
        ofx_cmy = np.array([1.001534, 0.429265, 1.978867])
        diff = np.abs(python_cmy - ofx_cmy)
        print(f"Difference: {diff}")
        print(f"Max difference: {np.max(diff):.6f}")
        
        print(f"\n" + "="*80)
        print("STAGE 2: EMULSION (CMY Light → Density)")
        print("="*80)
        
        # Emulsion stage - convert CMY light to log exposure, then to density
        gamma = 1.244186
        exposure_ev = 0.0
        
        # Step 1: Convert CMY light to log exposure
        log_exposure = np.log10(np.maximum(python_cmy, 1e-10))
        print(f"Log exposure (log10 of CMY light): {log_exposure}")
        print(f"Type: LOG EXPOSURE VALUES")
        print(f"Range: Typically [-3, 4] for log exposure")
        
        # Step 2: Apply exposure adjustment
        log_exposure_adjusted = log_exposure + exposure_ev * 0.30103
        print(f"Log exposure (after EV adjustment): {log_exposure_adjusted}")
        
        # Step 3: Apply gamma correction
        log_exposure_gamma = log_exposure_adjusted / gamma
        print(f"Log exposure (after gamma): {log_exposure_gamma}")
        
        # Step 4: Convert to density using film curves
        log_raw = log_exposure_gamma.reshape(1, 1, 3)
        python_density = film.develop(log_raw, pixel_size_um=47.339535, bypass_grain=True).flatten()
        
        print(f"Python density output: {python_density}")
        print(f"OFX density output: [0.000000, 1.994974, 0.000000]")
        print(f"Type: DENSITY VALUES")
        print(f"Range: Typically [0, ~3] for density")
        
        print(f"\n" + "="*80)
        print("STAGE 3: DIR COUPLER (Density → RGB Light)")
        print("="*80)
        
        # DIR Coupler stage - converts density back to light
        dir_matrix = compute_dir_couplers_matrix([1.395349, 1.395349, 1.395349], 1.395349)
        python_corrected = np.dot(dir_matrix, python_density)
        
        print(f"Python DIR output: {python_corrected}")
        print(f"OFX DIR output: [2.330595, 0.024586, 2.176574]")
        print(f"Type: RGB LIGHT VALUES")
        print(f"Range: [0, ∞) for light values (can exceed 1.0)")
        
        print(f"\n" + "="*80)
        print("DATA TYPE SUMMARY")
        print("="*80)
        print("Stage 1 (Camera LUT): RGB Light [0,1] → CMY Light [0,~10]")
        print("Stage 2 (Emulsion):   CMY Light [0,~10] → Log Exposure [-3,4] → Density [0,~3]")
        print("Stage 3 (DIR):        Density [0,~3] → RGB Light [0,∞)")
        print("Stage 4 (Paper):      RGB Light [0,∞) → sRGB [0,1]")
        
        print(f"\n" + "="*80)
        print("ISSUE ANALYSIS")
        print("="*80)
        print("The OFX Emulsion stage is producing [0.000000, 1.994974, 0.000000]")
        print("This suggests the film curves are not being loaded correctly.")
        print("Expected range should be [0, ~3] for density values.")
        print("The zeros indicate the lookup is failing or returning minimum values.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_pipeline_data_types() 