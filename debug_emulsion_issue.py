#!/usr/bin/env python3
"""
Debug script to test the exact same values as the OFX plugin
"""

import numpy as np
import sys
sys.path.insert(0, 'ref')

def debug_emulsion_stage():
    """Debug the emulsion stage with exact OFX values"""
    print("="*80)
    print("EMULSION STAGE DEBUG")
    print("="*80)
    
    # Exact values from OFX debug output
    input_cmy = np.array([1.001534, 0.429265, 1.978867], dtype=np.float32)
    gamma = 1.244186
    exposure_ev = 0.0
    
    print(f"Input CMY: {input_cmy}")
    print(f"Gamma: {gamma}")
    print(f"Exposure EV: {exposure_ev}")
    
    # Step 1: Convert CMY to log exposure
    # CMY values are already in density space, so we need to convert to log exposure
    log_raw = np.log10(input_cmy + 1e-10)
    print(f"Log raw (log10 of CMY): {log_raw}")
    
    # Step 2: Apply exposure adjustment
    log_exposure = log_raw + exposure_ev
    print(f"Log exposure (after EV adjustment): {log_exposure}")
    
    # Step 3: Apply gamma correction
    # This should convert log exposure to density
    density = log_exposure / gamma
    print(f"Density (after gamma): {density}")
    
    # Step 4: Clamp to valid range
    density = np.clip(density, 0.0, 3.0)
    print(f"Final density (clamped): {density}")
    
    print("\n" + "="*80)
    print("COMPARISON WITH OFX OUTPUT")
    print("="*80)
    print(f"OFX Input CMY:  {input_cmy}")
    print(f"OFX Output:     [0.000000, 1.994974, 0.000000]")
    print(f"Python Output:  {density}")
    
    # Check if there's an issue with the conversion
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # The issue might be in how we're interpreting the CMY values
    # Let's check if they should be treated as light values instead of density
    print("If CMY values are light values (not density):")
    light_values = input_cmy
    log_light = np.log10(light_values + 1e-10)
    print(f"Log light: {log_light}")
    
    # Apply exposure and gamma
    density_from_light = log_light / gamma
    print(f"Density from light: {density_from_light}")
    
    # Or maybe the issue is in the gamma application
    print("\nAlternative gamma application:")
    # Maybe gamma should be applied differently
    density_alt = log_exposure * gamma  # Instead of dividing
    print(f"Density (alt method): {density_alt}")
    
    # Let's also check the film profile to understand the expected range
    try:
        from agx_emulsion.profiles.io import load_profile
        from agx_emulsion.model.emulsion import Film
        
        profile = load_profile('kodak_portra_400')
        film = Film(profile)
        
        print(f"\nFilm profile info:")
        print(f"  Stock: {profile.info.stock}")
        print(f"  Log exposure range: {profile.data.log_exposure.min():.3f} to {profile.data.log_exposure.max():.3f}")
        print(f"  Density range: {profile.data.density_curves.min():.3f} to {profile.data.density_curves.max():.3f}")
        
        # Test with the exact input
        test_input = input_cmy.reshape(1, 1, 3)
        print(f"\nTesting with film.develop():")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Input values: {test_input.flatten()}")
        
        result = film.develop(test_input, pixel_size_um=47.339535, bypass_grain=True)
        print(f"  Film.develop() output: {result.flatten()}")
        
    except Exception as e:
        print(f"Error loading film profile: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_emulsion_stage() 