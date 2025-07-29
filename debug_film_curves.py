#!/usr/bin/env python3
"""
Debug script to examine film curve ranges
"""

import numpy as np
import sys
sys.path.insert(0, 'ref')

def debug_film_curves():
    """Debug the film curve ranges"""
    print("="*80)
    print("FILM CURVE DEBUG")
    print("="*80)
    
    try:
        from agx_emulsion.profiles.io import load_profile
        
        profile = load_profile('kodak_portra_400')
        
        print(f"Film: {profile.info.stock}")
        print(f"Log exposure range: {profile.data.log_exposure.min():.6f} to {profile.data.log_exposure.max():.6f}")
        print(f"Density curves shape: {profile.data.density_curves.shape}")
        
        # Check the first few and last few values
        print(f"\nFirst 5 log exposure values: {profile.data.log_exposure[:5]}")
        print(f"Last 5 log exposure values: {profile.data.log_exposure[-5:]}")
        
        print(f"\nFirst 5 density values (R): {profile.data.density_curves[:5, 0]}")
        print(f"Last 5 density values (R): {profile.data.density_curves[-5:, 0]}")
        
        print(f"First 5 density values (G): {profile.data.density_curves[:5, 1]}")
        print(f"Last 5 density values (G): {profile.data.density_curves[-5:, 1]}")
        
        print(f"First 5 density values (B): {profile.data.density_curves[:5, 2]}")
        print(f"Last 5 density values (B): {profile.data.density_curves[-5:, 2]}")
        
        # Test the exact values from OFX
        input_cmy = np.array([1.001534, 0.429265, 1.978867])
        gamma = 1.244186
        exposure_ev = 0.0
        
        print(f"\n" + "="*80)
        print("TESTING OFX VALUES")
        print("="*80)
        
        # Convert to log exposure
        log_exposure = np.log10(np.maximum(input_cmy, 1e-6))
        print(f"Log exposure: {log_exposure}")
        
        # Apply exposure and gamma
        log_exposure_adjusted = log_exposure + exposure_ev * 0.30103
        print(f"Log exposure (after EV): {log_exposure_adjusted}")
        
        log_exposure_gamma = log_exposure_adjusted / gamma
        print(f"Log exposure (after gamma): {log_exposure_gamma}")
        
        # Check if these values are within the film curve range
        min_loge = profile.data.log_exposure.min()
        max_loge = profile.data.log_exposure.max()
        
        print(f"\nFilm curve range: [{min_loge:.6f}, {max_loge:.6f}]")
        
        for i, val in enumerate(log_exposure_gamma):
            channel = ['R', 'G', 'B'][i]
            if val < min_loge:
                print(f"  {channel}: {val:.6f} < {min_loge:.6f} (CLAMPED TO MIN)")
            elif val > max_loge:
                print(f"  {channel}: {val:.6f} > {max_loge:.6f} (CLAMPED TO MAX)")
            else:
                print(f"  {channel}: {val:.6f} (IN RANGE)")
        
        # Simulate the lookupDensity function
        print(f"\n" + "="*80)
        print("SIMULATING LOOKUP")
        print("="*80)
        
        for i, val in enumerate(log_exposure_gamma):
            channel = ['R', 'G', 'B'][i]
            curve = profile.data.density_curves[:, i]
            
            if val <= min_loge:
                result = curve[0]
                print(f"  {channel}: {val:.6f} <= {min_loge:.6f} -> {result:.6f}")
            elif val >= max_loge:
                result = curve[-1]
                print(f"  {channel}: {val:.6f} >= {max_loge:.6f} -> {result:.6f}")
            else:
                # Find the index
                idx = 0
                for j in range(1, len(profile.data.log_exposure)):
                    if val < profile.data.log_exposure[j]:
                        idx = j - 1
                        break
                
                # Interpolate
                t = (val - profile.data.log_exposure[idx]) / (profile.data.log_exposure[idx+1] - profile.data.log_exposure[idx])
                result = curve[idx] + t * (curve[idx+1] - curve[idx])
                print(f"  {channel}: {val:.6f} -> interpolated {result:.6f} (idx={idx}, t={t:.3f})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_film_curves() 