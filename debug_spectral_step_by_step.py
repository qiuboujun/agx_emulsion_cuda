#!/usr/bin/env python3

import numpy as np
import sys
import os
import colour

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.model.emulsion import Film
from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025, rgb_to_tc_b, rgb_to_spectrum, HANATOS2025_SPECTRA_LUT
from agx_emulsion.profiles.io import load_profile

def debug_spectral_upsampling():
    """Debug spectral upsampling step by step"""
    
    print("=== Debug Spectral Upsampling Step-by-Step ===")
    
    # Input RGB from DaVinci debug
    input_rgb = np.array([[[0.176700, 0.036017, 0.007001]]])
    print(f"Input RGB: {input_rgb.flatten()}")
    
    # Load film 
    profile = load_profile('kodak_portra_400')
    film = Film(profile)
    camera_sens = film.sensitivity
    
    print(f"\nCamera sensitivity shape: {camera_sens.shape}")
    print(f"Camera sens at wl=0: R={camera_sens[0,0]:.6f}, G={camera_sens[0,1]:.6f}, B={camera_sens[0,2]:.6f}")
    print(f"Camera sens at wl=40: R={camera_sens[40,0]:.6f}, G={camera_sens[40,1]:.6f}, B={camera_sens[40,2]:.6f}")
    print(f"Camera sens at wl=80: R={camera_sens[80,0]:.6f}, G={camera_sens[80,1]:.6f}, B={camera_sens[80,2]:.6f}")
    
    # Step 1: RGB to XYZ conversion
    print(f"\n=== Step 1: RGB to XYZ conversion ===")
    
    # Get ACES2065-1 to XYZ matrix from colour library  
    aces_cs = colour.RGB_COLOURSPACES['ACES2065-1']
    print(f"ACES2065-1 matrix:\n{aces_cs.matrix_RGB_to_XYZ}")
    
    # Manual calculation
    manual_xyz = colour.RGB_to_XYZ(input_rgb, colourspace='ACES2065-1')
    print(f"Manual XYZ: {manual_xyz.flatten()}")
    
    # Step 2: rgb_to_tc_b function 
    print(f"\n=== Step 2: rgb_to_tc_b function ===")
    tc, b = rgb_to_tc_b(input_rgb, color_space='ACES2065-1', apply_cctf_decoding=False, reference_illuminant='D55')
    print(f"tc (triangular coords): {tc.flatten()}")
    print(f"b (brightness): {b.flatten()}")
    
    # Step 3: Fetch spectrum at tc coordinates
    print(f"\n=== Step 3: Fetch spectrum ===")
    spectrum = rgb_to_spectrum(input_rgb, color_space='ACES2065-1', apply_cctf_decoding=False, reference_illuminant='D55')
    print(f"Spectrum shape: {spectrum.shape}")
    print(f"Spectrum range: [{np.min(spectrum):.6f}, {np.max(spectrum):.6f}]")
    print(f"Spectrum first 5: {spectrum[:5]}")
    print(f"Spectrum last 5: {spectrum[-5:]}")
    
    # Step 4: Apply camera sensitivity 
    print(f"\n=== Step 4: Apply camera sensitivity ===")
    raw_rgb = np.einsum('l,lm->m', spectrum, camera_sens)
    print(f"Raw RGB: {raw_rgb}")
    
    # Step 5: Midgray normalization
    print(f"\n=== Step 5: Midgray normalization ===")
    midgray_rgb = np.array([[[0.184, 0.184, 0.184]]])
    midgray_spectrum = rgb_to_spectrum(midgray_rgb, color_space='ACES2065-1', apply_cctf_decoding=False, reference_illuminant='D55')
    midgray_raw = np.einsum('l,lm->m', midgray_spectrum, camera_sens)
    print(f"Midgray spectrum range: [{np.min(midgray_spectrum):.6f}, {np.max(midgray_spectrum):.6f}]")
    print(f"Midgray raw RGB: {midgray_raw}")
    print(f"Midgray normalization factor: {1.0/midgray_raw[1]:.6f}")
    
    # Final result
    final_cmy = raw_rgb / midgray_raw[1]
    print(f"\nFinal CMY: {final_cmy}")
    
    # Compare with full function
    print(f"\n=== Compare with full function ===")
    full_result = rgb_to_raw_hanatos2025(input_rgb, camera_sens, color_space='ACES2065-1', apply_cctf_decoding=False, reference_illuminant='D55')
    print(f"Full function result: {full_result.flatten()}")
    print(f"Manual calculation: {final_cmy}")
    print(f"Difference: {full_result.flatten() - final_cmy}")

if __name__ == "__main__":
    debug_spectral_upsampling() 