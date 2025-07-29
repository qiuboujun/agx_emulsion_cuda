#!/usr/bin/env python3

import sys
import os
sys.path.append('ref')

import numpy as np

def test_simple_spectral():
    print("=== Simple Spectral Upsampling Test ===")
    
    # Test with exact ACES input from debug log
    input_rgb = np.array([[[0.1767, 0.036017, 0.007001]]])
    print(f"Input RGB: {input_rgb[0,0]}")
    
    try:
        # Import only the spectral upsampling function
        from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025
        
        # Create a dummy camera sensitivity (we can use the one we know from the CUDA debug)
        # Based on CUDA debug: "Sensitivity R[40]=72.167427, G[40]=82.952286, B[40]=0.000000"
        # Let's create a basic sensitivity curve
        wavelengths = np.linspace(400, 720, 81)
        sens_r = np.zeros(81)
        sens_g = np.zeros(81) 
        sens_b = np.zeros(81)
        
        # Set some basic sensitivity values (not exact, but for testing)
        sens_r[30:50] = 70.0  # Red peak around 40
        sens_g[35:55] = 80.0  # Green peak around 40
        sens_b[0:20] = 90.0   # Blue peak early
        
        camera_sens = np.array([sens_r, sens_g, sens_b])
        print(f"Camera sensitivity shape: {camera_sens.shape}")
        
        # Perform spectral upsampling
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
        
        # Check if CUDA is just passing through input
        print(f"\nInput RGB vs CUDA CMY:")
        print(f"Input:  [{input_rgb[0,0,0]:.6f}, {input_rgb[0,0,1]:.6f}, {input_rgb[0,0,2]:.6f}]")
        print(f"CUDA:   [0.176700, 0.036017, 0.007001]")
        print(f"The CUDA values are identical to input - spectral upsampling is not working!")
        
        if np.allclose(cmy_result[0,0], input_rgb[0,0], rtol=0.01):
            print("⚠️  Python is also returning input unchanged - this suggests the spectral LUT is not working")
        else:
            print("✅ Python spectral upsampling is working correctly")
        
    except Exception as e:
        print(f"Error in spectral upsampling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_spectral() 