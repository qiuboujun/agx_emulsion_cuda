#!/usr/bin/env python3
"""
Test to show the key differences between Python and C++ handling of NaN values
"""

import json
import numpy as np

def test_python_vs_cpp_handling():
    """Test the key differences between Python and C++ handling"""
    print("="*80)
    print("PYTHON vs C++ HANDLING DIFFERENCES")
    print("="*80)
    
    # Test the JSON file that the OFX plugin will load
    json_path = "/usr/OFX/Plugins/data/profiles/kodak_portra_400.json"
    
    try:
        # Load with Python (reference implementation)
        with open(json_path, 'r') as f:
            profile = json.load(f)
        
        # Extract data arrays
        data = profile['data']
        log_exposure_array = data['log_exposure']
        density_curves_array = data['density_curves']
        
        # Convert to numpy arrays
        log_exposure = np.array(log_exposure_array)
        density_curves = np.array(density_curves_array)
        
        print(f"Original data:")
        print(f"  Log exposure length: {len(log_exposure)}")
        print(f"  Density curves shape: {density_curves.shape}")
        print(f"  Log exposure range: [{log_exposure.min():.6f}, {log_exposure.max():.6f}]")
        
        # Count NaN values
        nan_count_r = np.isnan(density_curves[:, 0]).sum()
        nan_count_g = np.isnan(density_curves[:, 1]).sum()
        nan_count_b = np.isnan(density_curves[:, 2]).sum()
        
        print(f"\nNaN Analysis:")
        print(f"  NaN values in R channel: {nan_count_r}")
        print(f"  NaN values in G channel: {nan_count_g}")
        print(f"  NaN values in B channel: {nan_count_b}")
        
        print(f"\n" + "="*80)
        print("KEY DIFFERENCE #1: Python subtracts nanmin")
        print("="*80)
        
        # Python approach: subtract nanmin (line 105 in emulsion.py)
        density_curves_python = density_curves.copy()
        density_curves_python -= np.nanmin(density_curves_python, axis=0)
        
        print(f"Python approach (subtract nanmin):")
        print(f"  Original R range: [{np.nanmin(density_curves[:, 0]):.6f}, {np.nanmax(density_curves[:, 0]):.6f}]")
        print(f"  After nanmin subtraction R range: [{np.nanmin(density_curves_python[:, 0]):.6f}, {np.nanmax(density_curves_python[:, 0]):.6f}]")
        print(f"  Original G range: [{np.nanmin(density_curves[:, 1]):.6f}, {np.nanmax(density_curves[:, 1]):.6f}]")
        print(f"  After nanmin subtraction G range: [{np.nanmin(density_curves_python[:, 1]):.6f}, {np.nanmax(density_curves_python[:, 1]):.6f}]")
        print(f"  Original B range: [{np.nanmin(density_curves[:, 2]):.6f}, {np.nanmax(density_curves[:, 2]):.6f}]")
        print(f"  After nanmin subtraction B range: [{np.nanmin(density_curves_python[:, 2]):.6f}, {np.nanmax(density_curves_python[:, 2]):.6f}]")
        
        print(f"\n" + "="*80)
        print("KEY DIFFERENCE #2: Python uses fast_interp with NaN handling")
        print("="*80)
        
        # Python approach: uses fast_interp which handles NaN values internally
        # The commented code shows the old approach:
        # sel = ~np.isnan(density_curves[:,channel])
        # density_cmy[:,:,channel] = np.interp(log_exposure_rgb[:,:,channel],
        #                                      log_exposure[sel]/gamma_factor[channel],
        #                                      density_curves[sel,channel])
        
        print(f"Python approach (fast_interp):")
        print(f"  Uses fast_interp which internally filters out NaN values")
        print(f"  Only uses valid (non-NaN) data points for interpolation")
        print(f"  This means the starting zeros don't affect the interpolation")
        
        print(f"\n" + "="*80)
        print("C++ APPROACH (Current Implementation)")
        print("="*80)
        
        # C++ approach: convert NaN to 0.0, then use all data points
        density_curves_cpp = np.nan_to_num(density_curves, nan=0.0)
        
        print(f"C++ approach (nan_to_num):")
        print(f"  Converts all NaN values to 0.0")
        print(f"  Uses ALL data points including the zeros")
        print(f"  This means the starting zeros ARE used in interpolation")
        print(f"  R range after nan_to_num: [{density_curves_cpp[:, 0].min():.6f}, {density_curves_cpp[:, 0].max():.6f}]")
        print(f"  G range after nan_to_num: [{density_curves_cpp[:, 1].min():.6f}, {density_curves_cpp[:, 1].max():.6f}]")
        print(f"  B range after nan_to_num: [{density_curves_cpp[:, 2].min():.6f}, {density_curves_cpp[:, 2].max():.6f}]")
        
        print(f"\n" + "="*80)
        print("THE PROBLEM")
        print("="*80)
        
        print(f"❌ C++ approach is WRONG because:")
        print(f"  1. It doesn't subtract nanmin like Python does")
        print(f"  2. It includes the starting zeros in interpolation")
        print(f"  3. This causes incorrect lookup results")
        
        print(f"\n✅ Python approach is CORRECT because:")
        print(f"  1. It subtracts nanmin to normalize the curves")
        print(f"  2. It only uses valid (non-NaN) data points for interpolation")
        print(f"  3. The starting zeros are effectively ignored")
        
        print(f"\n" + "="*80)
        print("SOLUTION")
        print("="*80)
        
        print(f"To fix the C++ implementation, we need to:")
        print(f"  1. Subtract the minimum value from each channel (like Python)")
        print(f"  2. OR filter out NaN values before interpolation (like Python)")
        print(f"  3. OR implement the same fast_interp logic as Python")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_python_vs_cpp_handling() 