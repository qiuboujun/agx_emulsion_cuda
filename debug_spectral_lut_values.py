#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import os

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.utils.spectral_upsampling import HANATOS2025_SPECTRA_LUT

def debug_spectral_lut():
    """Debug spectral LUT values at specific coordinates"""
    
    print("=== Debug Spectral LUT Values ===")
    
    # Load the original NumPy LUT
    print(f"Original NumPy LUT shape: {HANATOS2025_SPECTRA_LUT.shape}")
    print(f"Original NumPy LUT range: [{np.min(HANATOS2025_SPECTRA_LUT):.6f}, {np.max(HANATOS2025_SPECTRA_LUT):.6f}]")
    
    # Check specific coordinates that match our test case
    # CUDA tc = (0.127645, 0.924517)
    # This maps to approximately:
    tc_x = 0.127645
    tc_y = 0.924517
    
    # Map to LUT indices (192x192)
    lut_size = 192
    x_idx = int(tc_x * (lut_size - 1))
    y_idx = int(tc_y * (lut_size - 1))
    
    print(f"\nTest coordinates: tc=({tc_x:.6f}, {tc_y:.6f})")
    print(f"Mapped to LUT indices: ({x_idx}, {y_idx})")
    
    # Get spectrum at this coordinate from original LUT
    spectrum_orig = HANATOS2025_SPECTRA_LUT[y_idx, x_idx, :]
    print(f"Original spectrum at ({x_idx},{y_idx}):")
    print(f"  Shape: {spectrum_orig.shape}")
    print(f"  Range: [{np.min(spectrum_orig):.6f}, {np.max(spectrum_orig):.6f}]")
    print(f"  First 5: {spectrum_orig[:5]}")
    print(f"  spectrum[0]: {spectrum_orig[0]:.6f}, spectrum[40]: {spectrum_orig[40]:.6f}, spectrum[80]: {spectrum_orig[80]:.6f}")
    
    # Now check the CSV file
    csv_path = "AgXEmulsionOFX/AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/data/irradiance_xy_tc.csv"
    if os.path.exists(csv_path):
        print(f"\n=== CSV File Check ===")
        
        # Read first few rows of CSV
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        print(f"CSV has {len(lines)} lines")
        print(f"First line: {lines[0][:100]}...")
        
        # Find the line corresponding to our coordinate
        target_coord = y_idx * lut_size + x_idx
        print(f"Target coordinate index: {target_coord}")
        
        if target_coord < len(lines):
            # Parse the target line
            target_line = lines[target_coord].strip()
            values = target_line.split(',')
            
            print(f"CSV line {target_coord}: {target_line[:100]}...")
            print(f"Number of values in line: {len(values)}")
            
            if len(values) >= 82:  # 1 coordinate + 81 spectral values
                try:
                    coord_idx = int(float(values[0]))
                    spectrum_csv = [float(values[i]) for i in range(1, 82)]
                    
                    print(f"CSV coordinate index: {coord_idx}")
                    print(f"CSV spectrum range: [{min(spectrum_csv):.6f}, {max(spectrum_csv):.6f}]")
                    print(f"CSV spectrum[0]: {spectrum_csv[0]:.6f}, spectrum[40]: {spectrum_csv[40]:.6f}, spectrum[80]: {spectrum_csv[80]:.6f}")
                    
                    # Compare
                    print(f"\n=== Comparison ===")
                    print(f"Original vs CSV spectrum[0]: {spectrum_orig[0]:.6f} vs {spectrum_csv[0]:.6f} (diff: {abs(spectrum_orig[0] - spectrum_csv[0]):.6f})")
                    print(f"Original vs CSV spectrum[40]: {spectrum_orig[40]:.6f} vs {spectrum_csv[40]:.6f} (diff: {abs(spectrum_orig[40] - spectrum_csv[40]):.6f})")
                    print(f"Original vs CSV spectrum[80]: {spectrum_orig[80]:.6f} vs {spectrum_csv[80]:.6f} (diff: {abs(spectrum_orig[80] - spectrum_csv[80]):.6f})")
                    
                except Exception as e:
                    print(f"Error parsing CSV line: {e}")
            else:
                print(f"CSV line has insufficient values: {len(values)}")
        else:
            print(f"Target coordinate {target_coord} is beyond CSV length {len(lines)}")
    else:
        print(f"CSV file not found: {csv_path}")

if __name__ == "__main__":
    debug_spectral_lut() 