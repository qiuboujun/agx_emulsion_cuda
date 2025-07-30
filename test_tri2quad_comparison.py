#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.utils.spectral_upsampling import tri2quad

def test_tri2quad_function():
    """Test the tri2quad function specifically"""
    
    print("=== Testing tri2quad Function Directly ===")
    
    # Input xy coordinates from our debug
    xy = np.array([0.64272518, 0.33030524])
    print(f"Input xy: {xy}")
    
    # Python tri2quad calculation
    python_result = tri2quad(xy)
    print(f"Python tri2quad result: {python_result}")
    
    # Manual CUDA tri2quad calculation (from the CUDA code)
    tx = xy[0]
    ty = xy[1]
    y = ty / max(1.0 - tx, 1e-10)
    x = (1.0 - tx) * (1.0 - tx)
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    manual_result = np.array([x, y])
    
    print(f"Manual CUDA calculation: {manual_result}")
    print(f"Difference: {python_result - manual_result}")
    
    # The issue might be that the input coordinates are wrong
    # Let me check what ACTUAL coordinates should go into tri2quad
    print("\n=== Checking Alternative Interpretation ===")
    
    # Maybe the issue is the input coordinate space?
    # Let's see what happens if we treat input differently
    test_coords = [
        [0.642725, 0.330307],  # CUDA debug xy
        [0.25, 0.25],          # Test coordinates
        [0.5, 0.5],            # Center
        [0.1, 0.9],            # Edge case
    ]
    
    for coord in test_coords:
        py_result = tri2quad(np.array(coord))
        
        # Manual calculation
        tx, ty = coord
        y_manual = ty / max(1.0 - tx, 1e-10)
        x_manual = (1.0 - tx) * (1.0 - tx)
        x_manual = max(0.0, min(1.0, x_manual))
        y_manual = max(0.0, min(1.0, y_manual))
        manual_result = np.array([x_manual, y_manual])
        
        print(f"Input: {coord} -> Python: {py_result} | Manual: {manual_result} | Diff: {py_result - manual_result}")

if __name__ == "__main__":
    test_tri2quad_function() 