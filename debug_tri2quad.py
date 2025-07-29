#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the reference path
sys.path.insert(0, 'ref')

from agx_emulsion.utils.spectral_upsampling import tri2quad

def debug_tri2quad():
    """Debug tri2quad function"""
    
    print("=== Debug tri2quad Function ===")
    
    # Test the exact xy coordinates from CUDA debug
    xy = np.array([[0.642725, 0.330307]])
    print(f"Input xy: {xy.flatten()}")
    
    # Apply tri2quad
    tc_python = tri2quad(xy)
    print(f"Python tri2quad result: {tc_python.flatten()}")
    
    # CUDA result from debug log
    tc_cuda = np.array([0.127645, 0.924517])
    print(f"CUDA tri2quad result: {tc_cuda}")
    
    # Compare
    diff = tc_python.flatten() - tc_cuda
    print(f"Difference: {diff}")
    print(f"Max difference: {np.max(np.abs(diff))}")
    
    # Test the tri2quad formula manually
    tx = xy[0,0]
    ty = xy[0,1]
    print(f"\nManual calculation:")
    print(f"tx (input x): {tx}")
    print(f"ty (input y): {ty}")
    
    # tri2quad formula from Python:
    # y = ty / np.fmax(1.0 - tx, 1e-10)
    # x = (1.0 - tx)*(1.0 - tx)
    manual_y = ty / max(1.0 - tx, 1e-10)
    manual_x = (1.0 - tx) * (1.0 - tx)
    manual_x = max(0.0, min(manual_x, 1.0))
    manual_y = max(0.0, min(manual_y, 1.0))
    
    print(f"Manual x: {manual_x}")
    print(f"Manual y: {manual_y}")
    print(f"Manual result: [{manual_x}, {manual_y}]")

if __name__ == "__main__":
    debug_tri2quad() 