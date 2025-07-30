#!/usr/bin/env python3

import numpy as np
import math

def quad2tri(xy):
    """Python version of CUDA quad2tri function"""
    x, y = xy[0], xy[1]
    tx = 1.0 - math.sqrt(x)
    ty = y * math.sqrt(x)
    return [tx, ty]

def tri2quad(tc):
    """Python version of CUDA tri2quad function"""
    tx, ty = tc[0], tc[1]
    y = ty / max(1.0 - tx, 1e-10)
    x = (1.0 - tx) * (1.0 - tx)
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return [x, y]

def test_coordinate_transform():
    # Test with the exact xy coordinates from debug output
    xy_input = [0.642725, 0.330307]
    print(f"Input xy: {xy_input}")
    
    # Step 1: xy -> triangular coordinates
    triangular = quad2tri(xy_input)
    print(f"Triangular coordinates: {triangular}")
    
    # Step 2: triangular -> texture coordinates  
    tc_output = tri2quad(triangular)
    print(f"Output tc: {tc_output}")
    
    # Check if they're the same (which would explain the debug output)
    diff = [abs(tc_output[0] - xy_input[0]), abs(tc_output[1] - xy_input[1])]
    print(f"Difference: {diff}")
    
    if max(diff) < 1e-6:
        print("ERROR: Transformation returns identical coordinates!")
    else:
        print("OK: Transformation produces different coordinates")

if __name__ == "__main__":
    test_coordinate_transform() 