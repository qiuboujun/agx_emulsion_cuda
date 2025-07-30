#!/usr/bin/env python3

import math

def tri2quad(tc):
    """Python version of CUDA tri2quad function"""
    tx, ty = tc[0], tc[1]
    y = ty / max(1.0 - tx, 1e-10)
    x = (1.0 - tx) * (1.0 - tx)
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return [x, y]

def test_correct_transformation():
    # Test with the exact xy coordinates from debug output
    xy_input = [0.642725, 0.330307]
    print(f"Input xy: {xy_input}")
    
    # Direct tri2quad transformation (what Python and corrected CUDA should do)
    tc_output = tri2quad(xy_input)
    print(f"Correct tc output: {tc_output}")
    
    # This should match our earlier analysis of Python output
    expected_python_tc = [0.127645, 0.924517]
    print(f"Expected from earlier analysis: {expected_python_tc}")
    
    # Check if they match
    diff = [abs(tc_output[0] - expected_python_tc[0]), abs(tc_output[1] - expected_python_tc[1])]
    print(f"Difference: {diff}")
    
    if max(diff) < 1e-5:
        print("✅ PERFECT MATCH! CUDA should now produce correct coordinates")
    else:
        print("❌ Mismatch - need to investigate further")

if __name__ == "__main__":
    test_correct_transformation() 