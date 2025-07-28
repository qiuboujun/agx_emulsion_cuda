#!/usr/bin/env python3
"""
Export CIE 1931 2° CMFs and D50 illuminant data for CUDA implementation
"""
import numpy as np
import colour
import sys
import os

# Add ref/agx_emulsion to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ref', 'agx_emulsion'))

try:
    from config import SPECTRAL_SHAPE, STANDARD_OBSERVER_CMFS
    from model.illuminants import standard_illuminant
except ImportError as e:
    print(f"Failed to import agx_emulsion modules: {e}")
    print("Using fallback colour-science direct access...")
    
    # Fallback: access colour-science directly
    SPECTRAL_SHAPE = colour.SpectralShape(380, 780, 5)
    STANDARD_OBSERVER_CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(SPECTRAL_SHAPE)
    
    # Simple D50 illuminant
    def standard_illuminant(name):
        if name.upper() == 'D50':
            return colour.SDS_ILLUMINANTS['D50'].copy().align(SPECTRAL_SHAPE).values
        else:
            return colour.SDS_ILLUMINANTS['D65'].copy().align(SPECTRAL_SHAPE).values

def export_cie_data():
    print(f"Spectral shape: {SPECTRAL_SHAPE}")
    print(f"CMF shape: {STANDARD_OBSERVER_CMFS.shape}")
    
    # Get wavelengths
    wavelengths = STANDARD_OBSERVER_CMFS.wavelengths
    print(f"Wavelengths: {wavelengths[0]} to {wavelengths[-1]} nm, {len(wavelengths)} samples")
    
    # Get CMF values (x_bar, y_bar, z_bar)
    xbar = STANDARD_OBSERVER_CMFS.values[:, 0]  # X color matching function
    ybar = STANDARD_OBSERVER_CMFS.values[:, 1]  # Y color matching function  
    zbar = STANDARD_OBSERVER_CMFS.values[:, 2]  # Z color matching function
    
    # Get D50 illuminant SPD
    d50_spd = standard_illuminant('D50')
    print(f"D50 SPD shape: {d50_spd.shape}")
    
    # Get D65 illuminant SPD
    d65_spd = standard_illuminant('D65')
    print(f"D65 SPD shape: {d65_spd.shape}")
    
    # K75P - approximated as 7500K blackbody (common projection standard)
    k75p_spd = colour.sd_blackbody(7500, SPECTRAL_SHAPE).values
    print(f"K75P (7500K) SPD shape: {k75p_spd.shape}")
    
    # Bradford chromatic adaptation transform matrices
    # From D50 to D65 and back
    bradford_m = np.array([
        [ 0.8951,  0.2664, -0.1614],
        [-0.7502,  1.7135,  0.0367],
        [ 0.0389, -0.0685,  1.0296]
    ])
    bradford_m_inv = np.array([
        [ 0.9869929, -0.1470543,  0.1599627],
        [ 0.4323053,  0.5183603,  0.0492912],
        [-0.0085287,  0.0400428,  0.9684867]
    ])
    
    # Pre-compute D65 cone responses for CAT
    d65_white = np.array([0.95047, 1.0, 1.08883])  # D65 XYZ
    d65_lms = bradford_m @ d65_white
    print(f"D65 LMS cone responses: {d65_lms}")
    
    # Export to header file
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'AgXEmulsionOFX')
    
    with open(os.path.join(output_dir, 'CIE1931.cuh'), 'w') as f:
        f.write("// Auto-generated CIE 1931 2° CMFs and D50 illuminant data\n")
        f.write("// Spectral range: 380-780 nm, 5 nm step, 81 samples\n\n")
        f.write("#pragma once\n\n")
        
        f.write("// Number of spectral samples\n")
        f.write(f"#define CIE_SAMPLES {len(wavelengths)}\n\n")
        
        f.write("// CIE 1931 2° Standard Observer x-bar values\n")
        f.write("__constant__ float c_xBar[CIE_SAMPLES] = {\n")
        for i, val in enumerate(xbar):
            f.write(f"    {val:.6f}f")
            if i < len(xbar) - 1:
                f.write(",")
            if (i + 1) % 4 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        
        f.write("// CIE 1931 2° Standard Observer y-bar values\n")
        f.write("__constant__ float c_yBar[CIE_SAMPLES] = {\n")
        for i, val in enumerate(ybar):
            f.write(f"    {val:.6f}f")
            if i < len(ybar) - 1:
                f.write(",")
            if (i + 1) % 4 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        
        f.write("// CIE 1931 2° Standard Observer z-bar values\n")
        f.write("__constant__ float c_zBar[CIE_SAMPLES] = {\n")
        for i, val in enumerate(zbar):
            f.write(f"    {val:.6f}f")
            if i < len(zbar) - 1:
                f.write(",")
            if (i + 1) % 4 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        
        f.write("// D50 Standard Illuminant SPD values\n")
        f.write("__constant__ float c_d50SPD[CIE_SAMPLES] = {\n")
        for i, val in enumerate(d50_spd):
            f.write(f"    {val:.6f}f")
            if i < len(d50_spd) - 1:
                f.write(",")
            if (i + 1) % 4 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        
        f.write("// D65 Standard Illuminant SPD values\n")
        f.write("__constant__ float c_d65SPD[CIE_SAMPLES] = {\n")
        for i, val in enumerate(d65_spd):
            f.write(f"    {val:.6f}f")
            if i < len(d65_spd) - 1:
                f.write(",")
            if (i + 1) % 4 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        
        f.write("// K75P (7500K) Standard Illuminant SPD values\n")
        f.write("__constant__ float c_k75pSPD[CIE_SAMPLES] = {\n")
        for i, val in enumerate(k75p_spd):
            f.write(f"    {val:.6f}f")
            if i < len(k75p_spd) - 1:
                f.write(",")
            if (i + 1) % 4 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        
        f.write("// Bradford chromatic adaptation transform matrix\n")
        f.write("__constant__ float c_bradfordM[9] = {\n")
        for i, val in enumerate(bradford_m.flatten()):
            f.write(f"    {val:.7f}f")
            if i < 8:
                f.write(",")
            if (i + 1) % 3 == 0:
                f.write("\n")
        f.write("};\n\n")
        
        f.write("// Bradford inverse chromatic adaptation transform matrix\n")
        f.write("__constant__ float c_bradfordMinv[9] = {\n")
        for i, val in enumerate(bradford_m_inv.flatten()):
            f.write(f"    {val:.7f}f")
            if i < 8:
                f.write(",")
            if (i + 1) % 3 == 0:
                f.write("\n")
        f.write("};\n\n")
        
        f.write("// D65 cone responses (LMS) for Bradford CAT\n")
        f.write("__constant__ float c_d65LMS[3] = {\n")
        for i, val in enumerate(d65_lms):
            f.write(f"    {val:.7f}f")
            if i < 2:
                f.write(",")
        f.write("\n};\n\n")
        
        f.write("// Wavelength values (for reference)\n")
        f.write("// 380.0, 385.0, 390.0, ..., 775.0, 780.0 nm\n")
    
    print(f"Exported CIE data to {output_dir}/CIE1931.cuh")
    
    # Print sample values for verification
    print(f"\nSample values:")
    print(f"Wavelength 550nm (index ~34): xbar={xbar[34]:.6f}, ybar={ybar[34]:.6f}, zbar={zbar[34]:.6f}")
    print(f"D50 at 550nm: {d50_spd[34]:.6f}")

if __name__ == "__main__":
    export_cie_data() 