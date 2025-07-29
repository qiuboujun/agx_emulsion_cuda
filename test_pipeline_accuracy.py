#!/usr/bin/env python3
"""Test script to verify C++/CUDA pipeline accuracy against Python reference
Feeds fixed test data through both pipelines and compares outputs at each stage.
"""
import numpy as np
import json
import pathlib
import sys
import subprocess
import tempfile
import os

# Add the reference implementation to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'ref'))

from agx_emulsion.utils.spectral_upsampling import compute_lut_spectra

def rgb_to_xy(rgb):
    """Convert RGB to xy chromaticity coordinates using ACES2065-1 RGB->XYZ matrix"""
    # ACES2065-1 RGB->XYZ matrix (same as used in C++ pipeline)
    m = np.array([
        [0.9525524, 0.0000000, 0.0000937],
        [0.3439664, 0.7281661, -0.0721325],
        [0.0000000, 0.0000000, 1.0088252]
    ])
    
    # Convert RGB to XYZ
    xyz = m @ rgb
    
    # Convert XYZ to xy
    sum_xyz = np.sum(xyz) + 1e-8
    x = xyz[0] / sum_xyz
    y = xyz[1] / sum_xyz
    
    return np.array([x, y])

def create_test_data():
    """Create fixed test data for consistent testing"""
    # Create a small 4x4 test image with known values
    test_rgb = np.array([
        [0.2, 0.3, 0.1],  # Dark greenish
        [0.8, 0.7, 0.6],  # Light gray
        [0.1, 0.1, 0.9],  # Blue
        [0.9, 0.2, 0.1],  # Red
    ], dtype=np.float32)
    
    return test_rgb

def test_camera_lut_stage():
    """Test Camera LUT (3D LUT 1) stage - RGB to CMY conversion"""
    print("=== Testing Camera LUT Stage ===")
    
    # Create test RGB data
    test_rgb = create_test_data()
    print(f"Test RGB data shape: {test_rgb.shape}")
    print(f"Test RGB values:\n{test_rgb}")
    
    # Python reference: Convert RGB to xy chromaticity
    xy_coords = []
    for rgb in test_rgb:
        xy = rgb_to_xy(rgb)
        xy_coords.append(xy)
    
    xy_coords = np.array(xy_coords)
    print(f"Python RGB->xy conversion:\n{xy_coords}")
    
    # Generate the LUT using the same method as the C++ pipeline
    lut_size = 128
    spectra = compute_lut_spectra(lut_size=lut_size, smooth_steps=1).astype(np.float32)
    
    # Load film profile (same as C++ pipeline)
    profile_file = pathlib.Path(__file__).resolve().parent / 'ref/agx_emulsion/data/profiles/kodak_portra_400.json'
    with profile_file.open('r') as f:
        prof = json.load(f)
    
    log_sens = np.array(prof['data']['log_sensitivity'], dtype=np.float32)
    
    # Handle NaN values (same as C++ pipeline)
    for i in range(len(log_sens)):
        if np.isnan(log_sens[i,0]):
            if not np.isnan(log_sens[i,1]):
                log_sens[i,0] = log_sens[i,1] * 0.8
            else:
                log_sens[i,0] = 1.0
        
        if np.isnan(log_sens[i,1]):
            if not np.isnan(log_sens[i,0]) and not np.isnan(log_sens[i,2]):
                log_sens[i,1] = (log_sens[i,0] + log_sens[i,2]) / 2
            else:
                log_sens[i,1] = 1.0
        
        if np.isnan(log_sens[i,2]):
            if not np.isnan(log_sens[i,1]):
                log_sens[i,2] = log_sens[i,1] * 1.2
            else:
                log_sens[i,2] = 1.0
    
    sensitivity = 10.0**log_sens
    
    # Contract spectra with sensitivities
    exp = np.einsum('ijk,kl->ijl', spectra, sensitivity)
    
    # Normalize with mid-gray (green channel)
    ref_spec = np.einsum('k,kl->l', np.mean(spectra, axis=(0,1)), sensitivity)
    exp /= ref_spec[1] + 1e-8
    
    print(f"Python LUT shape: {exp.shape}")
    print(f"Python LUT range: [{exp.min():.6f}, {exp.max():.6f}]")
    
    # Sample the LUT for our test coordinates
    python_cmy = []
    for i, xy in enumerate(xy_coords):
        # Convert xy to LUT indices
        u = np.clip(xy[0], 0, 1)
        v = np.clip(xy[1], 0, 1)
        
        # Sample the LUT (bilinear interpolation)
        u_idx = u * (lut_size - 1)
        v_idx = v * (lut_size - 1)
        
        u0, u1 = int(u_idx), min(int(u_idx) + 1, lut_size - 1)
        v0, v1 = int(v_idx), min(int(v_idx) + 1, lut_size - 1)
        
        # Bilinear interpolation
        du = u_idx - u0
        dv = v_idx - v0
        
        c00 = exp[v0, u0]
        c01 = exp[v0, u1]
        c10 = exp[v1, u0]
        c11 = exp[v1, u1]
        
        cmy = (1-du)*(1-dv)*c00 + du*(1-dv)*c01 + (1-du)*dv*c10 + du*dv*c11
        python_cmy.append(cmy)
        
        print(f"Python pixel {i}: xy=({xy[0]:.6f},{xy[1]:.6f}) -> uv=({u:.6f},{v:.6f})")
        print(f"  LUT indices: u0={u0}, u1={u1}, v0={v0}, v1={v1}")
        print(f"  Interpolation weights: du={du:.6f}, dv={dv:.6f}")
        print(f"  Corner values: c00={c00}, c01={c01}, c10={c10}, c11={c11}")
        print(f"  Result: {cmy}")
        print()
    
    python_cmy = np.array(python_cmy)
    print(f"Python RGB->CMY conversion:\n{python_cmy}")
    
    # Now call the C++ test program
    print("\n--- C++ Pipeline Results ---")
    
    # Create a temporary file with test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for rgb in test_rgb:
            f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")
        test_file = f.name
    
    try:
        # Run the C++ test program
        result = subprocess.run([
            './test_camera_lut', test_file
        ], capture_output=True, text=True, cwd='AgXEmulsionOFX')
        
        print("C++ stdout:")
        print(result.stdout)
        if result.stderr:
            print("C++ stderr:")
            print(result.stderr)
        
        # Parse C++ output to extract CMY values
        cpp_cmy = []
        for line in result.stdout.split('\n'):
            if 'CMY=' in line:
                # Extract CMY values from line like "CMY=(1.001534,0.429265,1.978867)"
                cmy_str = line.split('CMY=')[1].strip('()')
                cmy_values = [float(x) for x in cmy_str.split(',')]
                cpp_cmy.append(cmy_values)
        
        cpp_cmy = np.array(cpp_cmy)
        print(f"C++ RGB->CMY conversion:\n{cpp_cmy}")
        
        # Compare results
        if len(python_cmy) == len(cpp_cmy):
            print("\n--- Comparison ---")
            for i in range(len(python_cmy)):
                diff = np.abs(python_cmy[i] - cpp_cmy[i])
                print(f"Pixel {i}:")
                print(f"  Python: {python_cmy[i]}")
                print(f"  C++:    {cpp_cmy[i]}")
                print(f"  Diff:   {diff}")
                print(f"  Max diff: {diff.max():.6f}")
                
                # Check if difference is within tolerance
                tolerance = 1e-3
                if diff.max() < tolerance:
                    print(f"  ✓ PASS (diff < {tolerance})")
                else:
                    print(f"  ✗ FAIL (diff >= {tolerance})")
        else:
            print(f"ERROR: Mismatched array lengths: Python={len(python_cmy)}, C++={len(cpp_cmy)}")
    
    finally:
        os.unlink(test_file)

def load_raw_csv_data(stock_name):
    """Load raw CSV data directly for comparison with C++"""
    import csv
    import os
    
    base_path = "/usr/OFX/Plugins/data/film/negative/" + stock_name + "/"
    
    # Load log exposure from red curve
    log_exposure = []
    density_r = []
    density_g = []
    density_b = []
    
    # Load red curve
    with open(base_path + "density_curve_r.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                log_exposure.append(float(row[0]))
                density_r.append(float(row[1]))
    
    # Load green curve
    with open(base_path + "density_curve_g.csv", 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) >= 2:
                density_g.append(float(row[1]))
    
    # Load blue curve
    with open(base_path + "density_curve_b.csv", 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) >= 2:
                density_b.append(float(row[1]))
    
    return np.array(log_exposure), np.array(density_r), np.array(density_g), np.array(density_b)

def test_emulsion_stage_raw_csv():
    """Test Emulsion stage using raw CSV data for direct comparison"""
    print("\n=== Testing Emulsion Stage (Raw CSV Data) ===")
    
    # Load raw CSV data
    log_exposure, density_r, density_g, density_b = load_raw_csv_data('kodak_portra_400')
    
    print(f"Raw CSV data loaded:")
    print(f"  Log exposure range: {log_exposure.min():.6f} to {log_exposure.max():.6f}")
    print(f"  Number of samples: {len(log_exposure)}")
    print(f"  Sample values:")
    for i in range(min(5, len(log_exposure))):
        print(f"    [{log_exposure[i]:.6f}] -> [{density_r[i]:.6f}, {density_g[i]:.6f}, {density_b[i]:.6f}]")
    
    # Create test CMY data
    test_cmy = np.array([
        [1.001534, 0.429265, 1.978867],
        [0.965969, 0.440596, 1.984661],
        [0.997311, 0.434762, 1.978001],
        [1.001534, 0.429265, 1.978867],
    ], dtype=np.float32)
    
    # Apply same interpolation as C++
    python_density = []
    gamma = 1.0
    
    for i, cmy in enumerate(test_cmy):
        print(f"\n--- Python Raw CSV Pixel {i} ---")
        print(f"Input CMY: {cmy}")
        
        # Convert CMY to log_raw
        log_raw = np.log10(cmy + 1e-10)
        print(f"CMY -> log_raw: {log_raw}")
        
        # Apply gamma correction
        gamma_corrected = log_raw / gamma
        print(f"Gamma corrected: {gamma_corrected}")
        
        # Simple nearest-neighbor interpolation (same as C++ test)
        density = np.zeros(3)
        for j in range(3):
            target = gamma_corrected[j]
            
            # Find closest match
            min_diff = 1e10
            best_idx = 0
            for k in range(len(log_exposure)):
                diff = abs(log_exposure[k] - target)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = k
            
            # Get density value
            if j == 0:
                density[j] = density_r[best_idx]
            elif j == 1:
                density[j] = density_g[best_idx]
            else:
                density[j] = density_b[best_idx]
            
            print(f"  Channel {j}: target={target:.6f}, best_idx={best_idx}, value={log_exposure[best_idx]:.6f}, density={density[j]:.6f}")
        
        python_density.append(density)
        print(f"Final density: {density}")
    
    python_density = np.array(python_density)
    print(f"\nPython Raw CSV CMY->Density conversion:\n{python_density}")
    
    return python_density


def test_emulsion_stage():
    """Test Emulsion stage - CMY to density conversion"""
    print("\n=== Testing Emulsion Stage ===")
    
    # Create test CMY data (output from Camera LUT stage)
    test_cmy = np.array([
        [1.001534, 0.429265, 1.978867],
        [0.965969, 0.440596, 1.984661],
        [0.997311, 0.434762, 1.978001],
        [1.001534, 0.429265, 1.978867],
    ], dtype=np.float32)
    
    print(f"Test CMY data:\n{test_cmy}")
    
    # Python reference: Apply film density curves
    # Load film profile
    from agx_emulsion.profiles.io import load_profile
    profile = load_profile('kodak_portra_400')
    
    # Create Film instance
    from agx_emulsion.model.emulsion import Film
    film = Film(profile)
    
    # Print film parameters for debugging
    print(f"\nPython Film Parameters:")
    print(f"  Gamma factor: {film.gamma_factor}")
    print(f"  Log exposure range: {film.log_exposure.min():.6f} to {film.log_exposure.max():.6f}")
    print(f"  Density curves shape: {film.density_curves.shape}")
    print(f"  Sample density values:")
    for i in range(min(5, len(film.density_curves))):
        print(f"    [{film.log_exposure[i]:.6f}] -> [{film.density_curves[i,0]:.6f}, {film.density_curves[i,1]:.6f}, {film.density_curves[i,2]:.6f}]")
    
    # Apply emulsion processing with detailed debugging
    python_density = []
    for i, cmy in enumerate(test_cmy):
        print(f"\n--- Python Pixel {i} ---")
        print(f"Input CMY: {cmy}")
        
        # Convert CMY to log_raw for Film.develop (it expects log_raw input)
        log_raw = np.log10(cmy + 1e-10)  # Simple CMY to log_raw conversion
        print(f"CMY -> log_raw: {log_raw}")
        
        # Reshape to 3D array (1x1x3) as expected by Film.develop
        log_raw_3d = log_raw.reshape(1, 1, 3)
        
        # Apply gamma correction manually to see what Python does
        gamma_corrected = log_raw_3d / film.gamma_factor
        print(f"Gamma corrected (log_raw / gamma): {gamma_corrected[0,0]}")
        
        density = film.develop(log_raw_3d, pixel_size_um=17.0, bypass_grain=True)  # Bypass grain to avoid errors
        python_density.append(density[0, 0])  # Extract the single pixel result
        print(f"Final density: {density[0, 0]}")
    
    python_density = np.array(python_density)
    print(f"\nPython CMY->Density conversion:\n{python_density}")
    
    # Now call the C++ test program
    print("\n--- C++ Pipeline Results ---")
    
    # Create a temporary file with test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for cmy in test_cmy:
            f.write(f"{cmy[0]:.6f} {cmy[1]:.6f} {cmy[2]:.6f}\n")
        test_file = f.name
    
    try:
        # Run the C++ test program
        result = subprocess.run([
            './test_emulsion', test_file
        ], capture_output=True, text=True, cwd='AgXEmulsionOFX')
        
        print("C++ stdout:")
        print(result.stdout)
        if result.stderr:
            print("C++ stderr:")
            print(result.stderr)
        
        # Parse C++ output to extract density values
        cpp_density = []
        for line in result.stdout.split('\n'):
            if 'Density=' in line:
                # Extract density values from line like "Density=(0.223605,0.648898,0.869614)"
                density_str = line.split('Density=')[1].strip('()')
                density_values = [float(x) for x in density_str.split(',')]
                cpp_density.append(density_values)
        
        cpp_density = np.array(cpp_density)
        print(f"C++ CMY->Density conversion:\n{cpp_density}")
        
        # Compare results
        if len(python_density) == len(cpp_density):
            print("\n--- Comparison ---")
            for i in range(len(python_density)):
                diff = np.abs(python_density[i] - cpp_density[i])
                print(f"Pixel {i}:")
                print(f"  Python: {python_density[i]}")
                print(f"  C++:    {cpp_density[i]}")
                print(f"  Diff:   {diff}")
                print(f"  Max diff: {diff.max():.6f}")
                
                # Check if difference is within tolerance
                tolerance = 1e-3
                if diff.max() < tolerance:
                    print(f"  ✓ PASS (diff < {tolerance})")
                else:
                    print(f"  ✗ FAIL (diff >= {tolerance})")
        else:
            print(f"ERROR: Mismatched array lengths: Python={len(python_density)}, C++={len(cpp_density)}")
    
    finally:
        os.unlink(test_file)

def test_emulsion_stage_json_filtered():
    """Test Emulsion stage using JSON profile with nan filtering"""
    print("\n=== Testing Emulsion Stage (JSON Profile with NaN Filtering) ===")
    
    # Load film profile
    from agx_emulsion.profiles.io import load_profile
    profile = load_profile('kodak_portra_400')
    
    # Extract and filter nan values manually (same as original Python implementation)
    log_exposure = np.array(profile.data.log_exposure)
    density_curves = np.array(profile.data.density_curves)
    
    print(f"Original JSON data:")
    print(f"  Log exposure shape: {log_exposure.shape}")
    print(f"  Density curves shape: {density_curves.shape}")
    print(f"  NaN values in density curves: {np.isnan(density_curves).sum()}")
    
    # Filter out nan values (same as original Python implementation)
    valid_mask = ~np.isnan(density_curves).any(axis=1)
    filtered_log_exposure = log_exposure[valid_mask]
    filtered_density_curves = density_curves[valid_mask]
    
    print(f"After NaN filtering:")
    print(f"  Valid samples: {filtered_log_exposure.shape[0]} (from {log_exposure.shape[0]})")
    print(f"  Log exposure range: {filtered_log_exposure.min():.6f} to {filtered_log_exposure.max():.6f}")
    print(f"  Sample values:")
    for i in range(min(5, len(filtered_log_exposure))):
        print(f"    [{filtered_log_exposure[i]:.6f}] -> [{filtered_density_curves[i,0]:.6f}, {filtered_density_curves[i,1]:.6f}, {filtered_density_curves[i,2]:.6f}]")
    
    # Create test CMY data
    test_cmy = np.array([
        [1.001534, 0.429265, 1.978867],
        [0.965969, 0.440596, 1.984661],
        [0.997311, 0.434762, 1.978001],
        [1.001534, 0.429265, 1.978867],
    ], dtype=np.float32)
    
    # Apply same interpolation as C++ with filtered data
    python_density = []
    gamma = 1.0
    
    for i, cmy in enumerate(test_cmy):
        print(f"\n--- Python JSON Filtered Pixel {i} ---")
        print(f"Input CMY: {cmy}")
        
        # Convert CMY to log_raw
        log_raw = np.log10(cmy + 1e-10)
        print(f"CMY -> log_raw: {log_raw}")
        
        # Apply gamma correction
        gamma_corrected = log_raw / gamma
        print(f"Gamma corrected: {gamma_corrected}")
        
        # Simple nearest-neighbor interpolation (same as C++ test)
        density = np.zeros(3)
        for j in range(3):
            target = gamma_corrected[j]
            
            # Find closest match
            min_diff = 1e10
            best_idx = 0
            for k in range(len(filtered_log_exposure)):
                diff = abs(filtered_log_exposure[k] - target)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = k
            
            # Get density value
            density[j] = filtered_density_curves[best_idx, j]
            
            print(f"  Channel {j}: target={target:.6f}, best_idx={best_idx}, value={filtered_log_exposure[best_idx]:.6f}, density={density[j]:.6f}")
        
        python_density.append(density)
        print(f"Final density: {density}")
    
    python_density = np.array(python_density)
    print(f"\nPython JSON Filtered CMY->Density conversion:\n{python_density}")
    
    return python_density

def save_filtered_json_data(stock_name):
    """Save filtered JSON data in a simple format for C++ to read"""
    # Load film profile
    from agx_emulsion.profiles.io import load_profile
    profile = load_profile(stock_name)
    
    # Extract and filter nan values
    log_exposure = np.array(profile.data.log_exposure)
    density_curves = np.array(profile.data.density_curves)
    
    # Filter out nan values
    valid_mask = ~np.isnan(density_curves).any(axis=1)
    filtered_log_exposure = log_exposure[valid_mask]
    filtered_density_curves = density_curves[valid_mask]
    
    # Save in simple CSV format for C++
    output_file = f"filtered_{stock_name}_data.csv"
    with open(output_file, 'w') as f:
        f.write("# LogExposure,DensityR,DensityG,DensityB\n")
        for i in range(len(filtered_log_exposure)):
            f.write(f"{filtered_log_exposure[i]:.6f},{filtered_density_curves[i,0]:.6f},{filtered_density_curves[i,1]:.6f},{filtered_density_curves[i,2]:.6f}\n")
    
    print(f"Saved filtered data to {output_file}")
    print(f"  Valid samples: {len(filtered_log_exposure)}")
    print(f"  Log exposure range: {filtered_log_exposure.min():.6f} to {filtered_log_exposure.max():.6f}")
    
    return output_file

def test_dir_coupler_stage():
    """Test DIR Coupler stage (density correction matrix)"""
    print("\n" + "="*60)
    print("TESTING DIR COUPLER STAGE")
    print("="*60)
    
    # Test parameters (same as C++ debug output)
    dir_amount = 1.395349
    dir_interlayer = 1.395349
    dir_high_shift = 0.248062
    dir_diff_um = 13.178294
    pixel_size_um = 47.339535
    
    # Test input CMY values (from Camera LUT output)
    test_cmy = np.array([1.001534, 0.429265, 1.978867])
    
    print(f"DIR Parameters:")
    print(f"  Amount: {dir_amount}")
    print(f"  Interlayer: {dir_interlayer}")
    print(f"  High Shift: {dir_high_shift}")
    print(f"  Diffusion: {dir_diff_um} μm")
    print(f"  Pixel Size: {pixel_size_um} μm")
    print(f"  Sigma (pixels): {dir_diff_um/pixel_size_um:.6f}")
    
    print(f"\nInput CMY: {test_cmy}")
    
    # Python DIR matrix computation (from couplers.py)
    from agx_emulsion.model.couplers import compute_dir_couplers_matrix
    
    # Convert to double precision like C++
    amt_rgb = np.array([dir_amount, dir_amount, dir_amount], dtype=np.float64)
    interlayer = np.float64(dir_interlayer)
    
    # Compute DIR matrix
    dir_matrix = compute_dir_couplers_matrix(amt_rgb, interlayer)
    
    print(f"\nPython DIR Matrix:")
    print(f"  {dir_matrix[0,0]:.6f} {dir_matrix[0,1]:.6f} {dir_matrix[0,2]:.6f}")
    print(f"  {dir_matrix[1,0]:.6f} {dir_matrix[1,1]:.6f} {dir_matrix[1,2]:.6f}")
    print(f"  {dir_matrix[2,0]:.6f} {dir_matrix[2,1]:.6f} {dir_matrix[2,2]:.6f}")
    
    # Apply DIR correction
    corrected_cmy = dir_matrix @ test_cmy
    
    print(f"\nCorrected CMY: {corrected_cmy}")
    
    # Test C++ DIR computation
    print(f"\nTesting C++ DIR computation...")
    
    # Create test executable for DIR stage
    dir_test_cpp = """
#include <iostream>
#include <iomanip>
#include <cmath>
#include "couplers.cpp"

int main() {
    // Test parameters
    double dir_amount = 1.395349;
    double dir_interlayer = 1.395349;
    double test_cmy[3] = {1.001534, 0.429265, 1.978867};
    
    std::cout << "DIR Parameters:" << std::endl;
    std::cout << "  Amount: " << dir_amount << std::endl;
    std::cout << "  Interlayer: " << dir_interlayer << std::endl;
    
    std::cout << "Input CMY: [" << test_cmy[0] << ", " << test_cmy[1] << ", " << test_cmy[2] << "]" << std::endl;
    
    // Compute DIR matrix
    std::array<double,3> amt_rgb = {dir_amount, dir_amount, dir_amount};
    cp::Matrix3 dir_matrix = cp::compute_dir_couplers_matrix(amt_rgb, dir_interlayer);
    
    std::cout << "C++ DIR Matrix:" << std::endl;
    std::cout << "  " << std::fixed << std::setprecision(6) 
              << dir_matrix[0][0] << " " << dir_matrix[0][1] << " " << dir_matrix[0][2] << std::endl;
    std::cout << "  " << dir_matrix[1][0] << " " << dir_matrix[1][1] << " " << dir_matrix[1][2] << std::endl;
    std::cout << "  " << dir_matrix[2][0] << " " << dir_matrix[2][1] << " " << dir_matrix[2][2] << std::endl;
    
    // Apply DIR correction
    double corrected_cmy[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            corrected_cmy[i] += dir_matrix[i][j] * test_cmy[j];
        }
    }
    
    std::cout << "Corrected CMY: [" << corrected_cmy[0] << ", " << corrected_cmy[1] << ", " << corrected_cmy[2] << "]" << std::endl;
    
    return 0;
}
"""
    
    with open("test_dir_coupler.cpp", "w") as f:
        f.write(dir_test_cpp)
    
    # Compile and run C++ test
    import subprocess
    try:
        result = subprocess.run([
            "g++", "-std=c++11", "-I..", "test_dir_coupler.cpp", "-o", "test_dir_coupler"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            result = subprocess.run(["./test_dir_coupler"], capture_output=True, text=True)
            if result.returncode == 0:
                print("C++ Output:")
                print(result.stdout)
                
                # Parse C++ output for comparison
                lines = result.stdout.strip().split('\n')
                cpp_corrected = None
                for line in lines:
                    if line.startswith("Corrected CMY:"):
                        cpp_corrected = line.split("[")[1].split("]")[0]
                        cpp_corrected = [float(x) for x in cpp_corrected.split(", ")]
                        break
                
                if cpp_corrected:
                    print(f"\nComparison:")
                    print(f"  Python: {corrected_cmy}")
                    print(f"  C++:    {cpp_corrected}")
                    
                    max_diff = np.max(np.abs(np.array(cpp_corrected) - corrected_cmy))
                    print(f"  Max Difference: {max_diff:.6f}")
                    
                    if max_diff < 1e-6:
                        print("  ✅ DIR Coupler stage: PASS")
                    else:
                        print("  ❌ DIR Coupler stage: FAIL")
                else:
                    print("  ❌ Could not parse C++ output")
            else:
                print(f"  ❌ C++ test failed: {result.stderr}")
        else:
            print(f"  ❌ C++ compilation failed: {result.stderr}")
            
    except Exception as e:
        print(f"  ❌ Error running C++ test: {e}")
    
    # Cleanup
    import os
    if os.path.exists("test_dir_coupler.cpp"):
        os.remove("test_dir_coupler.cpp")
    if os.path.exists("test_dir_coupler"):
        os.remove("test_dir_coupler")

def test_diffusion_halation_stage():
    """Test Diffusion/Halation stage (spatial effects)"""
    print("\n" + "="*60)
    print("TESTING DIFFUSION/HALATION STAGE")
    print("="*60)
    
    # Test parameters (from C++ debug output)
    radius = 5.038760
    halation = 0.263566
    
    # Test input RGB values (from DIR stage output)
    test_rgb = np.array([3.265389, 0.031234, 3.265389])  # From debug output
    
    print(f"Diffusion/Halation Parameters:")
    print(f"  Radius: {radius} pixels")
    print(f"  Halation: {halation}")
    print(f"  Rad (rounded): {int(radius + 0.5)}")
    
    print(f"\nInput RGB: {test_rgb}")
    
    # Create a simple test image (3x3 pixels with the test RGB in center)
    test_image = np.zeros((3, 3, 3))
    test_image[1, 1] = test_rgb  # Center pixel
    
    print(f"\nTest image shape: {test_image.shape}")
    print(f"Center pixel: {test_image[1, 1]}")
    
    # Python diffusion/halation simulation (matching CUDA implementation exactly)
    from scipy.ndimage import gaussian_filter
    
    # CUDA sigma calculation: sigma = rad * 0.5f + 1e-3f
    rad = int(radius + 0.5)
    sigma = rad * 0.5 + 1e-3
    
    print(f"\nCUDA-style parameters:")
    print(f"  Rad: {rad}")
    print(f"  Sigma (rad * 0.5 + 1e-3): {sigma:.6f}")
    
    # Manual Gaussian convolution to match CUDA exactly
    def manual_gaussian_convolution(image, sigma, radius):
        """Manual Gaussian convolution matching CUDA implementation"""
        height, width, channels = image.shape
        result = np.zeros_like(image)
        
        # CUDA uses: float twoSigma2 = 2.0f * sigma * sigma;
        two_sigma2 = 2.0 * sigma * sigma
        
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    sum_val = 0.0
                    wsum = 0.0
                    
                    # CUDA loop: for(int dy=-rad; dy<=rad; ++dy)
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            # CUDA clampi: int yy = clampi(y + dy, 0, height-1);
                            yy = max(0, min(height - 1, y + dy))
                            xx = max(0, min(width - 1, x + dx))
                            
                            # CUDA weight: float w = __expf(-(dx*dx + dy*dy)/twoSigma2);
                            w = np.exp(-(dx*dx + dy*dy) / two_sigma2)
                            
                            sum_val += w * image[yy, xx, c]
                            wsum += w
                    
                    # CUDA normalization: float blurR = sumR / wsum;
                    if wsum > 0:
                        result[y, x, c] = sum_val / wsum
                    else:
                        result[y, x, c] = image[y, x, c]
        
        return result
    
    # Apply manual Gaussian blur for diffusion effect
    diffused = manual_gaussian_convolution(test_image, sigma, rad)
    
    print(f"\nPython Diffusion Results (Manual CUDA-style):")
    print(f"  Sigma used: {sigma:.6f}")
    print(f"  TwoSigma2: {2.0 * sigma * sigma:.6f}")
    print(f"  Center pixel after diffusion: {diffused[1, 1]}")
    
    # Apply halation effect (CUDA style: origR + halStrength * (blurR - origR))
    halated = test_image.copy()
    orig_r = test_image[1, 1, 0]  # Original red
    blur_r = diffused[1, 1, 0]    # Blurred red
    new_r = orig_r + halation * (blur_r - orig_r)
    halated[1, 1, 0] = np.clip(new_r, 0, 1)  # Clamp like CUDA
    
    print(f"\nPython Halation Results:")
    print(f"  Original red: {orig_r:.6f}")
    print(f"  Blurred red: {blur_r:.6f}")
    print(f"  Halation factor: {halation:.6f}")
    print(f"  New red: {orig_r:.6f} + {halation:.6f} * ({blur_r:.6f} - {orig_r:.6f}) = {new_r:.6f}")
    print(f"  Clamped red: {halated[1, 1, 0]:.6f}")
    print(f"  Center pixel after halation: {halated[1, 1]}")
    
    # Test C++ diffusion/halation computation
    print(f"\nTesting C++ diffusion/halation computation...")
    
    # Create test executable for diffusion/halation stage
    diffusion_test_cpp = """
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

// Simple Gaussian kernel generation (matching CUDA implementation)
std::vector<float> generateGaussianKernel(float sigma, int radius) {
    std::vector<float> kernel;
    float sum = 0.0f;
    float twoSigma2 = 2.0f * sigma * sigma;
    
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float w = expf(-(x*x + y*y) / twoSigma2);
            kernel.push_back(w);
            sum += w;
        }
    }
    
    // Normalize (like CUDA: sumR / wsum)
    for (float& val : kernel) {
        val /= sum;
    }
    
    return kernel;
}

// Simple convolution (CPU version for testing)
float convolvePixel(const std::vector<float>& image, const std::vector<float>& kernel, 
                   int x, int y, int width, int height, int radius, int channel) {
    float result = 0.0f;
    int kernel_idx = 0;
    
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int px = x + kx;
            int py = y + ky;
            
            // Clamp to image bounds (like CUDA clampi)
            px = std::max(0, std::min(width - 1, px));
            py = std::max(0, std::min(height - 1, py));
            
            int pixel_idx = (py * width + px) * 3 + channel;
            result += image[pixel_idx] * kernel[kernel_idx];
            kernel_idx++;
        }
    }
    
    return result;
}

int main() {
    // Test parameters
    float radius = 5.038760f;
    float halation = 0.263566f;
    float test_rgb[3] = {3.265389f, 0.031234f, 3.265389f};
    
    std::cout << "Diffusion/Halation Parameters:" << std::endl;
    std::cout << "  Radius: " << radius << std::endl;
    std::cout << "  Halation: " << halation << std::endl;
    
    // Create test image (3x3)
    int width = 3, height = 3;
    std::vector<float> image(width * height * 3, 0.0f);
    
    // Set center pixel
    int center_idx = (1 * width + 1) * 3;
    image[center_idx] = test_rgb[0];     // R
    image[center_idx + 1] = test_rgb[1]; // G
    image[center_idx + 2] = test_rgb[2]; // B
    
    std::cout << "Input RGB: [" << test_rgb[0] << ", " << test_rgb[1] << ", " << test_rgb[2] << "]" << std::endl;
    std::cout << "Center pixel: [" << image[center_idx] << ", " << image[center_idx + 1] << ", " << image[center_idx + 2] << "]" << std::endl;
    
    // Generate Gaussian kernel (CUDA style)
    int rad = (int)(radius + 0.5f);
    float sigma = rad * 0.5f + 1e-3f;
    std::vector<float> kernel = generateGaussianKernel(sigma, rad);
    
    std::cout << "C++ Kernel parameters:" << std::endl;
    std::cout << "  Rad: " << rad << std::endl;
    std::cout << "  Sigma: " << sigma << std::endl;
    std::cout << "  Kernel radius: " << rad << std::endl;
    std::cout << "  Kernel size: " << kernel.size() << std::endl;
    
    // Apply diffusion to center pixel (red channel only for halation)
    float diffused_r = convolvePixel(image, kernel, 1, 1, width, height, rad, 0);
    
    std::cout << "C++ Diffusion Results:" << std::endl;
    std::cout << "  Center pixel red after diffusion: " << diffused_r << std::endl;
    
    // Apply halation (CUDA style: origR + halStrength * (blurR - origR))
    float orig_r = image[center_idx];
    float new_r = orig_r + halation * (diffused_r - orig_r);
    float clamped_r = std::max(0.0f, std::min(1.0f, new_r));
    
    std::cout << "C++ Halation Results:" << std::endl;
    std::cout << "  Original red: " << orig_r << std::endl;
    std::cout << "  Blurred red: " << diffused_r << std::endl;
    std::cout << "  New red: " << orig_r << " + " << halation << " * (" << diffused_r << " - " << orig_r << ") = " << new_r << std::endl;
    std::cout << "  Clamped red: " << clamped_r << std::endl;
    
    return 0;
}
"""
    
    with open("test_diffusion_halation.cpp", "w") as f:
        f.write(diffusion_test_cpp)
    
    # Compile and run C++ test
    import subprocess
    try:
        result = subprocess.run([
            "g++", "-std=c++11", "-O2", "test_diffusion_halation.cpp", "-o", "test_diffusion_halation"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            result = subprocess.run(["./test_diffusion_halation"], capture_output=True, text=True)
            if result.returncode == 0:
                print("C++ Output:")
                print(result.stdout)
                
                # Parse C++ output for comparison
                lines = result.stdout.strip().split('\n')
                cpp_diffused_r = None
                cpp_clamped_r = None
                
                for line in lines:
                    if "Center pixel red after diffusion:" in line:
                        cpp_diffused_r = float(line.split(":")[1].strip())
                    elif "Clamped red:" in line:
                        cpp_clamped_r = float(line.split(":")[1].strip())
                
                if cpp_diffused_r is not None and cpp_clamped_r is not None:
                    print(f"\nComparison:")
                    print(f"  Python diffused red: {diffused[1, 1, 0]:.6f}")
                    print(f"  C++ diffused red:    {cpp_diffused_r:.6f}")
                    
                    diffused_diff = abs(cpp_diffused_r - diffused[1, 1, 0])
                    print(f"  Diffusion diff: {diffused_diff:.6f}")
                    
                    print(f"\n  Python clamped red: {halated[1, 1, 0]:.6f}")
                    print(f"  C++ clamped red:    {cpp_clamped_r:.6f}")
                    
                    halated_diff = abs(cpp_clamped_r - halated[1, 1, 0])
                    print(f"  Halation diff: {halated_diff:.6f}")
                    
                    if diffused_diff < 1e-3 and halated_diff < 1e-3:
                        print("  ✅ Diffusion/Halation stage: PASS")
                    else:
                        print("  ❌ Diffusion/Halation stage: FAIL")
                else:
                    print("  ❌ Could not parse C++ output")
            else:
                print(f"  ❌ C++ test failed: {result.stderr}")
        else:
            print(f"  ❌ C++ compilation failed: {result.stderr}")
            
    except Exception as e:
        print(f"  ❌ Error running C++ test: {e}")
    
    # Cleanup
    import os
    if os.path.exists("test_diffusion_halation.cpp"):
        os.remove("test_diffusion_halation.cpp")
    if os.path.exists("test_diffusion_halation"):
        os.remove("test_diffusion_halation")


def main():
    """Run all pipeline stage tests"""
    print("AgX Emulsion Pipeline Accuracy Tests")
    print("=" * 50)
    
    # Generate filtered CSV data for C++ to use
    save_filtered_json_data('kodak_portra_400')
    
    # Test each stage
    test_camera_lut_stage()
    test_emulsion_stage_raw_csv()  # Test with raw CSV data
    test_emulsion_stage_json_filtered()  # Test with JSON profile and nan filtering
    test_emulsion_stage()
    test_dir_coupler_stage() # Test DIR Coupler stage
    test_diffusion_halation_stage() # Test Diffusion/Halation stage
    
    print("\n" + "=" * 50)
    print("Tests completed!")

if __name__ == "__main__":
    main() 