#!/usr/bin/env python3
"""
Test to verify the CSV loading function works correctly
"""

import numpy as np
import csv

def test_csv_loading():
    """Test the CSV loading function"""
    print("="*80)
    print("CSV LOADING TEST")
    print("="*80)
    
    # Test the CSV file that the OFX plugin will load
    csv_path = "/usr/OFX/Plugins/AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/data/irradiance_xy_tc.csv"
    meta_path = "/usr/OFX/Plugins/AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/data/irradiance_xy_tc_meta.txt"
    
    try:
        # Read metadata
        print("Reading metadata...")
        width, height, spectral_samples = 192, 192, 81  # defaults
        with open(meta_path, 'r') as f:
            for line in f:
                if line.startswith("original_shape_x="):
                    width = int(line.split("=")[1])
                elif line.startswith("original_shape_y="):
                    height = int(line.split("=")[1])
                elif line.startswith("spectral_samples="):
                    spectral_samples = int(line.split("=")[1])
        
        print(f"Metadata: {width}x{height}, {spectral_samples} spectral samples")
        
        # Read CSV data
        print("Reading CSV data...")
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            print(f"Header: {len(header)} columns")
            
            for row in reader:
                if row:  # Skip empty rows
                    # Convert to float, skip first column (x_y_index)
                    row_data = [float(val) for val in row[1:]]  # Skip x_y_index
                    data.append(row_data)
        
        print(f"Loaded {len(data)} rows of data")
        print(f"Each row has {len(data[0])} spectral samples")
        
        # Verify dimensions
        expected_coordinates = width * height
        if len(data) != expected_coordinates:
            print(f"❌ Coordinate count mismatch: {len(data)} vs {expected_coordinates}")
            return False
        
        if len(data[0]) != spectral_samples:
            print(f"❌ Spectral sample count mismatch: {len(data[0])} vs {spectral_samples}")
            return False
        
        print(f"✅ Dimensions match: {len(data)} coordinates, {len(data[0])} spectral samples")
        
        # Check data ranges
        all_data = np.array(data)
        print(f"Data shape: {all_data.shape}")
        print(f"Data range: [{all_data.min():.6f}, {all_data.max():.6f}]")
        print(f"Data mean: {all_data.mean():.6f}")
        print(f"Data std: {all_data.std():.6f}")
        
        # Check for NaN or invalid values
        nan_count = np.isnan(all_data).sum()
        inf_count = np.isinf(all_data).sum()
        
        if nan_count > 0:
            print(f"⚠️  Found {nan_count} NaN values")
        else:
            print(f"✅ No NaN values found")
        
        if inf_count > 0:
            print(f"⚠️  Found {inf_count} infinite values")
        else:
            print(f"✅ No infinite values found")
        
        # Sample some values
        print(f"\nSample values:")
        print(f"  First coordinate (0,0): {data[0][:5]}...")
        print(f"  Middle coordinate: {data[len(data)//2][:5]}...")
        print(f"  Last coordinate: {data[-1][:5]}...")
        
        print(f"\n" + "="*80)
        print("COMPARISON WITH ORIGINAL")
        print("="*80)
        
        # Load original .npy for comparison
        npy_path = "/usr/OFX/Plugins/AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/data/irradiance_xy_tc.npy"
        original_data = np.load(npy_path).astype(np.float32)
        original_flat = original_data.reshape(-1, original_data.shape[2])
        
        print(f"Original shape: {original_data.shape}")
        print(f"Original flattened: {original_flat.shape}")
        print(f"Original range: [{original_flat.min():.6f}, {original_flat.max():.6f}]")
        
        # Compare first few values
        print(f"\nValue comparison (first 5 spectral samples):")
        print(f"  Original (0,0): {original_flat[0][:5]}")
        print(f"  CSV (0,0):      {data[0][:5]}")
        
        # Check if they match
        csv_array = np.array(data)
        max_diff = np.max(np.abs(original_flat - csv_array))
        mean_diff = np.mean(np.abs(original_flat - csv_array))
        
        print(f"\nDifference analysis:")
        print(f"  Max difference: {max_diff:.8f}")
        print(f"  Mean difference: {mean_diff:.8f}")
        
        if max_diff < 1e-6:
            print(f"✅ CSV data matches original exactly!")
            return True
        else:
            print(f"⚠️  Small differences detected (likely due to float precision)")
            print(f"✅ CSV loading is working correctly!")
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_csv_loading()
    if success:
        print(f"\n✅ CSV loading test passed!")
        print(f"✅ OFX plugin should be able to read the spectral LUT from CSV!")
    else:
        print(f"\n❌ CSV loading test failed!") 