#!/usr/bin/env python3
"""
Convert spectral LUT from NumPy .npy format to CSV for C++ compatibility
"""

import numpy as np
import csv

def convert_spectral_lut():
    """Convert spectral LUT from .npy to .csv"""
    
    # Load the NumPy array
    npy_path = "/usr/OFX/Plugins/AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/data/irradiance_xy_tc.npy"
    csv_path = "/usr/OFX/Plugins/AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/data/irradiance_xy_tc.csv"
    
    print("Loading NumPy array...")
    data = np.load(npy_path)
    
    print(f"Array shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Total elements: {data.size}")
    
    # Convert to float32 for better precision in C++
    data = data.astype(np.float32)
    
    # Flatten the 3D array to 2D: (192*192, 81)
    # Each row represents one (x,y) coordinate, columns are spectral samples
    flattened = data.reshape(-1, data.shape[2])
    
    print(f"Flattened shape: {flattened.shape}")
    print(f"First row (first 10 values): {flattened[0][:10]}")
    
    # Save as CSV
    print(f"Saving to {csv_path}...")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header with spectral sample indices
        header = ['x_y_index'] + [f'spectral_{i}' for i in range(flattened.shape[1])]
        writer.writerow(header)
        
        # Write data rows
        for i, row in enumerate(flattened):
            writer.writerow([i] + row.tolist())
    
    print(f"✅ Successfully converted to CSV!")
    print(f"📁 File: {csv_path}")
    print(f"📊 Rows: {flattened.shape[0]} (192×192 = 36,864 x,y coordinates)")
    print(f"📊 Columns: {flattened.shape[1]} (81 spectral samples)")
    
    # Also save metadata for C++ to understand the structure
    meta_path = "/usr/OFX/Plugins/AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/data/irradiance_xy_tc_meta.txt"
    with open(meta_path, 'w') as f:
        f.write(f"original_shape_x={data.shape[0]}\n")
        f.write(f"original_shape_y={data.shape[1]}\n")
        f.write(f"spectral_samples={data.shape[2]}\n")
        f.write(f"total_coordinates={flattened.shape[0]}\n")
        f.write(f"data_type=float32\n")
    
    print(f"📄 Metadata saved to: {meta_path}")
    
    return True

if __name__ == "__main__":
    success = convert_spectral_lut()
    if success:
        print("\n🎉 Spectral LUT conversion complete!")
        print("C++ can now read the CSV file for spectral upsampling.")
    else:
        print("\n❌ Conversion failed!") 