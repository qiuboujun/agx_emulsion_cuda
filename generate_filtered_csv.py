#!/usr/bin/env python3
"""
Generate filtered CSV files for all film stocks from JSON profiles.
This script loads JSON profiles, filters out nan values, and saves them as CSV files
that the C++/CUDA OFX plugin can easily load.
"""

import os
import sys
import numpy as np

# Add the ref directory to the path so we can import agx_emulsion
sys.path.insert(0, 'ref')

def load_and_filter_profile(stock_name):
    """Load a film profile and filter out nan values"""
    try:
        from agx_emulsion.profiles.io import load_profile
        profile = load_profile(stock_name)
        
        # Extract data
        log_exposure = np.array(profile.data.log_exposure)
        density_curves = np.array(profile.data.density_curves)
        
        # Filter out nan values
        valid_mask = ~np.isnan(density_curves).any(axis=1)
        filtered_log_exposure = log_exposure[valid_mask]
        filtered_density_curves = density_curves[valid_mask]
        
        return filtered_log_exposure, filtered_density_curves
        
    except Exception as e:
        print(f"ERROR loading {stock_name}: {e}")
        return None, None

def save_filtered_csv(stock_name, log_exposure, density_curves):
    """Save filtered data as CSV file"""
    output_file = f"filtered_{stock_name}_data.csv"
    
    with open(output_file, 'w') as f:
        f.write("# LogExposure,DensityR,DensityG,DensityB\n")
        for i in range(len(log_exposure)):
            f.write(f"{log_exposure[i]:.6f},{density_curves[i,0]:.6f},{density_curves[i,1]:.6f},{density_curves[i,2]:.6f}\n")
    
    print(f"  Saved: {output_file} ({len(log_exposure)} samples)")
    return output_file

def main():
    """Generate filtered CSV files for all available film stocks"""
    print("Generating filtered CSV files for film stocks...")
    
    # List of film stocks to process
    film_stocks = [
        'kodak_portra_400',
        'kodak_portra_160',
        'kodak_portra_800',
        'kodak_ektar_100',
        'kodak_gold_200',
        'kodak_ultramax_400',
        'kodak_vision3_50d',
        'kodak_vision3_200t',
        'kodak_vision3_250d',
        'kodak_vision3_500t',
        'fujifilm_c200',
        'fujifilm_pro_400h',
        'fujifilm_xtra_400',
        'fujifilm_provia_100f',
    ]
    
    successful = 0
    failed = 0
    
    for stock in film_stocks:
        print(f"\nProcessing {stock}...")
        
        log_exposure, density_curves = load_and_filter_profile(stock)
        
        if log_exposure is not None and density_curves is not None:
            save_filtered_csv(stock, log_exposure, density_curves)
            successful += 1
        else:
            print(f"  FAILED: Could not load {stock}")
            failed += 1
    
    print(f"\nSummary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {successful + failed}")
    
    if successful > 0:
        print(f"\nFiltered CSV files have been generated in the current directory.")
        print(f"Copy them to /usr/OFX/Plugins/data/ for the OFX plugin to use them.")
        
        # Create a script to copy files to the plugin directory
        copy_script = "copy_filtered_csv.sh"
        with open(copy_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Script to copy filtered CSV files to OFX plugin directory\n")
            f.write("sudo mkdir -p /usr/OFX/Plugins/data/\n")
            f.write("sudo cp filtered_*_data.csv /usr/OFX/Plugins/data/\n")
            f.write("echo 'Filtered CSV files copied to /usr/OFX/Plugins/data/'\n")
        
        os.chmod(copy_script, 0o755)
        print(f"Created {copy_script} to copy files to plugin directory.")

if __name__ == "__main__":
    main() 