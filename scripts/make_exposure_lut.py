#!/usr/bin/env python3
"""Generate exposure_lut.bin for CameraLUTKernel
Based on the Python reference spectral up-sampling utilities.
Writes a W×H×3 float16 binary (row-major, R index varying fastest).
"""
import json, pathlib, struct, sys, numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'ref'))
from agx_emulsion.utils.spectral_upsampling import compute_lut_spectra

LUT_SIZE = 128
FILM_PROFILE = 'kodak_portra_400'  # default negative used in plugin today
DATA_DIR = pathlib.Path(__file__).resolve().parents[1]/'ref/agx_emulsion/data/profiles'
OUT_PATH = pathlib.Path(__file__).resolve().parents[1]/'AgXEmulsionOFX'/ 'data'/ 'exposure_lut.bin'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print('Generating spectral LUT…')
# 1. Get spectra (shape LUT_SIZE×LUT_SIZE×441)
spectra = compute_lut_spectra(lut_size=LUT_SIZE, smooth_steps=1).astype(np.float32)  # half later
print(f'Spectra shape: {spectra.shape}, min: {spectra.min()}, max: {spectra.max()}')

print('Loading film profile:', FILM_PROFILE)
profile_file = DATA_DIR/f'{FILM_PROFILE}.json'
with profile_file.open('r') as f:
    prof = json.load(f)
log_sens = np.array(prof['data']['log_sensitivity'], dtype=np.float32)  # shape (441,3)

# Handle NaN values in sensitivity data
print(f'Original sensitivity shape: {log_sens.shape}')
print(f'NaN count in log_sens: {np.isnan(log_sens).sum()}')
print(f'NaN count per channel: R={np.isnan(log_sens[:,0]).sum()}, G={np.isnan(log_sens[:,1]).sum()}, B={np.isnan(log_sens[:,2]).sum()}')

# Replace NaN values with reasonable defaults
# For missing red channel data, use a simple approximation
for i in range(len(log_sens)):
    if np.isnan(log_sens[i,0]):  # Red channel NaN
        # Use a simple approximation based on green channel or default value
        if not np.isnan(log_sens[i,1]):  # Green channel available
            log_sens[i,0] = log_sens[i,1] * 0.8  # Red typically less sensitive than green
        else:
            log_sens[i,0] = 1.0  # Default value
    
    if np.isnan(log_sens[i,1]):  # Green channel NaN
        if not np.isnan(log_sens[i,0]) and not np.isnan(log_sens[i,2]):
            log_sens[i,1] = (log_sens[i,0] + log_sens[i,2]) / 2  # Average of R and B
        else:
            log_sens[i,1] = 1.0  # Default value
    
    if np.isnan(log_sens[i,2]):  # Blue channel NaN
        if not np.isnan(log_sens[i,1]):  # Green channel available
            log_sens[i,2] = log_sens[i,1] * 1.2  # Blue typically more sensitive than green
        else:
            log_sens[i,2] = 1.0  # Default value

print(f'After NaN handling - NaN count: {np.isnan(log_sens).sum()}')

sensitivity = 10.0**log_sens
print(f'Sensitivity shape: {sensitivity.shape}, min: {sensitivity.min()}, max: {sensitivity.max()}')

print('Contracting spectra with sensitivities…')
# einsum ijk,lk->ijl where k wavelengths
exp = np.einsum('ijk,kl->ijl', spectra, sensitivity)  # shape LUT×LUT×3
print(f'Exp shape: {exp.shape}, min: {exp.min()}, max: {exp.max()}')
print(f'Exp has NaN: {np.isnan(exp).any()}, Inf: {np.isinf(exp).any()}')

# normalise with mid-gray (green channel)
midgray_rgb = np.array([0.184,0.184,0.184], dtype=np.float32)
ref_spec = np.einsum('k,kl->l', np.mean(spectra, axis=(0,1)), sensitivity)  # crude ref
print(f'Ref spec: {ref_spec}')
exp /= ref_spec[1] + 1e-8
print(f'After norm - Exp min: {exp.min()}, max: {exp.max()}')
print(f'After norm - Exp has NaN: {np.isnan(exp).any()}, Inf: {np.isinf(exp).any()}')

# Check first few values
print(f'First few values:')
for i in range(3):
    for j in range(3):
        print(f'  exp[{i},{j}] = {exp[i,j]}')

exp = exp.astype(np.float16)
print(f'After float16 - min: {exp.min()}, max: {exp.max()}')
print(f'After float16 - has NaN: {np.isnan(exp).any()}, Inf: {np.isinf(exp).any()}')

print('Writing binary:', OUT_PATH)
with OUT_PATH.open('wb') as f:
    f.write(exp.tobytes())
print('Done.') 