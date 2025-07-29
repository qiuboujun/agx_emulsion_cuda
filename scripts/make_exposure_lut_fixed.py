#!/usr/bin/env python3
"""Generate exposure_lut.bin for CameraLUTKernel - FIXED VERSION
Based on the Python reference spectral up-sampling utilities.
Writes a W×H×3 float16 binary (row-major, R index varying fastest).
"""
import json, pathlib, struct, sys, numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'ref'))
from agx_emulsion.utils.spectral_upsampling import compute_lut_spectra
from agx_emulsion.model.color_filters import compute_band_pass_filter

LUT_SIZE = 128
FILM_PROFILE = 'kodak_portra_400'  # default negative used in plugin today
DATA_DIR = pathlib.Path(__file__).resolve().parents[1]/'ref/agx_emulsion/data/profiles'
OUT_PATH = pathlib.Path(__file__).resolve().parents[1]/'AgXEmulsionOFX'/ 'data'/ 'exposure_lut.bin'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print('Generating spectral LUT (FIXED VERSION)…')
# 1. Get spectra (shape LUT_SIZE×LUT_SIZE×441)
spectra = compute_lut_spectra(lut_size=LUT_SIZE, smooth_steps=1).astype(np.float32)  # half later
print(f'Spectra shape: {spectra.shape}, min: {spectra.min()}, max: {spectra.max()}')

print('Loading film profile:', FILM_PROFILE)
profile_file = DATA_DIR/f'{FILM_PROFILE}.json'
with profile_file.open('r') as f:
    prof = json.load(f)
log_sens = np.array(prof['data']['log_sensitivity'], dtype=np.float32)  # shape (441,3)

# Handle NaN values in sensitivity data - EXACTLY like Python reference
print(f'Original sensitivity shape: {log_sens.shape}')
print(f'NaN count in log_sens: {np.isnan(log_sens).sum()}')

# Use EXACT same NaN handling as Python reference: np.nan_to_num()
sensitivity = 10.0**log_sens
sensitivity = np.nan_to_num(sensitivity)  # Replace NaN with 0 (same as Python reference)
print(f'Sensitivity shape: {sensitivity.shape}, min: {sensitivity.min()}, max: {sensitivity.max()}')

# Apply band pass filter - EXACTLY like Python reference
filter_uv = (1, 410, 8)  # Same as Python reference
filter_ir = (1, 675, 15)  # Same as Python reference

print(f'Applying band pass filter: UV={filter_uv}, IR={filter_ir}')
if filter_uv[0] > 0 or filter_ir[0] > 0:
    band_pass_filter = compute_band_pass_filter(filter_uv, filter_ir)
    print(f'Band pass filter shape: {band_pass_filter.shape}')
    print(f'Band pass filter range: [{band_pass_filter.min():.6f}, {band_pass_filter.max():.6f}]')
    sensitivity *= band_pass_filter[:, None]  # Same as Python reference
    print(f'After band pass filter, sensitivity range: [{sensitivity.min():.6f}, {sensitivity.max():.6f}]')

print('Contracting spectra with sensitivities…')
# einsum ijk,lk->ijl where k wavelengths
exp = np.einsum('ijk,kl->ijl', spectra, sensitivity)  # shape LUT×LUT×3
print(f'Exp shape: {exp.shape}, min: {exp.min()}, max: {exp.max()}')
print(f'Exp has NaN: {np.isnan(exp).any()}, Inf: {np.isinf(exp).any()}')

# Apply exposure adjustment - EXACTLY like Python reference
exposure_ev = 0.0  # Default exposure (same as Python reference)
exp *= 2**exposure_ev
print(f'After exposure adjustment (EV={exposure_ev}): min: {exp.min()}, max: {exp.max()}')

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