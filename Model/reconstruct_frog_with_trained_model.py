#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
reconstruct_frog_improved.py

Reconstruct ultrashort pulses from experimental FROG data using trained DeepFROG model.

Handles 400x400 experimental data with proper interpolation to 64x64 for model input.
Automatically detects units (fs , ps) from the data.

Usage example:
    python reconstruct_frog_improved.py --ckpt checkpoints/epoch_060.pt --txt FROG400/06271822.txt
    python reconstruct_frog_improved.py --ckpt checkpoints/epoch_060.pt --folder FROG400/ --outdir results/
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import model from training file
try:
    sys.path.insert(0, os.getcwd())
    from train_deepfrog_supervised_or_hybrid_ascii import MultiresNet, FROGNetSHG, normalize_Evec_maxabs
except ImportError:
    print("ERROR: Could not import from train_deepfrog_supervised_or_hybrid_ascii.py")
    print("Make sure the training file is in the same directory.")
    sys.exit(1)

# Optional scipy for Gaussian filtering
try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using manual Gaussian filter")


# 1 Read FROG .txt data


def read_frog_txt(filepath):
    """
    Read FROG ASCII .txt file to get FROG matrix and header information for physical parameters
    
    
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        # Read header
        header = f.readline().strip()
        parts = header.replace(',', ' ').split()
        
        if len(parts) < 5:
            raise ValueError(f"Header must have 5 values, got {len(parts)}")
        
        n_delay = int(float(parts[0]))
        n_freq = int(float(parts[1]))
        dt_fs = float(parts[2])  # Already in fs according to header format
        df = float(parts[3])     # In PHz
        f_central = float(parts[4])  # In PHz
        
        # Read data
        data = np.loadtxt(f)
    
    if data.size != n_delay * n_freq:
        raise ValueError(f"Expected {n_delay}x{n_freq}={n_delay*n_freq} values, got {data.size}")
    
    trace = data.reshape((n_delay, n_freq))
    
    metadata = {
        'n_delay': n_delay,
        'n_freq': n_freq,
        'dt_fs': dt_fs,  # Delay step in fs
        'df': df,        # Frequency step in PHz
        'f_central': f_central,  # Central frequency in PHz
    }
    
    return metadata, trace



# 2 Interpolation 400×400 to 64×64


def gaussian_filter_delay_axis(arr, sigma):
    """
    Apply Gaussian filter along delay axis (axis=0) to avoid aliasing.
   
    """
    if HAS_SCIPY:
        return gaussian_filter1d(arr, sigma=sigma, axis=0, mode='nearest')
    
    # Manual implementation with reflection padding
    if sigma <= 0:
        return arr
    
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()
    
    # Reflect padding
    pad_top = np.repeat(arr[:1, :], radius, axis=0)
    pad_bot = np.repeat(arr[-1:, :], radius, axis=0)
    padded = np.vstack([pad_top, arr, pad_bot])
    
    # Convolve each frequency column
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kernel, mode='valid')
    
    return out


def interpolate_400_to_64(trace_400, dt_fs, df_PHz):
    """
   
    Interpolation 400x400 FROG trace to 64x64

    
    """
    N_old, M_old = trace_400.shape
    N_new = 64
    
    if N_old < 64 or M_old < 64:
        raise ValueError(f"Input trace {trace_400.shape} must be at least 64x64")
    
    # Calculate new delay step to match training data convention: dt·df = 1/N
    dt_new_fs = 1.0 / (N_new * df_PHz)  # fs, as df is in PHz (1/ps)
    
    print(f"    Original: dt={dt_fs:.3f} fs, df={df_PHz:.6f} PHz, size={N_old}x{M_old}")
    print(f"    Target:   dt={dt_new_fs:.3f} fs, df={df_PHz:.6f} PHz (same), size={N_new}x{N_new}")
    print(f"    Downsampling ratio (delay): {N_old/N_new:.2f}x")
    print(f"    [NOTE] Training data: dt≈21.39 fs, pulse widths ~107-320 fs (5-15 points)")
    print(f"    [NOTE] Narrower pulses (<60 fs) may be undersampled ( systematic error )")
    
    # 1 Prefilter delay axis to avoid aliasing
    downsample_ratio = N_old / N_new
    sigma_samples = 0.5 * downsample_ratio
    
    print(f"    Applying Gaussian prefilter (σ={sigma_samples:.2f} samples)...")
    filtered = gaussian_filter_delay_axis(trace_400, sigma=sigma_samples)
    
    # 2Crop frequency axis to central 64 columns
    freq_center = M_old // 2
    freq_half = N_new // 2
    freq_start = freq_center - freq_half
    freq_end = freq_start + N_new
    
    cropped = filtered[:, freq_start:freq_end]  # (400, 64)
    print(f"    Cropped frequency: columns {freq_start} to {freq_end}")
    
    # 3 Interpolate delay axis using pixel indices 
    row_indices_old = np.arange(N_old, dtype=float)  # 0, 1, 2, ..., 399
    row_indices_new = np.linspace(0, N_old - 1, N_new)  # 64 evenly-spaced points
    
    trace_64 = np.empty((N_new, N_new), dtype=np.float32)
    for j in range(N_new):
        trace_64[:, j] = np.interp(row_indices_new, row_indices_old, cropped[:, j])
    
    # 4 Clean up and normalize
    trace_64 = np.maximum(trace_64, 0.0)
    max_val = trace_64.max()
    if max_val > 0:
        trace_64 /= max_val
    
    print(f"    Final 64x64 trace: min={trace_64.min():.3e}, max={trace_64.max():.3f}")
    
    return trace_64, dt_new_fs



# 3 Model calling


def reconstruct_pulse(trace_64, model, device):
    """
    Use trained model to reconstruct pulse from 64x64 FROG trace.
    
    
    """
    model.eval()
    frog_forward = FROGNetSHG(N=64, normalize_input=True, normalize_trace=True).to(device)
    frog_forward.eval()
    
    with torch.no_grad():
        # Fix for time-direction ambiguity


        trace_64_flipped = np.flip(trace_64, axis=0).copy()  # Flip delay axis (rows)
        
        I_input = torch.from_numpy(trace_64_flipped[None, ...]).float().to(device)  # (1, 64, 64)
        
        # Normalize (same as training)
        I_input = I_input / I_input.flatten(1).amax(dim=1, keepdim=True).clamp_min(1e-20).view(-1, 1, 1)
        
        # Apply model to get predicted electric field
        E_pred = model(I_input)  # (1, 128)
        
        # Normalize by max |E|
        E_pred_n = normalize_Evec_maxabs(E_pred)
        
        # Forward model to check consistency
        I_pred = frog_forward(E_pred_n)  # (1, 64, 64)
    
    E_pred_ri = E_pred_n.squeeze(0).cpu().numpy().astype(np.float32)
    I_pred_np = I_pred.squeeze(0).cpu().numpy().astype(np.float32)
    
    # Flip predicted trace back to match original data orientation
    I_pred_np = np.flip(I_pred_np, axis=0).copy()
    
    return E_pred_ri, I_pred_np


def ri_to_complex(E_ri):
    """Convert [Re, Im] format to complex array."""
    N = E_ri.shape[-1] // 2
    return E_ri[:N].astype(np.float32) + 1j * E_ri[N:].astype(np.float32)



# 4 Reconstructed pulse visualization


def compute_pulse_metrics(t_fs, E_complex):
    """
    Compute pulse duration and draw the reconstructed pulse
    
   
    """
    I = np.abs(E_complex)**2
    if I.max() <= 0:
        return {'FWHM_fs': np.nan, 'RMS_fs': np.nan, 'peak_time_fs': 0.0}
    
    I = I / I.max()
    
    
    peak_idx = np.argmax(I)
    peak_time = float(t_fs[peak_idx])
    
    # FWHM 
    half = 0.5
    left_t, right_t = None, None
    
    
    for i in range(peak_idx, 0, -1):
        if I[i-1] <= half <= I[i]:
            frac = (half - I[i-1]) / (I[i] - I[i-1] + 1e-20)
            left_t = t_fs[i-1] + frac * (t_fs[i] - t_fs[i-1])
            break
    
    
    for i in range(peak_idx, len(I)-1):
        if I[i] >= half >= I[i+1]:
            frac = (half - I[i+1]) / (I[i] - I[i+1] + 1e-20)
            right_t = t_fs[i+1] + frac * (t_fs[i] - t_fs[i+1])
            break
    
    FWHM = float(right_t - left_t) if (left_t is not None and right_t is not None) else np.nan
    
    # RMS duration
    I_sum = I.sum() + 1e-20
    t_mean = np.sum(t_fs * I) / I_sum
    sigma = np.sqrt(np.sum(((t_fs - t_mean)**2) * I) / I_sum)
    
    return {
        'FWHM_fs': float(FWHM),
        'RMS_fs': float(sigma),
        'RMS_FWHM_equiv_fs': float(2.355 * sigma),
        'peak_time_fs': peak_time,
    }


def plot_reconstruction(save_path, name,  I_meas, I_pred, E_complex, dt_fs, df_PHz, f_central_PHz, metrics, n_freq_original=400, n_delay_original=400, dt_fs_original=None):
   
    N = 64
    t_axis = (np.arange(N) - N/2) * dt_fs  # fs 
    f_axis_rel = (np.arange(N) - N/2) * df_PHz  # PHz 
    f_axis_abs = f_central_PHz + f_axis_rel  # PHz 
    
    # Calculate full frequency range from original 400x400 data
    f_axis_full_rel = (np.arange(n_freq_original) - n_freq_original/2) * df_PHz
    f_axis_full_abs = f_central_PHz + f_axis_full_rel  
    # Calculate full delay range from original 400x400 data
    if dt_fs_original is None:
        dt_fs_original = dt_fs  
    t_axis_full = (np.arange(n_delay_original) - n_delay_original/2) * dt_fs_original 
    
    # Prepare field quantities in time domain
    I_time = np.abs(E_complex)**2
    I_time = I_time / (I_time.max() + 1e-20)
    phase_time = np.unwrap(np.angle(E_complex))
    phase_time -= np.average(phase_time, weights=I_time + 1e-20)


    
    freq_marg_pred = I_pred.sum(axis=0)  
    I_freq = freq_marg_pred / (freq_marg_pred.max() + 1e-20)
    
    # Frequency axis from FROG data
    freq_axis_fft = f_axis_abs  
    

    # Get phase information from FFT of E(t) for spectral 
    E_spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_complex)))
    phase_freq = np.unwrap(np.angle(E_spectrum))
    phase_freq -= np.average(phase_freq, weights=I_freq + 1e-20)
    
    # Create figure 
    fig = plt.figure(figsize=(17, 11), dpi=120)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.1, 1.1], hspace=0.35, wspace=0.3)
    
    # Row 1 FROG traces
    ax00 = fig.add_subplot(gs[0, 0])
    im0 = ax00.pcolormesh(f_axis_abs, t_axis, I_meas, shading='auto', cmap='hot')
    ax00.set_xlabel('Frequency (PHz)')
    ax00.set_ylabel('Delay (fs)')
    ax00.set_title('Measured FROG (64x64)')
    fig.colorbar(im0, ax=ax00, label='Normalized Intensity')
    
    ax01 = fig.add_subplot(gs[0, 1])
    im1 = ax01.pcolormesh(f_axis_abs, t_axis, I_pred, shading='auto', cmap='hot')
    ax01.set_xlabel('Frequency (PHz)')
    ax01.set_ylabel('Delay (fs)')
    ax01.set_title('Predicted FROG (Reconstructed)')
    fig.colorbar(im1, ax=ax01, label='Normalized Intensity')
    
    ax02 = fig.add_subplot(gs[0, 2])
    diff = np.abs(I_meas - I_pred)
    im2 = ax02.pcolormesh(f_axis_abs, t_axis, diff, shading='auto', cmap='RdBu')
    ax02.set_xlabel('Frequency (PHz)')
    ax02.set_ylabel('Delay (fs)')
    mse = np.mean(diff**2)
    ax02.set_title(f'Absolute Difference (MSE={mse:.3e})')
    fig.colorbar(im2, ax=ax02)
    
    # Row 2 Picture of reconstructed pulse in time domain
    ax10 = fig.add_subplot(gs[1, 0])
    # Plot only the 64-point valid data
    ax10.plot(t_axis, I_time, 'b-', linewidth=2.5, label='|E(t)|²')
    ax10.set_xlabel('Time (fs)', fontsize=10)
    ax10.set_ylabel('Normalized Amplitude', fontsize=10)
    ax10.set_title('Reconstructed Pulse - Time Domain', fontsize=11)
    # Set x-axis to show full 400-point delay range
    ax10.set_xlim(t_axis_full.min(), t_axis_full.max())
    ax10.axvspan(t_axis_full.min(), t_axis_full.max(), alpha=0.12, color='blue', 
                 label='Full delay range', zorder=0)
    ax10.legend(loc='upper right', fontsize=7, framealpha=0.7)
    ax10.grid(True, alpha=0.3)
    
    ax10_phase = ax10.twinx()
    ax10_phase.plot(t_axis, phase_time, 'k:', linewidth=1.5, label='Phase')
    ax10_phase.set_ylabel('Phase (rad)', color='k', fontsize=10)
    ax10_phase.tick_params(axis='y', labelcolor='k')
    
    # Row 2 Picture of reconstructed pulse in frequency domain
    ax11 = fig.add_subplot(gs[1, 1])
    # Plot only the 64-point cropped valid data
    ax11.plot(freq_axis_fft, I_freq, 'b-', linewidth=2.5, label='|E(ω)|²')
    ax11.set_xlabel('Frequency (PHz)', fontsize=10)
    ax11.set_ylabel('Normalized Spectral Intensity', fontsize=10)
    ax11.set_title('Reconstructed Pulse - Frequency Domain', fontsize=11)
    # Set x-axis to show full 400-point frequency range
    ax11.set_xlim(f_axis_full_abs.min(), f_axis_full_abs.max())
   
    ax11.axvspan(freq_axis_fft.min(), freq_axis_fft.max(), alpha=0.12, color='green', 
                 label='Cropped region', zorder=0)
    ax11.legend(loc='upper right', fontsize=7, framealpha=0.7)
    ax11.grid(True, alpha=0.3)
    
    ax11_phase = ax11.twinx()
    ax11_phase.plot(freq_axis_fft, phase_freq, 'k:', linewidth=1.5, label='Spectral Phase')
    ax11_phase.set_ylabel('Spectral Phase (rad)', color='k', fontsize=10)
    ax11_phase.tick_params(axis='y', labelcolor='k')
    
    # Row 3 FROG marginals
    ax20 = fig.add_subplot(gs[2, 0])
    delay_marg_meas = I_meas.sum(axis=1)
    delay_marg_pred = I_pred.sum(axis=1)
    ax20.plot(t_axis, delay_marg_meas / delay_marg_meas.max(), 'b-', linewidth=2, label='Measured')
    ax20.plot(t_axis, delay_marg_pred / delay_marg_pred.max(), 'r--', linewidth=2, label='Predicted')
    ax20.set_xlabel('Delay (fs)', fontsize=10)
    ax20.set_ylabel('Normalized Intensity', fontsize=10)
    ax20.set_title('FROG Delay Marginal', fontsize=11)
    # Set x-axis to show full 400-point delay range
    ax20.set_xlim(t_axis_full.min(), t_axis_full.max())
    # Shade the FULL delay range 
    ax20.axvspan(t_axis_full.min(), t_axis_full.max(), alpha=0.12, color='blue', 
                 label='Full delay range', zorder=0)
    ax20.legend(loc='upper right', fontsize=7, framealpha=0.7)
    ax20.grid(True, alpha=0.3)
    
    ax21 = fig.add_subplot(gs[2, 1])
    freq_marg_meas = I_meas.sum(axis=0)
    freq_marg_pred = I_pred.sum(axis=0)
    ax21.plot(f_axis_abs, freq_marg_meas / freq_marg_meas.max(), 'b-', linewidth=2, label='Measured')
    ax21.plot(f_axis_abs, freq_marg_pred / freq_marg_pred.max(), 'r--', linewidth=2, label='Predicted')
    ax21.set_xlabel('Frequency (PHz)', fontsize=10)
    ax21.set_ylabel('Normalized Intensity', fontsize=10)
    ax21.set_title('FROG Frequency Marginal', fontsize=11)
    # Set x-axis to show full 400-point frequency range
    ax21.set_xlim(f_axis_full_abs.min(), f_axis_full_abs.max())
    # Shade the 64-point cropped region
    ax21.axvspan(f_axis_abs.min(), f_axis_abs.max(), alpha=0.12, color='green', 
                 label='Cropped region', zorder=0)
    ax21.legend(loc='upper right', fontsize=7, framealpha=0.7)
    ax21.grid(True, alpha=0.3)
    
    # Row 3 Reconstruction pulse information summary
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.axis('off')
    info_text = f"File: {name}\n\n"
    info_text += "Pulse Duration Metrics:\n"
    info_text += f"  FWHM:          {metrics['FWHM_fs']:.2f} fs\n"
    info_text += f"  RMS σ:         {metrics['RMS_fs']:.2f} fs\n"
    info_text += f"  RMS→FWHM:      {metrics['RMS_FWHM_equiv_fs']:.2f} fs\n"
    info_text += f"  Peak time:     {metrics['peak_time_fs']:.2f} fs\n\n"
    info_text += "Sampling Parameters:\n"
    info_text += f"  dt (64-grid):  {dt_fs:.3f} fs\n"
    info_text += f"  df:            {df_PHz:.6f} PHz\n"
    info_text += f"  f_central:     {f_central_PHz:.6f} PHz\n\n"
    info_text += "Consistency Check:\n"
    info_text += f"  FROG MSE:      {mse:.3e}\n"
    ax22.text(0.1, 1.1, info_text, transform=ax22.transAxes,
              fontsize=10, verticalalignment='center', horizontalalignment='left', family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    # Apply tight_layout
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    
    plt.savefig(save_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)



#Main

def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct ultrashort pulses from FROG data using trained DeepFROG model'
    )
    
    # Input specification 
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--txt', type=str, help='Single FROG .txt file')
    input_group.add_argument('--folder', type=str, help='Folder containing multiple .txt files')
    
    # Model and output
    parser.add_argument('--ckpt', type=str, required=True, help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--outdir', type=str, default='recon_results', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*70)
    print("DeepFROG Pulse Reconstruction from Experimental FROG Data")
    print("="*70)
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.ckpt}")
    model = MultiresNet(N=64).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    print("[OK] Model loaded successfully")
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir.resolve()}")
    
    # Collect input files
    if args.txt:
        files = [args.txt]
    else:
        files = sorted(glob.glob(os.path.join(args.folder, '*.txt')))
    
    if not files:
        print("ERROR: No .txt files found!")
        return
    
    print(f"\nProcessing {len(files)} file(s)...")
    print("="*70)
    
    # Process each file
    results_summary = []
    
    for filepath in files:
        filename = Path(filepath).name
        filestem = Path(filepath).stem
        
        print(f"\n[{filename}]")
        
        try:
            # 1 Read FROG data 
            metadata, trace_400 = read_frog_txt(filepath)
            print(f"  Read {metadata['n_delay']}x{metadata['n_freq']} FROG trace")
            print(f"  Units: dt={metadata['dt_fs']:.3f} fs, df={metadata['df']:.6f} PHz")
            print(f"  Central frequency: {metadata['f_central']:.6f} PHz")
            
            # 2  interpolate to 64×64
            trace_64, dt_new_fs = interpolate_400_to_64(
                trace_400, 
                metadata['dt_fs'],
                metadata['df']
            )
            
            # 3 Reconstruct pulse
            print("  Running model inference...")
            E_pred_ri, I_pred = reconstruct_pulse(trace_64, model, device)
            E_complex = ri_to_complex(E_pred_ri)
            
            # 4 Compute metrics
            t_axis = (np.arange(64) - 32) * dt_new_fs
            metrics = compute_pulse_metrics(t_axis, E_complex)
            
            print(f"  [OK] Reconstruction complete!")
            print(f"    FWHM: {metrics['FWHM_fs']:.2f} fs")
            print(f"    RMS:  {metrics['RMS_fs']:.2f} fs")
            
            # 5 Save outputs
            np.save(outdir / f'{filestem}_trace64.npy', trace_64)
            np.save(outdir / f'{filestem}_E_pred_ri.npy', E_pred_ri)
            np.save(outdir / f'{filestem}_E_pred_complex.npy', E_complex.astype(np.complex64))
            np.save(outdir / f'{filestem}_I_pred.npy', I_pred)
            
            # Save metadata
            output_meta = {
                'filename': filename,
                'input_size': [metadata['n_delay'], metadata['n_freq']],
                'dt_original_fs': metadata['dt_fs'],  # Original dt in fs
                'dt_resampled_fs': float(dt_new_fs),  # Resampled dt for 64-grid in fs
                'df_PHz': metadata['df'],
                'f_central_PHz': metadata['f_central'],
                'metrics': metrics,
            }
            with open(outdir / f'{filestem}_info.json', 'w') as jf:
                json.dump(output_meta, jf, indent=2)
            
            # 6 Create visualization
            plot_reconstruction(
                outdir / f'{filestem}_reconstruction',
                filename,
                trace_64, I_pred, E_complex,
                dt_new_fs, metadata['df'], metadata['f_central'],
                metrics,
                n_freq_original=metadata['n_freq'], 
                n_delay_original=metadata['n_delay'], 
                dt_fs_original=metadata['dt_fs'] 
            )
            
            print(f"  [OK] Saved outputs to {outdir}")
            
            results_summary.append({
                'file': filename,
                'FWHM_fs': metrics['FWHM_fs'],
                'RMS_fs': metrics['RMS_fs'],
            })
            
        except Exception as e:
            print(f"  [ERROR]: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary of results
    print("\n" + "="*70)
    print("RECONSTRUCTION SUMMARY")
    print("="*70)
    for res in results_summary:
        print(f"{res['file']:30s}  FWHM={res['FWHM_fs']:7.2f} fs  RMS={res['RMS_fs']:7.2f} fs")
    
    print(f"\n[OK] All results saved to: {outdir.resolve()}")


if __name__ == '__main__':
    main()

