#!/usr/bin/env python3
"""
spectrogram_comparison.py
==========================

Compare a noisy input recording against a speech-enhancement model's output,
side by side, plus a third "gain" panel that shows exactly where the model
added or removed energy (in dB) at each time/frequency cell.

Meant to be run once per (noisy, enhanced) pair -- e.g. once per training
checkpoint -- so you can flip between the output images and watch how a
model's behavior changes across epochs.

Dependencies: numpy, scipy, matplotlib (no librosa required).

USAGE
-----
    python spectrogram_comparison.py NOISY.wav ENHANCED.wav

    # custom output path + label the enhanced panel with an epoch number
    python spectrogram_comparison.py noisy.wav enhanced_v2_30epochs.wav \
        -o comparisons/epoch30.png \
        --enhanced-label "Enhanced output (epoch 30)"

    # finer time resolution, wider gain color range
    python spectrogram_comparison.py noisy.wav enhanced.wav \
        --nperseg 2048 --hop 512 --gain-range 20


HOW TO READ THE OUTPUT
-----------------------
Panel 1 (Noisy input):
    The wavy horizontal stacked stripes are speech harmonics/formants --
    that's the signal you want the model to preserve. Diffuse haze that
    ISN'T stacked in stripes is noise (steady hiss/fan noise looks like a
    horizontal haze; keyboard clicks or claps look like thin vertical
    stripes spanning every frequency).

Panel 2 (Enhanced output):
    Good sign:  the harmonic stripes from panel 1 are still visible and
                unbroken, while the space between them has gone darker.
    Bad sign:   stripes look broken or patchy -- the model is deleting
                parts of speech, not just noise.
    Bad sign:   everything looks smeared/blurred -- loss of high-frequency
                detail, which is where consonants like s/f/th live
                (roughly 4-8 kHz). This shows up as muffled speech.
    Bad sign:   the panel is almost entirely black. That is usually NOT
                great denoising -- it means the model is being too
                aggressive and deleting quiet speech along with the noise.

Panel 3 (Gain applied = enhanced_dB - noisy_dB) -- the most diagnostic panel:
    Solid red in silent/pause regions   = noise correctly identified and
                                           removed. You want this to get
                                           MORE consistent across epochs.
    Red bleeding INTO the harmonic       = the model is suppressing speech
    stripe pattern from panel 1            energy, not just noise. This is
                                           the #1 failure mode to watch for
                                           -- it causes muffled, robotic, or
                                           "underwater" sounding speech.
    Speckled/checkerboard red+blue       = "musical noise", a classic
    noise with no structure                artifact of spectral-subtraction
                                           and some DNN enhancement models.
                                           Audible as random chirps/warbles.
    Solid blue in one frequency band     = the model is boosting that band.
                                           Check whether it lines up with a
                                           speech formant (good -- helps
                                           intelligibility) or with where
                                           the noise floor was (bad -- it's
                                           amplifying residual noise).

Comparing multiple epochs of the SAME file: look for panel 3 becoming
cleaner and more "block-structured" over training (clearly red where speech
is absent, close to white/neutral where speech is present). Speckle
appearing, or red creeping into speech-shaped regions as epochs increase,
usually signals the model overfitting to noise removal at the expense of
speech fidelity.
"""

import argparse
import os
import sys
from math import gcd

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import stft, resample_poly
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Audio loading / alignment
# ---------------------------------------------------------------------------

def load_audio(path):
    """Load a WAV file and return (samples as float32 in [-1, 1], sample_rate)."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        # Multi-channel file: collapse to mono by averaging channels so the
        # comparison always runs on a single spectrogram per file.
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        # int16/int32 PCM -> normalize by the dtype's full-scale range so
        # 0 dBFS means "the loudest possible sample", consistently across files.
        max_val = np.iinfo(data.dtype).max + 1
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return data, sr


def align_audio(noisy, sr_noisy, enhanced, sr_enhanced):
    """Ensure both signals share a sample rate and length so their
    spectrograms land on the exact same time/frequency grid (required for
    the gain panel to be a valid cell-by-cell subtraction)."""
    if sr_noisy != sr_enhanced:
        print(f"[warn] sample rates differ ({sr_noisy} vs {sr_enhanced} Hz); "
              f"resampling enhanced audio to {sr_noisy} Hz", file=sys.stderr)
        g = gcd(sr_noisy, sr_enhanced)
        enhanced = resample_poly(enhanced, sr_noisy // g, sr_enhanced // g)

    n = min(len(noisy), len(enhanced))
    if len(noisy) != len(enhanced):
        print(f"[warn] length mismatch ({len(noisy)} vs {len(enhanced)} samples); "
              f"trimming both to {n} samples", file=sys.stderr)
    return noisy[:n], enhanced[:n], sr_noisy


# ---------------------------------------------------------------------------
# Spectrogram computation
# ---------------------------------------------------------------------------

def compute_db_spectrogram(signal, sr, nperseg, hop):
    """STFT magnitude converted to dBFS (0 dB = a full-scale amplitude of 1.0).

    Using a fixed dBFS reference (rather than normalizing each clip to its
    own max, e.g. librosa's default ref=np.max) matters here: it keeps the
    noisy and enhanced panels on the SAME loudness scale. A quieter enhanced
    clip will actually look darker instead of being auto-brightened to match
    the noisy clip, which is what makes the "how much did the model remove"
    comparison meaningful in the first place.
    """
    noverlap = nperseg - hop
    f, t, Z = stft(signal, fs=sr, window='hann', nperseg=nperseg,
                    noverlap=noverlap, boundary=None)
    db = 20 * np.log10(np.abs(Z) + 1e-10)
    return f, t, db


# ---------------------------------------------------------------------------
# Diagnostics: quick numeric read on where the model helped vs hurt
# ---------------------------------------------------------------------------

def print_diagnostics(f, noisy_db, gain_db, speech_band, speech_active_db):
    """Print a short numeric summary alongside the plot. This is a coarse,
    energy-based proxy for "is the model damaging speech or just removing
    noise" -- not a substitute for listening to the file or running a real
    VAD/PESQ/STOI metric, but it's a fast first check.
    """
    band_mask = (f >= speech_band[0]) & (f <= speech_band[1])

    # Within the speech band, split time-frequency cells into "likely speech"
    # (the noisy signal has real energy there) vs. "likely noise/silence"
    # (the noisy signal is already near the noise floor).
    noisy_band = noisy_db[band_mask, :]
    gain_band = gain_db[band_mask, :]
    gain_outside = gain_db[~band_mask, :]

    speech_cells = noisy_band > speech_active_db
    quiet_cells = ~speech_cells

    def safe_mean(x):
        return float(np.mean(x)) if x.size else float('nan')

    speech_gain = safe_mean(gain_band[speech_cells])
    quiet_gain = safe_mean(gain_band[quiet_cells])
    outside_gain = safe_mean(gain_outside)

    # % of "likely speech" cells hit with heavy suppression (>10 dB removed).
    # High numbers here = the model is chewing into speech, not just noise.
    collateral_pct = (100.0 * np.mean(gain_band[speech_cells] < -10)
                       if speech_cells.sum() else float('nan'))

    # % of "likely noise-only" cells barely touched (<3 dB change).
    # High numbers here = noise is passing through mostly untouched.
    residual_pct = (100.0 * np.mean(gain_band[quiet_cells] > -3)
                     if quiet_cells.sum() else float('nan'))

    print("\n--- Diagnostics ---")
    print(f"Speech band: {speech_band[0]:.0f}-{speech_band[1]:.0f} Hz  "
          f"(cells above {speech_active_db:.0f} dBFS treated as 'likely speech')")
    print(f"  Avg gain where speech likely present : {speech_gain:+6.1f} dB  "
          "(closer to 0 = speech preserved; very negative = speech being eaten)")
    print(f"  Avg gain where speech likely absent  : {quiet_gain:+6.1f} dB  "
          "(more negative = more noise removed during pauses -- good)")
    print(f"  Avg gain outside the speech band      : {outside_gain:+6.1f} dB")
    print(f"  Speech cells suppressed >10 dB        : {collateral_pct:5.1f}%  "
          "(possible collateral damage to speech if this is high)")
    print(f"  Noise-only cells barely touched <3 dB : {residual_pct:5.1f}%  "
          "(residual noise leaking through if this is high)")
    print("---------------------\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(f, t, noisy_db, enhanced_db, gain_db,
                     noisy_label, enhanced_label,
                     vmin, vmax, gain_range, max_freq, dpi, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # --- Panel 1: noisy input -------------------------------------------------
    im0 = axes[0].pcolormesh(t, f, np.clip(noisy_db, vmin, vmax),
                              cmap='magma', vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title(noisy_label)
    axes[0].set_ylabel('Hz')
    axes[0].set_ylim(0, max_freq)
    fig.colorbar(im0, ax=axes[0]).set_label('dB')

    # --- Panel 2: enhanced output ---------------------------------------------
    # Same cmap/vmin/vmax as panel 1 on purpose -- if the model is genuinely
    # quieter/darker here, that should be visible, not normalized away.
    im1 = axes[1].pcolormesh(t, f, np.clip(enhanced_db, vmin, vmax),
                              cmap='magma', vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title(enhanced_label)
    axes[1].set_ylabel('Hz')
    axes[1].set_ylim(0, max_freq)
    fig.colorbar(im1, ax=axes[1]).set_label('dB')

    # --- Panel 3: gain applied (enhanced - noisy) -------------------------
    # Diverging colormap centered at 0: red = energy removed (suppressed),
    # blue = energy added (boosted). See the module docstring for how to
    # read the patterns in this panel -- it's the most useful one for
    # spotting speech damage vs. clean noise removal.
    im2 = axes[2].pcolormesh(t, f, np.clip(gain_db, -gain_range, gain_range),
                              cmap='RdBu', vmin=-gain_range, vmax=gain_range,
                              shading='auto')
    axes[2].set_title('Gain applied (enhanced - noisy, dB) '
                       '\u2014 blue = boosted, red = suppressed')
    axes[2].set_ylabel('Hz')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylim(0, max_freq)
    fig.colorbar(im2, ax=axes[2]).set_label('dB change')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare a noisy vs. enhanced audio file with a 3-panel "
                    "spectrogram plot (noisy / enhanced / gain applied).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('noisy', help='Path to the noisy/input WAV file')
    parser.add_argument('enhanced', help='Path to the enhanced/model-output WAV file')
    parser.add_argument('-o', '--output', default=None,
                         help='Output image path (default: auto-generated next '
                              'to the enhanced file)')
    parser.add_argument('--noisy-label', default=None,
                         help='Custom title for panel 1 (default includes the filename)')
    parser.add_argument('--enhanced-label', default=None,
                         help='Custom title for panel 2, e.g. "Enhanced (epoch 30)" '
                              '(default includes the filename)')
    parser.add_argument('--nperseg', type=int, default=1024,
                         help='STFT window length in samples (default: 1024)')
    parser.add_argument('--hop', type=int, default=256,
                         help='STFT hop size in samples (default: 256)')
    parser.add_argument('--vmin', type=float, default=-60,
                         help='dB floor for panels 1-2 color scale (default: -60)')
    parser.add_argument('--vmax', type=float, default=0,
                         help='dB ceiling for panels 1-2 color scale (default: 0)')
    parser.add_argument('--gain-range', type=float, default=15,
                         help='+/- dB range for the gain color scale (default: 15)')
    parser.add_argument('--max-freq', type=float, default=None,
                         help='Y-axis upper limit in Hz (default: Nyquist frequency)')
    parser.add_argument('--speech-band-low', type=float, default=300,
                         help='Lower edge of the speech band for diagnostics, Hz (default: 300)')
    parser.add_argument('--speech-band-high', type=float, default=3400,
                         help='Upper edge of the speech band for diagnostics, Hz (default: 3400)')
    parser.add_argument('--speech-threshold', type=float, default=-30,
                         help='dBFS above which a cell is treated as "likely speech" '
                              'for diagnostics (default: -30)')
    parser.add_argument('--no-diagnostics', action='store_true',
                         help='Skip printing the numeric diagnostics summary')
    parser.add_argument('--dpi', type=int, default=130, help='Output image DPI (default: 130)')
    args = parser.parse_args()

    noisy, sr_noisy = load_audio(args.noisy)
    enhanced, sr_enhanced = load_audio(args.enhanced)
    noisy, enhanced, sr = align_audio(noisy, sr_noisy, enhanced, sr_enhanced)

    f, t, noisy_db = compute_db_spectrogram(noisy, sr, args.nperseg, args.hop)
    _, _, enhanced_db = compute_db_spectrogram(enhanced, sr, args.nperseg, args.hop)
    gain_db = enhanced_db - noisy_db

    max_freq = args.max_freq if args.max_freq is not None else sr / 2

    noisy_label = args.noisy_label or f'Noisy input ({os.path.basename(args.noisy)})'
    enhanced_label = args.enhanced_label or f'Enhanced output ({os.path.basename(args.enhanced)})'

    if args.output:
        output_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.enhanced))[0]
        out_dir = os.path.dirname(os.path.abspath(args.enhanced))
        output_path = os.path.join(out_dir, f'spectrogram_comparison_{stem}.png')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

    plot_comparison(f, t, noisy_db, enhanced_db, gain_db,
                     noisy_label, enhanced_label,
                     args.vmin, args.vmax, args.gain_range, max_freq,
                     args.dpi, output_path)

    print(f"Saved comparison to {output_path}")

    if not args.no_diagnostics:
        print_diagnostics(f, noisy_db, gain_db,
                           speech_band=(args.speech_band_low, args.speech_band_high),
                           speech_active_db=args.speech_threshold)


if __name__ == '__main__':
    main()
