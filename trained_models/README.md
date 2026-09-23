# Trained Models

This folder contains the two trained SPASHTA model variants.

## Model Summary

| Model | Total Parameters | Trainable Parameters | Base Channels | DPGRNN Stages | Dilations | Deep Filter |
|---|---:|---:|---:|---:|---|---:|
| **model_lite** | **49,048** | 24,472 | 16 | 2 | 1, 2, 5 | 3 × 48 bins |
| **model_high** | **149,348** | 124,772 | 32 | 3 | 1, 2, 4, 8, 16 | 5 × 64 bins |

Both variants use the causal GTCRN-DF architecture implemented in `model.py`.

## Training & Data Configuration

| Parameter | model_lite | model_high |
|---|---:|---:|
| Sample rate | 16 kHz | 16 kHz |
| Segment length | 4 s | 4 s |
| Training SNR | -15 to +20 dB | -15 to +20 dB |
| SNR curriculum | -5 → -15 dB / 12 epochs | -5 → -15 dB / 15 epochs |
| Hard examples | 45% / 10 dB span | 45% / 10 dB span |
| Speech-only probability | 6% | 6% |
| Noise-only probability | 6% | 6% |
| Burst-noise probability | 55% | 55% |
| Mix RMS | 0.1 | 0.1 |
| FFT / hop | 512 / 256 | 512 / 256 |
| TRA bands | 8 | 8 |
| Epochs | 120 | 150 |
| Steps / epoch | 400 | 500 |
| Validation items | 768 | 768 |
| Batch size | 16 | 16 |
| Learning rate | 1e-3 | 1e-3 |
| Warmup | 3 epochs | 3 epochs |
| Minimum LR ratio | 0.02 | 0.02 |
| Weight decay | 1e-5 | 1e-5 |
| Gradient clipping | 5.0 | 5.0 |
| EMA decay | 0.999 | 0.999 |

## Loss Configuration

| Parameter | Value |
|---|---:|
| Spectral loss weight | 1.0 |
| SNR loss weight | 0.25 |
| Multi-resolution STFT loss weight | 0.35 |
| Speech preservation weight | 3.0 |
| Active-frame suppression | 1.0 |
| Silent-frame suppression | 2.0 → 8.0 |
| Silent suppression ramp | 40 epochs (lite) / 50 epochs (high) |
| Magnitude compression power | 0.3 |
| Speech band | 250–5000 Hz |
| Speech-band weight | 2.0 |
| Speech activity threshold | -40 dB |

### Checkpoints

Each model directory contains:

- `best_model.pt` — best validation checkpoint
- `last_model.pt` — final training checkpoint
- `history.csv` — training history
- `run_args.json` — exact configuration used for the run