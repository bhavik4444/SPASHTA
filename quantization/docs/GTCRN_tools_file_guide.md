# GTCRN ESP32 — `tools` File Guide

This document explains the scripts and utilities used during GTCRN model export, verification, PC-side testing, and ESP32 audio experiments.

## Core model-development tools

These are the main tools that belong to the documented GTCRN workflow.

| File | Purpose |
|---|---|
| `export_int8.py` | Converts the trained PyTorch checkpoint into the flat `gtcrn_int8.bin` model format used by the ESP32 C runtime. It also writes the model JSON sidecar. |
| `verify_export.py` | Verification step for the exported model. Compares float/int8 behavior on an evaluation set and can generate `gtcrn_testvec.bin` for the ESP32 boot self-test. |
| `gtcrn_ref.py` | NumPy reference implementation of the GTCRN streaming pipeline. This is the numerical reference/specification used to compare against the C runtime. |
| `host_test.c` | Runs the C GTCRN runtime on a normal computer without the ESP32. Used to verify the C implementation and measure desktop runtime before flashing hardware. |
| `make_synthetic_blob.py` | Creates a structurally valid synthetic/random GTCRN model blob. Useful for testing the blob parser/runtime without a trained PyTorch model. |
| `qat_finetune.py` | Optional quantisation-aware fine-tuning script. Applies fake quantisation to the tensors that are eventually quantised for ESP32 inference. Use only when verification shows quantisation needs improvement. |

## PC audio / resampling tools

| File | Purpose |
|---|---|
| `audio_resampler.py` | Stateful FIR resampler for the PC audio path. Supports 48 kHz → 16 kHz and 16 kHz → 48 kHz while preserving FIR history between blocks. |
| `audio_resampler_fixed.py` | Corrected/fixed version of the resampler used during development after streaming block-boundary/output-length issues were found. |
| `esp_audio_bridge.py` | PC-side serial audio bridge. Sends 16 kHz mono PCM hops to the ESP32 over the 921600-baud serial link and receives processed audio hops. |
| `esp_audio_bridgeNG.py` | Experimental/new-generation variant of the serial bridge developed during the project. Keep outside the final repo unless this version is explicitly selected as the maintained bridge. |
| `esp_audio_bridge_old.py` | Older serial-bridge implementation retained for development/history. Not needed for the final repo. |

## Live-audio experiments

| File | Purpose |
|---|---|
| `esp_audio_live.py` | First PC live microphone → serial → ESP32 → speaker pipeline. Uses queues and a background worker so the audio callback does not directly wait on the ESP32. |
| `esp_audio_live_fixed.py` | Revised live pipeline with queue/underflow handling and simplified gain control. Used to test whether processed audio could reach the speaker reliably. |
| `esp_audio_live_48k.py` | 48 kHz USB microphone/headphone live pipeline. Resamples 48 kHz → 16 kHz for GTCRN and 16 kHz → 48 kHz for playback, with streaming AGC and output restoration. |
| `esp_audio_live_pre_post.py` | Live pipeline experiment with explicit preprocessing/input AGC and output post-processing/gain restoration. |
| `gtcrn_2core_demo_6s.py` | Short demo/test script intended for the two-core GTCRN pipeline work. Used to exercise the newer pipelined firmware over the serial interface. |
| `gtcrn_20s_demo_48k.py` | Longer PC demo using 48 kHz audio, recording first and then processing through the ESP32 pipeline. Used for demonstrations rather than the core firmware. |

## Stable/batch testing tools

| File | Purpose |
|---|---|
| `gtcrn_stable_lite_demo.py` | Stable batch/demo script for the lighter GTCRN model. Records a short clip, normalises it for model input, sends it hop-by-hop, receives the output, restores gain, and plays/saves the result. |
| `gtcrn_stable_lite_demo_6s.py` | Six-second version of the stable lite-model demo used for quick repeatable testing. |
| `esp_audio_replay_normalized.py` | Replays previously recorded/normalised audio through the serial GTCRN bridge instead of requiring a fresh microphone recording. Useful for repeatable comparisons. |
| `esp_audio_bridge_normalized.py` | Serial demo bridge with whole-recording RMS normalisation before sending audio to GTCRN and gain restoration after receiving the processed output. |
| `test_lite_reference.py` | Tests the lite-model reference path, including 48 kHz → 16 kHz resampling, model-input normalisation, reference enhancement, and conversion back to 48 kHz. |
| `gtcrn_demo.py` | Earlier larger demo/GUI-oriented script used during presentation and interactive testing. Primarily a development/demo tool rather than a core model-development utility. |

## Profiling / optimisation files

| File | Purpose |
|---|---|
| `gtcrn_stage_profiler.patch` | Patch used to add timing/profiling information to the GTCRN runtime so frontend, encoder, DPGRNN, decoder, post-processing, and FFT stages could be measured separately. |
| `gtcrn_net_weight_cache.patch` | Experimental patch related to keeping model weights in faster memory / improving weight access performance on the ESP32-S3. |
| `gtcrn_ops_fastmath2.c` | Experimental faster version of the GTCRN numeric operations implementation. Used during performance optimisation; not necessarily the maintained final source. |
| `gtcrn_ops_small_dot_fast.c` | Experimental optimisation of small int8 dot products, especially for small GRU dimensions. |
| `gtcrn_ops_n8_fast.c` | Experimental optimisation of the same GTCRN numeric core with additional small-vector/dot-product optimisations. |

## What the main tools produce

```text
PyTorch checkpoint
       |
       v
  export_int8.py
       |
       v
gtcrn_int8.bin
gtcrn_int8.bin.json
       |
       v
 verify_export.py
       |
       +--> numerical verification
       |
       +--> gtcrn_testvec.bin
                   |
                   v
             ESP32 boot self-test
```

For PC/runtime validation:

```text
gtcrn_ref.py
     |
     +---- reference output

host_test.c
     |
     +---- C runtime output

Compare the two before relying on ESP32 results.
```

For the audio demo path:

```text
Microphone
   |
   v
PC Python tool
   |
   +--> optional resampling / normalisation
   |
   v
UART 921600
   |
   v
ESP32 GTCRN
   |
   v
UART
   |
   v
PC Python tool
   |
   +--> optional gain restoration / resampling
   |
   v
Speaker / WAV
```

## Suggested final `tools/` set

For a clean public repository, the documented model-development tools are the most important:

```text
tools/
├── export_int8.py
├── verify_export.py
├── gtcrn_ref.py
├── host_test.c
├── make_synthetic_blob.py
└── qat_finetune.py
```

A maintained PC demo/bridge can also be kept, but the many older/live/experimental scripts should be moved to a separate `experiments/` directory or removed before publishing the final repository.

> Note: several files in the `tools/` folder were created during iterative ESP32 performance/audio experiments. Their descriptions above identify their development role; they should not all be presented as part of the final production pipeline.
