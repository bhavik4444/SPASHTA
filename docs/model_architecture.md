# GTCRN-DF: Model Architecture

A low-complexity, fully causal speech-enhancement network for **adaptive noise cancellation at very low SNR** (trained down to **−15 dB**), built on top of [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) with four targeted changes.

> **In one line:** GTCRN's efficient encoder–decoder backbone, plus level-invariant input features, a bounded identity-initialised mask, and a deep-filter refinement stage, so speech stays intelligible when noise is louder than the speech.

---

## Table of contents

1. [At a glance](#1-at-a-glance)
2. [Inspiration: GTCRN, and what we changed](#2-inspiration-gtcrn-and-what-we-changed)
3. [End-to-end pipeline](#3-end-to-end-pipeline)
4. [Components in detail (and why each exists)](#4-components-in-detail-and-why-each-exists)
5. [Causality and latency](#5-causality-and-latency)
6. [Training design](#6-training-design)
7. [Configuration and model size](#7-configuration-and-model-size)
8. [Built-in sanity tests](#8-built-in-sanity-tests)
9. [References and related work](#9-references-and-related-work)

---

## 1. At a glance

| Property | Value |
|---|---|
| Task | Single-channel speech enhancement / adaptive noise cancellation |
| Target condition | Low SNR, where noise dominates speech (training range −15 to +20 dB) |
| Domain | Complex STFT → predicts a **mask** and a **deep filter** |
| Audio / STFT | 16 kHz, `n_fft=512`, `hop=256`, √Hann window → 257 bins, 32 ms window, 16 ms hop, 31.25 Hz per bin |
| Input / output | `(B, 257, T, 2)` real/imag STFT → `(B, 257, T, 2)` enhanced STFT |
| Backbone | Grouped temporal-conv encoder–decoder + dual-path grouped RNN bottleneck |
| Default size | 32 channels, 5 GTConv blocks per side, 3 stacked DPGRNNs, ≈ **125 K trainable parameters** |
| Causality | Fully causal: no frame ever sees the future |
| Algorithmic latency | One STFT frame (512 samples = 32 ms at 16 kHz) |
| Network output | 13 channels = 3 mask channels + 10 deep-filter coefficients (5 complex taps) |

---

## 2. Inspiration: GTCRN, and what we changed

**GTCRN** (*Grouped Temporal Convolutional Recurrent Network*, Rong et al., ICASSP 2024) is an ultra-lightweight speech-enhancement model. It simplifies a stronger but heavier model (DPCRN) using grouped convolutions and grouped RNNs, and adds subband feature extraction and temporal recurrent attention to recover quality. That makes it attractive for edge devices. The original authors themselves list degraded performance in low-SNR conditions as a limitation, and low SNR is exactly the regime this project targets.

### What we kept from GTCRN

| Component | Role in the model | Why it is used |
|---|---|---|
| **ERB filterbank** | Compresses 257 linear bins to 129 (65 linear + 64 perceptual bands) | Fewer frequency positions means less compute; ERB spacing follows how human hearing resolves frequency |
| **SFE** (Subband Feature Extraction) | Gives each bin a window of its 3 neighbours | Lets cheap 1×1 convolutions see local frequency context |
| **GTConvBlock** | ShuffleNetV2-style grouped, dilated temporal conv | Large temporal context at a fraction of the cost of full convolutions |
| **TRA** (Temporal Recurrent Attention) | GRU-driven sigmoid gate on channels | Lets the network gate features based on how energy evolves over time |
| **DPGRNN** | Dual-path grouped RNN at the bottleneck | Models both frequency structure and temporal dynamics cheaply |
| **Encoder–decoder with skips** | Downsample → process → upsample | Preserves fine detail while the middle runs at low resolution |

### What we changed

| # | Change | GTCRN / our earlier version | GTCRN-DF (this model) | Target problem |
|---|---|---|---|---|
| 1 | **Input features** | Raw STFT magnitude / real / imag | Level-normalised, compressed features **plus a per-bin signal-to-noise-floor channel** | Absolute magnitudes are meaningless when SNR is −15 dB |
| 2 | **Stationary pre-filter** | Our earlier version ran spectral subtraction before the network | **Removed.** Same noise-floor estimate is now a *feature*, not a subtraction | Pre-subtraction destroys speech that sits below the noise floor |
| 3 | **Mask head** | Unbounded complex mask through a Tanh | **Bounded magnitude (sigmoid) + separate phase rotation, initialised to identity** | Avoids the "output near-silence" failure mode and muffled speech |
| 4 | **Deep filtering** | None | **Complex FIR filter across 5 frames on the low band, added as a zero-initialised residual** | A mask can only attenuate; it cannot recover a bin drowned by noise |
| 5 | **Band-wise TRA** | One gate for the whole frequency axis | Independent gates for `num_bands` frequency chunks, at zero extra parameters | A noise transient in one band should not suppress speech in another |
| 6 | **Capacity** | 16 channels | 32 channels, 3 stacked DPGRNNs, 5 dilations | Separating speech harmonics from noise partials at negative SNR needs more spectral capacity |

Everything else (ERB, SFE, grouped conv blocks, dual-path RNN) is structurally the same as the paper's, with more width and depth.

---

## 3. End-to-end pipeline

```
 noisy waveform (16 kHz)
        │  STFT  (n_fft 512, hop 256, sqrt-Hann)
        ▼
 noisy spectrum  (B, 257, T, 2) ─────────────────────────────────────┐
        │                                                            │
        ▼                                                            │
 ┌─────────────────┐                                                 │
 │  FeatureFront   │  4 level-invariant channels   (B, 4, T, 257)    │
 └────────┬────────┘                                                 │
          ▼                                                          │
   ERB band merge     257 bins → 129 (65 linear + 64 ERB)            │
          ▼                                                          │
   SFE (3 neighbours)                          (B, 12, T, 129)       │
          ▼                                                          │
 ┌────────────────────────── Encoder ─────────────────────────────┐  │
 │ ConvBlock  12→32, freq stride 2                (B,32,T,65)     │  │
 │ ConvBlock  32→32, freq stride 2, groups=2      (B,32,T,33)     │  │
 │ GTConvBlock × 5, dilations 1, 2, 4, 8, 16      (B,32,T,33)     │  │
 └────────────────────────────┬───────────────────────────────────┘  │
                              ▼                                      │
              DPGRNN × 3  (intra-freq BiGRU + inter-time GRU)        │
                              ▼                                      │
 ┌────────────────────────── Decoder ─────────────────────────────┐  │
 │ additive skip connection from the encoder at every level       │  │
 │ GTConvBlock (transposed) × 5, dilations 16, 8, 4, 2, 1         │  │
 │ ConvBlock (transposed), freq stride 2          (B,32,T,65)     │  │
 │ Head: ConvTranspose 32→13, freq stride 2       (B,13,T,129)    │  │
 └───────────────┬─────────────────────────────┬──────────────────┘  │
     3 mask channels (ch 0-2)         10 deep-filter channels (ch 3-12)
              ▼                                   │
   ERB band split → (B,3,T,257)                   │
              ▼                                   │
   magnitude: mask_max · sigmoid(·)               │
   phase:     unit complex rotation               │
              ▼                                   │
   mask × noisy spectrum  ◄───────────────────────┼─────────────────
              ▼                                   │
      masked spectrum ──────► DeepFilter ◄────────┘  (low 64 bins only)
              │                    │
              └──────── + ─────────┘   residual add
                        ▼
              enhanced spectrum  (B, 257, T, 2)
                        │  iSTFT
                        ▼
              enhanced waveform
```

**Tensor layout:** activations are `(B, C, T, F)`: batch, channels, time frames, frequency positions. Convolutions downsample only along frequency; time resolution is never reduced.

---

## 4. Components in detail (and why each exists)

### 4.1 STFT front-end (`train.py`)

The model operates on the complex spectrum. Analysis and synthesis both use a **square-root Hann** window. Their product is a Hann window, which sums to a constant at 50% overlap, so reconstruction is exact and the overlap-add does not colour whatever the mask did.

### 4.2 FeatureFront: level-invariant features *(innovation 1)*

The raw spectrum is converted into **4 channels**, each unchanged if the whole input is scaled by any constant:

| Ch | Feature | Definition |
|---|---|---|
| 0 | Compressed normalised magnitude | `(mag / level) ^ 0.3` |
| 1 | Compressed normalised real part | same gain applied to `real / level` |
| 2 | Compressed normalised imaginary part | same gain applied to `imag / level` |
| 3 | **Local signal-to-noise-floor** | `10·log10(smoothed power / noise floor) / 20`, clamped to `[-1, 2.5]` |

How the pieces are computed (all **causal** and **vectorised**, using left-padded pooling with no per-frame Python loop):

- **Level:** a causal running average of broadband power over 192 frames (~3 s), square-rooted. Dividing by it removes loudness.
- **Noise floor:** per-bin **minimum statistics**. Power is smoothed over 4 frames, then a causal running minimum over 96 frames (~1.5 s) is taken and multiplied by a bias factor (1.6), because a running minimum systematically under-estimates the true noise power.
- **Power-law compression (exponent 0.3):** shrinks the dynamic range so loud bins do not swamp quiet ones, while channels 1–2 still carry phase.

**Why:**
- At −15 dB, the absolute magnitude of a speech bin says almost nothing. Its **ratio to the local stationary noise floor** does, and that is the most useful cue a masking network can be given at low SNR. Previously the network had to infer it from scratch through BatchNorm.
- Level invariance means the mask depends on *relative* levels rather than absolute ones, so the model does not need per-recording gain tuning.
- The window lengths are a trade-off: long enough (~1.5 s) that a spoken utterance cannot drag the noise floor up with it, short enough to follow a genuinely drifting background.

### 4.3 No stationary pre-filter *(innovation 2)*

An earlier design ran spectral subtraction **before** the network. That permanently removes energy the network can never get back. It is tolerable at +10 dB, but destructive at −15 dB, where much of the speech sits at or below the tracked floor. The same floor estimate now enters as **feature channel 3**, so the network receives all the information and loses none of the signal.

### 4.4 ERB filterbank and SFE

- The lowest **65 bins (0 to ~2 kHz) stay at full linear resolution**; the remaining 192 bins are merged into **64 ERB bands** with fixed (non-trainable) triangular filters. The inverse (`bs`, band split) uses the transposed filters.
- Keeping the low band linear is deliberate: those positions stay literal FFT bins, which the deep filter relies on (§4.8).
- **SFE** then unfolds each position with its 2 neighbours, turning 4 feature channels into 12.

**Why:** speech energy and pitch structure are concentrated at low frequencies, so they get full resolution, while the upper spectrum is smooth enough to be described by fewer perceptual bands. This roughly halves the frequency axis (257 → 129) before the network sees anything.

### 4.5 Encoder and the GTConvBlock

The encoder has two strided `ConvBlock`s (12→32 channels; freq width 129 → 65 → 33; the second is grouped, `groups=2`) followed by **5 GTConvBlocks** with time dilations **1, 2, 4, 8, 16**.

Inside one GTConvBlock (ShuffleNetV2-style):

```
 input (32 ch)
   ├── split in half ──────────────────────────────┐
   ▼ (16 ch)                                       │ (16 ch, untouched)
 SFE (3-neighbour unfold)  → 48 ch                 │
 1×1 conv → BN → PReLU     → 32 ch                 │
 causal pad (past only)                            │
 3×3 depthwise dilated conv → BN → PReLU           │
 1×1 conv → BN             → 16 ch                 │
 BandTRA (band-wise attention)                     │
   └──────────── channel shuffle (interleave) ◄────┘
                      ▼
                 output (32 ch)
```

**Why:**
- **Half the channels bypass the block.** This is a large compute saving, and the interleaving shuffle mixes the two halves on the next block.
- **Depthwise + pointwise convolutions** are far cheaper than dense 3×3 convolutions.
- **Dilation 1→16** gives a wide temporal receptive field without a deep stack: each block adds `2·d` past frames, for **62 frames ≈ 1 s** of causal context in total.
- **Causal padding:** the temporal convolution is left-padded only, so it never reads future frames.

### 4.6 BandTRA: band-wise temporal recurrent attention *(innovation 5)*

The original TRA squares and averages over the *whole* frequency axis, so every frequency gets the same gate. **BandTRA** splits the frequency axis into `num_bands` chunks (8 in the training default), computes the mean energy per band per channel, and runs a GRU over time on each band to produce a sigmoid gain.

**Why:** a burst of noise concentrated in the upper bands can now be gated there **without lowering the gain in the bands carrying speech formants in the same frame**. The GRU is *shared* across bands (bands are folded into the batch dimension), so this costs **zero extra parameters**.

### 4.7 DPGRNN: dual-path grouped RNN (bottleneck)

At the most downsampled resolution (33 frequency positions), 3 DPGRNN stages run in sequence. Each has two passes with a residual connection, a fully-connected layer, and LayerNorm:

| Pass | Runs across | Direction | Why |
|---|---|---|---|
| **Intra** | Frequency (per frame) | **Bidirectional** | Frequency is not a time axis: all bins of the current frame already exist, so looking both ways does not break causality |
| **Inter** | Time (per frequency position) | **Unidirectional** | Time must stay strictly causal |

Each RNN is a **grouped RNN**: channels are split in half and processed by two independent, smaller GRUs, which cuts recurrent parameters roughly in half. Stacking three stages is cheap because they run at the smallest frequency width.

### 4.8 Decoder, output head, mask and deep filter

The decoder mirrors the encoder (5 transposed GTConvBlocks with reversed dilations, a transposed ConvBlock, then the head), with an **additive skip connection from the matching encoder level at every stage**. The final layer is a plain linear transposed convolution, with no BatchNorm and no activation, because the head applies the right non-linearity to each output group itself.

The head emits **13 channels**: 3 for the mask, then 10 for the deep filter.

#### Bounded, decoupled, identity-initialised mask *(innovation 3)*

After band-splitting back to 257 bins, the 3 mask channels are turned into a complex mask:

```
magnitude   m_mag = mask_min + (mask_max − mask_min) · sigmoid(ch0)      # in [0, 2]
phase       (pr, pi) = (1 + ch1, ch2)  →  unit vector (pr, pi) / |(pr, pi)|
complex mask M = m_mag · (pr + j·pi) / |pr + j·pi|
enhanced    E = M × noisy spectrum   (complex multiply)
```

| Design choice | Why |
|---|---|
| **Sigmoid-bounded magnitude** in `[0, mask_max=2]` | Bounded gain keeps training stable, and a ceiling of 2 still lets the mask restore bins that were over-attenuated |
| **Magnitude and phase predicted separately** | The phase vector is normalised to unit length, so the phase branch can rotate a bin but never change its loudness; the two jobs do not interfere |
| **Identity initialisation** | `sigmoid(0) × 2 = 1` and phase `(1, 0)` means the untrained model passes input through unchanged. Mask channels get a tiny random init (std 0.01) so gradients still flow |

**Why identity init matters:** with strong suppression terms in the loss, the cheapest early descent direction from a random start is "output near-silence". Networks that fall into that basin produce exactly the muffled, unintelligible speech that has to be avoided. Starting from "pass everything through" removes that trap.

#### Deep filtering *(innovation 4)*

A per-bin mask multiplies each time-frequency bin by *one* complex number. When noise is ~15 dB above speech in a bin, no gain can recover the speech, because the bin's phase is corrupted too. **Deep filtering** instead predicts a short complex FIR filter across recent frames:

```
y[t, f] = Σ_{k=0}^{4}  c_k[t, f] · x[t−k, f]          (complex multiply-accumulate)
```

The model can therefore *reconstruct* a bin from its temporal neighbours (recovering harmonic structure) rather than merely gating it.

- **Order 5** (5 complex taps → 10 coefficient channels), strictly causal (`x[t−k]` only).
- Applied to the **lowest 64 bins (~0 to 2 kHz)**, where F0 and the first two formants live. Those positions are linear FFT bins, which is why the ERB filterbank keeps the low band linear. (A constructor assertion enforces `df_bins ≤ erb_subband_1`.)
- Applied as a **residual on top of the masked spectrum** and **zero-initialised**, so it starts as a no-op and can only add refinement.

**Why:** this is the main mechanism for *recovering* intelligibility, as opposed to just suppressing noise, below 0 dB.

---

## 5. Causality and latency

Every operation is causal by construction:

| Element | How causality is enforced |
|---|---|
| FeatureFront | Left-padded (past-only) running average and running minimum |
| GTConv temporal conv | Left-padded dilated conv; the decoder's transposed conv uses padding `2d`, so each output depends only on frames `t, t−d, t−2d` |
| Inter-frame RNNs / BandTRA GRUs | Unidirectional in time |
| Intra-frame RNN | Bidirectional **across frequency only** (allowed) |
| Deep filter | Uses `x[t], x[t−1], …, x[t−4]` only |

Consequently the **algorithmic latency is one STFT frame** (32 ms at 16 kHz with `n_fft=512`), independent of the dilation schedule. A test in `model.py` (see §8) checks this directly.

---

## 6. Training design

Training is in `train.py`. The model is only as good as its target and loss, so much of the design effort went here.

### 6.1 Data: dynamic mixing with an exact target

Mixing happens **inside the dataloader**: clean speech, noise, and background pools are combined on the fly, and the **clean target is exactly the signal summed into the mixture, at exactly that scale**.

**Why:** an earlier pipeline used a clean file rescaled to a fixed RMS as the target, while the mixture contained that file at a random weight and was then peak-normalised by an unlogged factor. That is a random error of several dB, and it grew worse with louder noise. Every amplitude-sensitive loss was trained against that error, so getting the target exactly right was the single most important fix.

| Setting | Default | Purpose |
|---|---|---|
| Segment length | 4 s @ 16 kHz | Training crop |
| SNR range | **−15 to +20 dB** | Wide coverage, biased to hard cases |
| SNR curriculum | floor starts at −5 dB, ramps to −15 dB over 12 epochs | Learn the easy cases first, then widen |
| Hard-sample fraction | 45% drawn from the hardest 10 dB span | Spend training time where the model fails |
| Burst probability | 55% | Noise placed as sparse bursts (e.g. gunshot-like) rather than continuous |
| Speech-only / noise-only | 6% each | The model must also handle inputs that are only speech or only noise |
| Validation split | 8% of source *files* held out | Validation uses **unseen speakers and unseen noise recordings** |

### 6.2 Loss function

```
L = 1.0·L_spec + 0.25·L_snr + 0.35·L_multires + L_asym
```

| Term | What it does | Why it is designed this way |
|---|---|---|
| **`L_spec`**: compressed complex spectral loss | L1 on `mag^0.3` and on the compressed complex value (50/50), with a **2× weight on 250 Hz–5 kHz** (raised-cosine taper at the edges) | Compression narrows the loud-vs-quiet gap from ~1000× to ~8×, so speech gradients are not drowned by loud noise bins. The complex term drives phase, so output sounds like speech rather than a vocoder. The band is wider than the 300–3400 Hz telephone band because consonants (fricatives, bursts) carry intelligibility up to ~5 kHz, and consonants are lost first at low SNR |
| **`L_snr`**: negative SNR | Soft-clamped at 30 dB, deliberately **not** scale-invariant | SI-SNR cannot see a model that outputs a uniformly *quieter* copy of the right answer, which is exactly the hedge a network adopts under a strong suppression loss. With an exact target, that blindness is not needed |
| **`L_multires`**: multi-resolution STFT | Compressed-magnitude L1 at FFT sizes 256, 512, 1024 | Short windows see transients (gunshot edges, stop consonants); long windows resolve individual harmonics. Using both stops the model trading one for the other |
| **`L_asym`**: activity-weighted asymmetric loss | Three separately normalised pressures, below | Produces strong suppression **without** teaching the model that attenuating everything is the safest option |

**The asymmetric loss** first labels each frame as *speech* or *pause* from the clean reference (a frame is speech if its energy is within 40 dB of the utterance's loudest frame). It then applies three different pressures on the compressed magnitude:

| Pressure | Region | Default weight | Meaning |
|---|---|---|---|
| `keep` | Speech frames | **3.0** | Penalises removing speech that should have stayed (the intelligibility knob) |
| `supp_active` | Speech frames | **1.0** | Penalises leftover noise *inside* speech, deliberately the mildest, since pushing it hard muffles speech |
| `supp_silent` | Pause frames | **2.0 → 8.0** (ramped over 40 epochs) | Penalises leftover noise in pauses, safe to push hard because there is nothing to protect |

Each term is averaged over **its own region**, so the weights keep their meaning regardless of how much of a batch happens to be speech or silence. Applying it in the *compressed* domain is what lets the gradient actually reach quiet speech.

### 6.3 Optimisation

| Component | Choice | Why |
|---|---|---|
| Optimiser | **AdamW**, lr 1e-3, betas (0.9, 0.98), weight decay 1e-5 on weight matrices only | Decay is not applied to biases and norm parameters |
| LR schedule | 3-epoch linear warm-up, then cosine decay to 2% of peak | Stable start, smooth convergence |
| Gradient clipping | Norm 5.0 | Recurrent layers plus a noisy objective can spike |
| **EMA of weights** | Decay 0.999; **EMA weights are evaluated and checkpointed** | Nearly free, and reliably gives a little extra quality when the mixing is random each step |
| Mixed precision | Auto (bf16 → fp16 → off) | Model forward runs in reduced precision; **iSTFT and all losses run in float32** |
| Batch / schedule | Batch 16, 400 steps/epoch, 120 epochs | Defaults; all overridable from the command line |

### 6.4 Evaluation and checkpoints

- Validation reports the **SI-SDR improvement bucketed by input SNR** (`<−10`, `−10..−5`, `−5..0`, `0..5`, `5..15`, `>15` dB), so failures at a specific SNR are visible rather than hidden in a single average.
- It also reports **pause-time noise reduction (dB)**: how much energy in speech-free frames actually disappeared.
- `best_model.pt` is saved on the lowest validation loss and stores both the EMA weights (`model_state_dict`) and the raw weights (`raw_state_dict`), plus the model config so the network can be rebuilt with `build_model_from_config`.
- An optional fixed, pre-mixed held-out set can be supplied via `--eval_root`.

---

## 7. Configuration and model size

Main hyper-parameters (`GTCRN(...)` in `model.py`; `train.py` exposes them as command-line flags):

| Parameter | Default | Meaning |
|---|---|---|
| `sample_rate` / `n_fft` | 16000 / 512 | Must match the STFT feeding the model |
| `erb_subband_1` / `erb_subband_2` | 65 / 64 | Linear low bins / ERB bands above them |
| `base_channels` | 32 | Network width (multiple of 4; the paper uses 16) |
| `n_dpgrnn` | 3 | Stacked dual-path RNN stages |
| `dilations` | (1, 2, 4, 8, 16) | One causal GTConvBlock per entry |
| `tra_bands` | 4 in `model.py`, **8 in `train.py`** | Independent attention bands (parameter-free) |
| `df_order` / `df_bins` | 5 / 64 | Deep-filter taps / low bins covered (`df_bins ≤ 65`) |
| `mask_max` / `mask_min` | 2.0 / 0.0 | Mask magnitude ceiling / floor. Keep `mask_max=2` for the identity init to hold |
| `compress` | 0.3 | Power-law exponent (shared with the loss's `--power`) |

`mask_min` can be set to a small value (e.g. 0.02) **at inference only** to leave a slight noise floor, which some listeners prefer to absolute silence between words.

**Approximate size** (trainable parameters, computed from the layer definitions):

| Preset | Configuration | Trainable params |
|---|---|---|
| Paper-style | 16 ch, 2 DPGRNN, dilations (1, 2, 5), no deep filter | ≈ 24 K |
| **Default (this project)** | 32 ch, 3 DPGRNN, 5 dilations, DF 5×64 | **≈ 125 K** |
| Large | 40 ch, 4 DPGRNN, DF 5×64 | ≈ 204 K |

The fixed ERB filterbank adds another ≈ 24.6 K *non-trainable* weights. Running `python model.py` prints the exact counts for each preset.

Basic usage:

```python
import torch
from model import GTCRN

model = GTCRN().eval()
spec = torch.randn(1, 257, 100, 2)      # (B, F, T, 2) real/imag STFT
with torch.no_grad():
    enhanced = model(spec)              # same shape as the input
```

To train (from the project root):

```bash
python train.py --data_root sample_data --epochs 120
```

---

## 8. Built-in sanity tests

Running `python model.py` executes three checks that verify the design claims above:

| Test | What it verifies |
|---|---|
| **Identity at initialisation** | An untrained model passes the input through almost unchanged (relative deviation should be well under 0.1) |
| **Causality** | Changing the last frame does not alter any earlier frame |
| **Level invariance** | A 20× louder input produces a 20× louder output (small relative error) |

---

## 9. References and related work

1. X. Rong, T. Sun, X. Zhang, Y. Hu, C. Zhu, J. Lu, *"GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources,"* ICASSP 2024. Code (MIT licence): https://github.com/Xiaobin-Rong/gtcrn
2. H. Schröter et al., *"DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering,"* ICASSP 2022. (Related work: deep filtering.)
3. R. Martin, *"Noise Power Spectral Density Estimation Based on Optimal Smoothing and Minimum Statistics,"* IEEE Trans. Speech and Audio Processing, 2001. (Related technique: minimum-statistics noise tracking.)