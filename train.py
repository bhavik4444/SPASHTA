"""
Training for GTCRN-DF — adaptive noise cancellation for defence audio.

WHAT CHANGED AND WHY
====================

1. THE TARGET IS NOW CORRECT
   Mixing moved into the dataloader (dataset.py). The clean target is the exact
   signal summed into the mixture, at the exact scale. Previously the target was
   the clean file RMS-normalised to 0.1, while the mixture contained that file
   scaled by a random weight and then peak-normalised by an unlogged factor —
   a random several-dB error that grew worse the louder the noise was. Every
   amplitude-sensitive loss was being trained against that error. This is the
   fix that matters most; nothing else here works without it.

2. ASYMMETRIC LOSSES MOVED TO THE COMPRESSED DOMAIN
   The old suppression penalty operated on raw STFT magnitudes. At -10 dB SNR a
   single gunshot bin can be a thousand times larger than a speech harmonic, so
   raw-magnitude gradients are dominated almost entirely by the loud noise bins
   and the quiet speech the model is supposed to preserve contributes almost
   nothing. Applying the same asymmetry to magnitude**0.3 puts loud and quiet
   bins within a factor of ~8 of each other instead of ~1000, so the gradient
   actually reaches the speech.

3. SUPPRESSION AND PRESERVATION ARE SEPARATED BY FRAME ACTIVITY
   Instead of one global under/over weighting, frames are classified by whether
   the clean reference has speech in them, and three different pressures are
   applied: protect speech energy in speech frames, remove residual noise in
   speech frames gently, remove residual noise in pauses hard. This is what
   produces "very high suppression" without the model learning that the safest
   way to score well is to attenuate everything. Each term is averaged over its
   own region, so the weights keep their meaning regardless of how much of a
   batch happens to be speech.

4. THE OLD LOSS ZOO IS GONE
   wav_l1, si_snr, the windowed gain-matching term and the early/late weight
   schedule were all compensating, in different directions, for the broken
   target. With a correct target they are redundant and they fight each other.
   What remains: power-compressed complex spectral loss, a soft-clamped SNR
   loss (not scale-invariant — the scale is meaningful now), a multi-resolution
   magnitude loss, and the activity-weighted asymmetric term.

5. TRAINING MECHANICS
   AdamW with warmup + cosine decay, an EMA of the weights for evaluation and
   checkpointing, mixed precision, and validation bucketed by input SNR so you
   can see exactly where the model is failing instead of watching one number.

Run it from the project root:

    python train.py --data_root sample_data --epochs 120

See the bottom of this file's docstring in the accompanying notes for a full
recommended command.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DynamicMixDataset, PremixedPairDataset, prepare_pools
from model import GTCRN


# ===========================================================================
# Reproducibility
# ===========================================================================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# STFT front-end
# ===========================================================================
class STFTFrontEnd(nn.Module):
    """Square-root Hann for both analysis and synthesis. Their product is a
    Hann window, which sums to a constant at 50% overlap, so masking happens in
    a domain where reconstruction is exact and the overlap-add does not colour
    whatever the mask did."""

    def __init__(self, n_fft=512, hop_length=256, win_length=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        win = torch.hann_window(self.win_length, periodic=True).clamp_min(0.0).sqrt()
        self.register_buffer("window", win, persistent=False)

    def stft(self, wav):
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window,
                          center=True, return_complex=True)
        return torch.view_as_real(spec)

    def istft(self, spec, length=None):
        spec_c = torch.view_as_complex(spec.contiguous())
        return torch.istft(spec_c, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length, window=self.window,
                           center=True, length=length)


# ===========================================================================
# Losses
# ===========================================================================
def make_speech_band_weight(n_fft, sample_rate, low_hz=250.0, high_hz=5000.0,
                            weight=2.0, taper_hz=250.0, device="cpu"):
    """(F,) weights: `weight` inside the band, 1.0 outside, raised-cosine taper
    at the edges. The band is wider than the old 300-3400 telephone range
    because fricatives and stop bursts carry a lot of consonant intelligibility
    up to ~5 kHz, and consonants are what get lost first at low SNR."""
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_freqs, device=device)
    if weight == 1.0:
        return torch.ones(n_freqs, device=device)

    w = torch.ones(n_freqs, device=device)
    w[(freqs >= low_hz) & (freqs <= high_hz)] = weight

    def taper(center, rising):
        lo, hi = center - taper_hz / 2, center + taper_hz / 2
        sel = (freqs >= lo) & (freqs <= hi)
        frac = ((freqs[sel] - lo) / max(hi - lo, 1e-8)).clamp(0, 1)
        ramp = 0.5 - 0.5 * torch.cos(frac * math.pi)
        if not rising:
            ramp = 1.0 - ramp
        w[sel] = 1.0 + ramp * (weight - 1.0)

    taper(low_hz, rising=True)
    taper(high_hz, rising=False)
    return w


def _compress(spec, power=0.3, eps=1e-8):
    """Returns (compressed magnitude, compressed complex as real 2-vector)."""
    c = torch.view_as_complex(spec.contiguous())
    mag = c.abs().clamp_min(eps)
    mag_c = mag.pow(power)
    comp = c / mag * mag_c
    return mag_c, torch.view_as_real(comp)


def compressed_spectral_loss(pred_spec, clean_spec, power=0.3, fw=None):
    """L1 on power-compressed magnitude and on the compressed complex value.
    The magnitude term drives suppression; the complex term drives phase, which
    is what makes the result sound like speech rather than like a vocoder."""
    pm, pc = _compress(pred_spec, power)
    cm, cc = _compress(clean_spec, power)
    if fw is None:
        return 0.5 * F.l1_loss(pm, cm) + 0.5 * F.l1_loss(pc, cc)
    w = fw.view(1, -1, 1)
    mag_loss = (w * (pm - cm).abs()).mean()
    cplx_loss = (w.unsqueeze(-1) * (pc - cc).abs()).mean()
    return 0.5 * mag_loss + 0.5 * cplx_loss


def frame_activity(clean_spec, rel_db=-40.0, eps=1e-12):
    """(B,T) float mask: 1 where the clean reference has speech. Threshold is
    relative to each utterance's own loudest frame, so it adapts to level."""
    mag = torch.view_as_complex(clean_spec.contiguous()).abs()
    e = mag.pow(2).sum(dim=1)                                  # (B,T)
    ref = e.max(dim=1, keepdim=True).values
    thr = ref * (10.0 ** (rel_db / 10.0))
    return ((e > thr) & (ref > eps)).float()


def asymmetric_loss(pred_spec, clean_spec, act, power=0.3, fw=None,
                    keep_w=3.0, supp_active=1.0, supp_silent=6.0, eps=1e-8):
    """Three separately-normalised pressures on the compressed magnitude:

      keep_w       cost of removing speech energy that should have stayed
                   (clean > pred), counted only in speech frames and weighted
                   toward the intelligibility band
      supp_active  cost of leftover noise inside speech frames — deliberately
                   the mildest term, because this is the one that damages
                   speech when it is pushed too hard
      supp_silent  cost of leftover noise in pauses — push this as hard as you
                   like, there is nothing there to protect

    Each term is averaged over its own region rather than over the whole
    tensor, so a batch that happens to be 80% silence does not silently
    multiply the suppression pressure relative to a batch that is 20% silence.
    """
    pm, _ = _compress(pred_spec, power, eps)
    cm, _ = _compress(clean_spec, power, eps)

    diff = cm - pm
    under = diff.clamp_min(0.0)      # model removed speech
    over = (-diff).clamp_min(0.0)    # model left noise in

    a = act.unsqueeze(1)             # (B,1,T)
    s = 1.0 - a
    n_act = a.sum() * pm.shape[1]
    n_sil = s.sum() * pm.shape[1]

    w = fw.view(1, -1, 1) if fw is not None else 1.0
    under_term = (w * under * a).sum() / n_act.clamp_min(1.0)
    over_act = (over * a).sum() / n_act.clamp_min(1.0)
    over_sil = (over * s).sum() / n_sil.clamp_min(1.0)

    return (keep_w * under_term + supp_active * over_act + supp_silent * over_sil,
            under_term.detach(), over_act.detach(), over_sil.detach())


def snr_loss(pred_wav, clean_wav, valid, tau=1e-3, eps=1e-10):
    """Negative SNR, soft-clamped at 10*log10(1/tau) = 30 dB.

    Not scale-invariant: SI-SNR cannot see a model that outputs a uniformly
    quieter copy of the right answer, which is precisely the hedge a network
    adopts when a strong suppression term is in the loss. With an exact target
    there is no reason to tolerate that blindness.
    """
    num = clean_wav.pow(2).sum(dim=-1)
    err = (clean_wav - pred_wav).pow(2).sum(dim=-1)
    ratio = num / (err + tau * num + eps)
    l = -10.0 * torch.log10(ratio + eps)
    denom = valid.sum().clamp_min(1.0)
    return (l * valid).sum() / denom


class MultiResSTFTLoss(nn.Module):
    """Compressed magnitude L1 at several resolutions. The short window sees
    transients (gunshot edges, stop consonants) that a 512-point window smears;
    the long window resolves individual harmonics. Training on both keeps the
    model from trading one for the other."""

    def __init__(self, ffts=(256, 512, 1024), power=0.3):
        super().__init__()
        self.ffts = tuple(ffts)
        self.hops = tuple(n // 4 for n in self.ffts)
        self.power = power
        for n in self.ffts:
            self.register_buffer(f"win_{n}", torch.hann_window(n), persistent=False)

    def forward(self, pred_wav, clean_wav):
        total = 0.0
        for n, hop in zip(self.ffts, self.hops):
            win = getattr(self, f"win_{n}")
            p = torch.stft(pred_wav, n, hop, window=win, center=True,
                           return_complex=True).abs().clamp_min(1e-8).pow(self.power)
            c = torch.stft(clean_wav, n, hop, window=win, center=True,
                           return_complex=True).abs().clamp_min(1e-8).pow(self.power)
            total = total + F.l1_loss(p, c)
        return total / len(self.ffts)


def si_sdr(pred, clean, eps=1e-10):
    """Reporting metric only (per item, in dB)."""
    p = pred - pred.mean(dim=-1, keepdim=True)
    c = clean - clean.mean(dim=-1, keepdim=True)
    proj = (torch.sum(p * c, dim=-1, keepdim=True) /
            (torch.sum(c * c, dim=-1, keepdim=True) + eps)) * c
    noise = p - proj
    return 10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


# ===========================================================================
# EMA
# ===========================================================================
class ModelEMA:
    """Exponential moving average of the weights. Evaluating and shipping the
    averaged weights instead of the last iterate is close to free and reliably
    buys a little quality, especially with a noisy dynamic-mixing objective."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow, self.other = {}, {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone().float()
            else:
                self.other[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.other[k] = v.detach().clone()

    def state_dict(self):
        sd = {k: v.clone() for k, v in self.shadow.items()}
        sd.update({k: v.clone() for k, v in self.other.items()})
        return sd


# ===========================================================================
# One epoch
# ===========================================================================
def compute_losses(pred_spec, clean_spec, pred_wav, clean_wav, act, mrstft, fw, cfg):
    valid = (clean_wav.pow(2).sum(dim=-1) > 1e-7).float()

    l_spec = compressed_spectral_loss(pred_spec, clean_spec, cfg["power"], fw)
    l_asym, t_under, t_over_a, t_over_s = asymmetric_loss(
        pred_spec, clean_spec, act, cfg["power"], fw,
        keep_w=cfg["keep_w"], supp_active=cfg["supp_active"],
        supp_silent=cfg["supp_silent"])
    l_snr = snr_loss(pred_wav, clean_wav, valid)
    l_mr = mrstft(pred_wav, clean_wav)

    total = (cfg["spec_w"] * l_spec + cfg["snr_w"] * l_snr
             + cfg["mr_w"] * l_mr + l_asym)
    parts = {"spec": l_spec.detach(), "snr": l_snr.detach(), "mr": l_mr.detach(),
             "asym": l_asym.detach(), "keep": t_under, "res_sp": t_over_a,
             "res_sil": t_over_s}
    return total, parts


def train_one_epoch(model, loader, stft, mrstft, optimizer, scaler, ema, device,
                    fw, cfg, sched_fn, global_step, log_every=50):
    model.train()
    agg, n = {}, 0
    t0 = time.time()

    for noisy_wav, clean_wav, _snr in loader:
        lr = sched_fn(global_step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        noisy_wav = noisy_wav.to(device, non_blocking=True)
        clean_wav = clean_wav.to(device, non_blocking=True)

        noisy_spec = stft.stft(noisy_wav)
        clean_spec = stft.stft(clean_wav)
        act = frame_activity(clean_spec, cfg["act_rel_db"])

        with torch.autocast(device_type=device.type, dtype=cfg["amp_dtype"],
                            enabled=cfg["amp"]):
            pred_spec = model(noisy_spec)
        pred_spec = pred_spec.float()
        pred_wav = stft.istft(pred_spec, length=noisy_wav.shape[-1])

        total, parts = compute_losses(pred_spec, clean_spec, pred_wav, clean_wav,
                                      act, mrstft, fw, cfg)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            optimizer.step()

        if ema is not None:
            ema.update(model)

        agg["total"] = agg.get("total", 0.0) + float(total.detach())
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1
        global_step += 1

        if n % log_every == 0:
            rate = n / max(time.time() - t0, 1e-6)
            print(f"    step {n:>5}/{len(loader)}  loss {agg['total']/n:.4f}  "
                  f"lr {lr:.2e}  {rate:.1f} it/s", flush=True)

    n = max(n, 1)
    return {k: v / n for k, v in agg.items()}, global_step


SNR_BUCKETS = [(-100, -10, "<-10"), (-10, -5, "-10..-5"), (-5, 0, "-5..0"),
               (0, 5, "0..5"), (5, 15, "5..15"), (15, 100, ">15")]


@torch.no_grad()
def evaluate(model, loader, stft, mrstft, device, fw, cfg):
    model.eval()
    agg, n = {}, 0
    buckets = {name: {"sisdr": 0.0, "imp": 0.0, "n": 0} for *_, name in SNR_BUCKETS}
    sil_pred, sil_mix = 0.0, 0.0

    for noisy_wav, clean_wav, snr in loader:
        noisy_wav = noisy_wav.to(device, non_blocking=True)
        clean_wav = clean_wav.to(device, non_blocking=True)

        noisy_spec = stft.stft(noisy_wav)
        clean_spec = stft.stft(clean_wav)
        act = frame_activity(clean_spec, cfg["act_rel_db"])

        pred_spec = model(noisy_spec).float()
        pred_wav = stft.istft(pred_spec, length=noisy_wav.shape[-1])

        total, parts = compute_losses(pred_spec, clean_spec, pred_wav, clean_wav,
                                      act, mrstft, fw, cfg)
        agg["total"] = agg.get("total", 0.0) + float(total)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1

        # how much of the pause-time noise actually disappeared
        sil = (1.0 - act).unsqueeze(1)
        pmag = torch.view_as_complex(pred_spec.contiguous()).abs().pow(2)
        nmag = torch.view_as_complex(noisy_spec.contiguous()).abs().pow(2)
        sil_pred += float((pmag * sil).sum())
        sil_mix += float((nmag * sil).sum())

        valid = clean_wav.pow(2).sum(dim=-1) > 1e-7
        if valid.any():
            out_sdr = si_sdr(pred_wav, clean_wav)
            in_sdr = si_sdr(noisy_wav, clean_wav)
            for i in range(noisy_wav.shape[0]):
                if not bool(valid[i]):
                    continue
                s = float(snr[i])
                for lo, hi, name in SNR_BUCKETS:
                    if lo <= s < hi:
                        buckets[name]["sisdr"] += float(out_sdr[i])
                        buckets[name]["imp"] += float(out_sdr[i] - in_sdr[i])
                        buckets[name]["n"] += 1
                        break

    n = max(n, 1)
    out = {k: v / n for k, v in agg.items()}
    out["sil_supp_db"] = 10.0 * math.log10(max(sil_pred, 1e-20) / max(sil_mix, 1e-20))
    out["buckets"] = {k: {"sisdr": v["sisdr"] / max(v["n"], 1),
                          "imp": v["imp"] / max(v["n"], 1),
                          "n": v["n"]} for k, v in buckets.items()}
    return out


# ===========================================================================
# Main
# ===========================================================================
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- data ----
    p.add_argument("--data_root", type=str, default="sample_data",
                   help="folder containing clean/, noise/ and bg/ subfolders "
                        "(scanned recursively)")
    p.add_argument("--cache_dir", type=str, default=".cache_16k",
                   help="where the decoded/resampled 16 kHz mono cache is kept")
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--val_fraction", type=float, default=0.08,
                   help="fraction of source FILES held out for validation, so "
                        "validation speakers and noise recordings are unseen")
    p.add_argument("--eval_root", type=str, default=None,
                   help="optional folder written by sound_mixer.py (mixed/, "
                        "target/, csvs/mix_log.csv) used as a fixed held-out set")

    # ---- mixing ----
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--segment_seconds", type=float, default=4.0)
    p.add_argument("--snr_min", type=float, default=-15.0)
    p.add_argument("--snr_max", type=float, default=20.0)
    p.add_argument("--snr_min_start", type=float, default=-5.0,
                   help="curriculum: snr_min at epoch 1, ramped down to --snr_min "
                        "over --curriculum_epochs. Set equal to --snr_min to disable.")
    p.add_argument("--curriculum_epochs", type=int, default=12)
    p.add_argument("--hard_frac", type=float, default=0.45,
                   help="fraction of samples drawn from the hardest --hard_span dB")
    p.add_argument("--hard_span", type=float, default=10.0)
    p.add_argument("--p_speech_only", type=float, default=0.06)
    p.add_argument("--p_noise_only", type=float, default=0.06)
    p.add_argument("--p_burst", type=float, default=0.55,
                   help="probability a noise source is placed as sparse bursts "
                        "rather than continuously")
    p.add_argument("--mix_rms", type=float, default=0.1)

    # ---- model ----
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--n_dpgrnn", type=int, default=3)
    p.add_argument("--tra_bands", type=int, default=8)
    p.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--df_order", type=int, default=5,
                   help="deep filter taps; 0 disables deep filtering")
    p.add_argument("--df_bins", type=int, default=64,
                   help="how many low linear bins the deep filter covers "
                        "(must be <= 65)")
    p.add_argument("--mask_max", type=float, default=2.0)
    p.add_argument("--mask_min", type=float, default=0.0)

    # ---- optimisation ----
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--steps_per_epoch", type=int, default=400)
    p.add_argument("--val_items", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr_ratio", type=float, default=0.02)
    p.add_argument("--warmup_epochs", type=float, default=3.0)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="0 disables EMA")
    p.add_argument("--amp", type=str, default="auto", choices=["auto", "bf16", "fp16", "off"])
    p.add_argument("--num_workers", type=int, default=4)

    # ---- loss weights ----
    p.add_argument("--power", type=float, default=0.3,
                   help="magnitude compression exponent used by every spectral term")
    p.add_argument("--spec_weight", type=float, default=1.0)
    p.add_argument("--snr_weight", type=float, default=0.25)
    p.add_argument("--mrstft_weight", type=float, default=0.35)
    p.add_argument("--keep_weight", type=float, default=3.0,
                   help="cost of removing speech that should have stayed; this is "
                        "the intelligibility knob")
    p.add_argument("--supp_active", type=float, default=1.0,
                   help="cost of residual noise inside speech frames; raising this "
                        "far above 1 is what makes speech muffled")
    p.add_argument("--supp_silent_start", type=float, default=2.0)
    p.add_argument("--supp_silent_end", type=float, default=8.0,
                   help="cost of residual noise in pauses at the end of training; "
                        "this is the suppression knob and is safe to push hard")
    p.add_argument("--supp_ramp_epochs", type=int, default=40)
    p.add_argument("--act_rel_db", type=float, default=-40.0,
                   help="frame is 'speech' if within this many dB of the "
                        "utterance's loudest frame")
    p.add_argument("--band_low_hz", type=float, default=250.0)
    p.add_argument("--band_high_hz", type=float, default=5000.0)
    p.add_argument("--band_weight", type=float, default=2.0)

    # ---- bookkeeping ----
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init_checkpoint", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    # ------------------------------------------------------------- data ----
    print("\nPreparing data pools")
    pools_tr, pools_va = prepare_pools(args.data_root, args.cache_dir,
                                       args.sample_rate, args.val_fraction,
                                       rebuild_cache=args.rebuild_cache)

    common = dict(sample_rate=args.sample_rate, segment_seconds=args.segment_seconds,
                  snr_min=args.snr_min_start, snr_max=args.snr_max,
                  hard_frac=args.hard_frac, hard_span=args.hard_span,
                  p_speech_only=args.p_speech_only, p_noise_only=args.p_noise_only,
                  p_burst=args.p_burst, mix_rms=args.mix_rms)

    train_ds = DynamicMixDataset(pools_tr["clean"], pools_tr["noise"], pools_tr["bg"],
                                 length=args.steps_per_epoch * args.batch_size,
                                 deterministic=False, seed=args.seed, **common)
    val_common = dict(common)
    val_common["snr_min"] = args.snr_min          # validation always covers the full range
    val_ds = DynamicMixDataset(pools_va["clean"], pools_va["noise"], pools_va["bg"],
                               length=args.val_items, deterministic=True,
                               seed=args.seed + 777, **val_common)

    # persistent_workers is off on purpose: the curriculum mutates the dataset's
    # snr_min between epochs, and workers only pick that up when they respawn.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                              drop_last=True, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(1, args.num_workers // 2),
                            pin_memory=(device.type == "cuda"), drop_last=False)

    eval_loader = None
    if args.eval_root:
        csv_path = Path(args.eval_root) / "csvs" / "mix_log.csv"
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            eval_ds = PremixedPairDataset(args.eval_root, rows, args.sample_rate,
                                          args.segment_seconds, args.mix_rms)
            eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=2)
            print(f"held-out pre-mixed evaluation set: {len(eval_ds)} file(s)")
        else:
            print(f"warning: --eval_root given but {csv_path} not found; skipping")

    # ------------------------------------------------------------ model ----
    model = GTCRN(sample_rate=args.sample_rate, n_fft=args.n_fft,
                  tra_bands=args.tra_bands, base_channels=args.base_channels,
                  n_dpgrnn=args.n_dpgrnn, dilations=tuple(args.dilations),
                  df_order=args.df_order, df_bins=args.df_bins,
                  mask_max=args.mask_max, mask_min=args.mask_min,
                  compress=args.power).to(device)
    n_par = sum(q.numel() for q in model.parameters())
    n_tr = sum(q.numel() for q in model.parameters() if q.requires_grad)
    rf = 2 * sum(args.dilations)
    print(f"\nGTCRN-DF: {n_par:,} params ({n_tr:,} trainable) | "
          f"channels={args.base_channels} dpgrnn={args.n_dpgrnn} "
          f"df={args.df_order}x{args.df_bins} | causal context "
          f"{rf} frames (~{rf * args.hop_length / args.sample_rate * 1000:.0f} ms)")

    if args.init_checkpoint:
        ck = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        print(f"resumed weights from {args.init_checkpoint}")

    stft = STFTFrontEnd(args.n_fft, args.hop_length).to(device)
    mrstft = MultiResSTFTLoss(power=args.power).to(device)
    fw = make_speech_band_weight(args.n_fft, args.sample_rate, args.band_low_hz,
                                 args.band_high_hz, args.band_weight, device=device)

    decay, no_decay = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        (no_decay if prm.ndim <= 1 else decay).append(prm)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))

    if args.amp == "auto":
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_mode = "bf16" if use_bf16 else ("fp16" if device.type == "cuda" else "off")
    else:
        amp_mode = args.amp
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(device.type) if amp_mode == "fp16" else None
    print(f"mixed precision: {amp_mode}")

    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    def sched_fn(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        prog = min(max(prog, 0.0), 1.0)
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cos)

    cfg = {"power": args.power, "spec_w": args.spec_weight, "snr_w": args.snr_weight,
           "mr_w": args.mrstft_weight, "keep_w": args.keep_weight,
           "supp_active": args.supp_active, "supp_silent": args.supp_silent_start,
           "act_rel_db": args.act_rel_db, "clip": args.clip,
           "amp": amp_mode != "off", "amp_dtype": amp_dtype}

    def build_config():
        return {"sample_rate": args.sample_rate, "n_fft": args.n_fft,
                "hop_length": args.hop_length, "tra_bands": args.tra_bands,
                "base_channels": args.base_channels, "n_dpgrnn": args.n_dpgrnn,
                "dilations": list(args.dilations), "df_order": args.df_order,
                "df_bins": args.df_bins, "mask_max": args.mask_max,
                "mask_min": args.mask_min, "compress": args.power,
                "mix_rms": args.mix_rms}

    (out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2))

    best = float("inf")
    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # curriculum: widen the SNR range downward, then harden suppression
        if args.curriculum_epochs > 0:
            f = min(1.0, (epoch - 1) / max(args.curriculum_epochs, 1))
            train_ds.snr_min = args.snr_min_start + f * (args.snr_min - args.snr_min_start)
        else:
            train_ds.snr_min = args.snr_min
        f2 = min(1.0, (epoch - 1) / max(args.supp_ramp_epochs, 1))
        cfg["supp_silent"] = (args.supp_silent_start
                              + f2 * (args.supp_silent_end - args.supp_silent_start))

        print(f"\nepoch {epoch:03d}/{args.epochs}  "
              f"snr_min={train_ds.snr_min:.1f}dB  supp_silent={cfg['supp_silent']:.2f}")
        tr, global_step = train_one_epoch(model, train_loader, stft, mrstft, optimizer,
                                          scaler, ema, device, fw, cfg, sched_fn,
                                          global_step)

        eval_model = model
        if ema is not None:
            eval_model = GTCRN(sample_rate=args.sample_rate, n_fft=args.n_fft,
                               tra_bands=args.tra_bands, base_channels=args.base_channels,
                               n_dpgrnn=args.n_dpgrnn, dilations=tuple(args.dilations),
                               df_order=args.df_order, df_bins=args.df_bins,
                               mask_max=args.mask_max, mask_min=args.mask_min,
                               compress=args.power).to(device)
            eval_model.load_state_dict(ema.state_dict())

        va = evaluate(eval_model, val_loader, stft, mrstft, device, fw, cfg)

        bstr = "  ".join(f"{k}:{v['imp']:+.1f}dB(n{v['n']})"
                         for k, v in va["buckets"].items() if v["n"] > 0)
        print(f"  train {tr['total']:.4f} | val {va['total']:.4f} "
              f"(spec {va['spec']:.4f} snr {va['snr']:.3f} mr {va['mr']:.4f} "
              f"keep {va['keep']:.4f} res_sp {va['res_sp']:.4f} res_sil {va['res_sil']:.4f})")
        print(f"  pause-time noise reduction: {va['sil_supp_db']:.1f} dB")
        print(f"  SI-SDR improvement by input SNR: {bstr}")

        history.append({"epoch": epoch, "train": tr["total"], "val": va["total"],
                        "sil_supp_db": va["sil_supp_db"],
                        **{f"val_{k}": va[k] for k in
                           ("spec", "snr", "mr", "keep", "res_sp", "res_sil")},
                        **{f"imp_{k}": v["imp"] for k, v in va["buckets"].items()}})

        if va["total"] < best:
            best = va["total"]
            torch.save({"model_state_dict": (ema.state_dict() if ema is not None
                                             else model.state_dict()),
                        "raw_state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": best, "config": build_config()},
                       out_dir / "best_model.pt")
            print(f"  -> new best (val {best:.4f}), checkpoint saved")

        torch.save({"model_state_dict": (ema.state_dict() if ema is not None
                                         else model.state_dict()),
                    "raw_state_dict": model.state_dict(),
                    "epoch": epoch, "val_loss": va["total"], "config": build_config()},
                   out_dir / "last_model.pt")

        keys = sorted({k for h in history for k in h})
        with open(out_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(history)

    print(f"\nDone. best val loss {best:.4f}. Checkpoints in {out_dir}/")

    if eval_loader is not None:
        ck = torch.load(out_dir / "best_model.pt", map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        ev = evaluate(model, eval_loader, stft, mrstft, device, fw, cfg)
        bstr = "  ".join(f"{k}:{v['imp']:+.1f}dB(n{v['n']})"
                         for k, v in ev["buckets"].items() if v["n"] > 0)
        print(f"Held-out pre-mixed set: loss {ev['total']:.4f} | "
              f"pause-time reduction {ev['sil_supp_db']:.1f} dB")
        print(f"  SI-SDR improvement by input SNR: {bstr}")


if __name__ == "__main__":
    main()