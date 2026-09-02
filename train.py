"""
Training script for GTCRN on a pre-mixed noisy-speech dataset (SIH26052 ANC).

WHAT CHANGED FROM THE FIRST VERSION
------------------------------------
Mixing is no longer done here -- sound_mixer.py (the dataset team's script)
already generated every noisy/clean pair and logged the exact recipe for
each one in csvs/mix_log.csv. This script just reads that CSV, loads the
matching (mixed, clean) pair for each row, and trains on it directly.

Expected project layout (train.py lives at the project root, next to the
dataset folder -- matches the SIH/ project tree with sample_data/ nested
inside it):

    SIH/                        (project root -- run train.py from here)
      train.py, infer.py, model.py, main.py
      checkpoints/               (created automatically)
      sample_data/                <- this is --data_root, defaults to "sample_data"
        mixed_dataset/   1.wav ... 200.wav      (already-mixed noisy audio)
        clean/           Clean clip1.wav, ...   (ground-truth clean speech)
        csvs/mix_log.csv                        (recipe for every mixed file)
        bg/, noise/, sound_mixer.py             (dataset team's mixing inputs/script)

By default, files are split by their numeric filename:
    1..100    -> training
    101..150  -> validation (drives checkpoint selection + LR schedule)
    151..200  -> held-out "extreme" set (-5dB floor), reported once at the
                 very end as a robustness number -- never trained or tuned on

That split matches the CSV's own "difficulty" column (normal = rows 1..150,
extreme = rows 151..200), with the first 150 further cut 100/50 into
train/val. Adjust --train_end / --val_end if you want a different split.

The clean target for each mixed file is loaded fresh from sample_data/clean/
and RMS-normalized to --clean_target_rms (0.1 by default, matching
sound_mixer.py's TARGET_INPUT_RMS) -- this puts every target on the same
loudness footing sound_mixer.py used, though it won't be bit-exact to the
scaled copy actually summed inside the mix (the mixer's final peak
normalization factor isn't logged in the CSV, so exact reconstruction isn't
possible -- this is the standard approach used by most speech-enhancement
datasets anyway: an independently-normalized clean reference, not a bit-exact
sub-component of the mix).

Run it from the project root (SIH/), with no arguments needed if your
dataset really is at sample_data/ and you're happy with the defaults:
    python train.py --epochs 30 --out_dir checkpoints

LOSS-WEIGHT SCHEDULE (optional)
--------------------------------
--wav_l1_weight / --supp_weight / --supp_under_weight / --supp_over_weight
set the EARLY-phase values used from epoch 1. Each has a *_late twin
(e.g. --supp_under_weight_late) that only takes effect once --schedule_epoch
is reached, letting the model first learn general enhancement with gentler
weights before the loss is pushed harder toward "never remove speech, even
if noise leaks through" for the back half of training. Omit --schedule_epoch
to just use the early values for the whole run (the old, unscheduled
behaviour).
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

from model import GTCRN


# --------------------------------------------------------------------------
# 0. Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# 1. Audio I/O helpers
# --------------------------------------------------------------------------
def load_audio(path, sample_rate):
    """Load a wav/mp3/etc file as a mono float32 tensor at the target sample rate."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = torch.from_numpy(wav)
    if wav.dim() > 1:                       # stereo -> mono
        wav = wav.mean(dim=-1)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.contiguous()


def save_audio(path, wav, sample_rate):
    wav = wav.detach().cpu().clamp(-1.0, 1.0).numpy()
    sf.write(str(path), wav, sample_rate)


def normalize_rms(wav, target_rms=0.1, eps=1e-6):
    """Scale wav so its RMS loudness equals target_rms -- mirrors
    sound_mixer.py's normalize_rms so the clean target sits on the same
    loudness footing the dataset team used before mixing."""
    current = wav.pow(2).mean().clamp_min(eps).sqrt()
    if current < eps:
        return wav
    return wav * (target_rms / current)


def fit_length(wav, length):
    """Deterministically tile/truncate wav to exactly `length` samples
    (same convention as sound_mixer.py's fit_length) -- used only when a row
    lists more than one clean file and they need to be summed together."""
    if wav.shape[-1] == 0:
        return torch.zeros(length, dtype=torch.float32)
    if wav.shape[-1] < length:
        reps = length // wav.shape[-1] + 1
        wav = wav.repeat(reps)
    return wav[..., :length]


def random_crop_pair_or_loop(mixed, clean, length, rng):
    """Crop (or loop, if too short) a `length`-sample segment from a
    time-aligned (mixed, clean) pair, using the SAME start position for both
    so they stay in sync."""
    assert mixed.shape[-1] == clean.shape[-1], "mixed/clean length mismatch"
    if mixed.shape[-1] < length:
        reps = length // mixed.shape[-1] + 1
        mixed = mixed.repeat(reps)
        clean = clean.repeat(reps)
    start = int(rng.integers(0, mixed.shape[-1] - length + 1))
    return mixed[start:start + length], clean[start:start + length]


# --------------------------------------------------------------------------
# 2. mix_log.csv loading / splitting
# --------------------------------------------------------------------------
def load_csv_rows(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    # sort by the numeric part of the filename (1.wav, 2.wav, ..., 200.wav)
    # so the index ranges below line up with "file N" regardless of row order
    rows.sort(key=lambda r: int(Path(r["filename"]).stem))
    return rows


def split_rows(rows, train_end, val_end):
    """Files 1..train_end -> train, train_end+1..val_end -> validation,
    everything after val_end -> held-out extreme/robustness set."""
    def idx(r):
        return int(Path(r["filename"]).stem)
    train_rows = [r for r in rows if idx(r) <= train_end]
    val_rows = [r for r in rows if train_end < idx(r) <= val_end]
    extreme_rows = [r for r in rows if idx(r) > val_end]
    return train_rows, val_rows, extreme_rows


# --------------------------------------------------------------------------
# 3. Dataset: loads pre-mixed pairs, no on-the-fly mixing
# --------------------------------------------------------------------------
class MixedPairDataset(Dataset):
    """One dataset row = one (mixed_dataset/N.wav, clean reference) pair,
    as described by a row of mix_log.csv. A random segment_seconds-long crop
    is taken fresh each __getitem__ call (same position for both signals),
    which still gives useful variety across epochs even though the
    underlying file list is fixed and small."""

    def __init__(self, data_root, csv_rows, sample_rate=16000,
                 segment_seconds=4.0, clean_target_rms=0.1, seed=0):
        self.mixed_dir = Path(data_root) / "mixed_dataset"
        self.clean_dir = Path(data_root) / "clean"
        self.rows = csv_rows
        self.sample_rate = sample_rate
        self.segment_len = int(segment_seconds * sample_rate)
        self.clean_target_rms = clean_target_rms
        self.rng = np.random.default_rng(seed)
        # cache decoded audio in memory -- the same clean file (e.g.
        # "Clean clip1.wav") is reused across many mixed_dataset rows
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def _load_cached(self, path):
        path = str(path)
        if path not in self._cache:
            self._cache[path] = load_audio(path, self.sample_rate)
        return self._cache[path]

    def _load_clean_target(self, row):
        names = [n for n in row["clean_files"].split(";") if n]
        sigs = [normalize_rms(self._load_cached(self.clean_dir / n), self.clean_target_rms)
                for n in names]
        target_len = max(s.shape[-1] for s in sigs)
        sigs = [fit_length(s, target_len) for s in sigs]
        return torch.stack(sigs).sum(dim=0)

    def __getitem__(self, idx):
        row = self.rows[idx]
        mixed = self._load_cached(self.mixed_dir / row["filename"])
        clean = self._load_clean_target(row)

        # sound_mixer.py builds the mix and the clean reference to the same
        # length, but guard against off-by-one / resampling rounding anyway
        length = min(mixed.shape[-1], clean.shape[-1])
        mixed, clean = mixed[..., :length], clean[..., :length]

        return random_crop_pair_or_loop(mixed, clean, self.segment_len, self.rng)


# --------------------------------------------------------------------------
# 4. STFT / iSTFT front-end matching what GTCRN expects: (B, F, T, 2)
# --------------------------------------------------------------------------
class STFTFrontEnd:
    def __init__(self, n_fft=512, hop_length=256, win_length=512, device="cpu"):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length, device=device)

    def to(self, device):
        self.window = self.window.to(device)
        return self

    def stft(self, wav):
        """wav: (B, samples) -> spec: (B, F, T, 2)"""
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length, window=self.window,
                           center=True, return_complex=True)
        return torch.view_as_real(spec)

    def istft(self, spec, length=None):
        """spec: (B, F, T, 2) -> wav: (B, samples)"""
        spec_c = torch.view_as_complex(spec.contiguous())
        return torch.istft(spec_c, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window,
                            center=True, length=length)


# --------------------------------------------------------------------------
# 5. Loss functions
# --------------------------------------------------------------------------
def compressed_spectral_loss(pred_spec, clean_spec, power=0.3):
    """L1 loss on power-law-compressed magnitude + compressed complex value.
    Compressing the magnitude (mag**0.3) puts more weight on quiet regions
    (like speech gaps and low-energy phonemes) than a plain L1 on the raw
    spectrogram would -- standard trick in low-complexity SE training."""
    pred_c = torch.view_as_complex(pred_spec.contiguous())
    clean_c = torch.view_as_complex(clean_spec.contiguous())

    pred_mag = pred_c.abs().clamp_min(1e-12)
    clean_mag = clean_c.abs().clamp_min(1e-12)
    pred_mag_comp = pred_mag.pow(power)
    clean_mag_comp = clean_mag.pow(power)

    pred_comp = pred_c / pred_mag * pred_mag_comp
    clean_comp = clean_c / clean_mag * clean_mag_comp

    mag_loss = F.l1_loss(pred_mag_comp, clean_mag_comp)
    complex_loss = F.l1_loss(torch.view_as_real(pred_comp), torch.view_as_real(clean_comp))
    return 0.5 * mag_loss + 0.5 * complex_loss


def si_snr_loss(pred_wav, clean_wav, eps=1e-8):
    """Negative scale-invariant SNR, averaged over the batch. This is a
    time-domain loss so it complements the frequency-domain loss above and
    correlates well with perceived enhancement quality.

    NOTE ON A KNOWN FAILURE MODE: because this loss is scale-invariant (it
    projects pred onto clean before scoring), a model that outputs a
    globally-scaled-down version of the clean speech (e.g. 0.6x) scores
    almost as well as one that outputs the correct amplitude -- the metric
    literally cannot see uniform under-suppression. Combined with a strong
    penalty on leftover noise, this gives the network a "free" incentive to
    hedge by turning everything down a bit. wav_l1_loss and
    suppression_penalty below are both NOT scale-invariant and exist
    specifically to counteract this."""
    pred = pred_wav - pred_wav.mean(dim=-1, keepdim=True)
    clean = clean_wav - clean_wav.mean(dim=-1, keepdim=True)
    proj = (torch.sum(pred * clean, dim=-1, keepdim=True) /
            (torch.sum(clean ** 2, dim=-1, keepdim=True) + eps)) * clean
    noise = pred - proj
    si_snr = 10 * torch.log10(
        (torch.sum(proj ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    )
    return -si_snr.mean()


def wav_l1_loss(pred_wav, clean_wav):
    """Plain (NOT scale-invariant) L1 loss on the waveform. Unlike si_snr_loss,
    this directly penalizes a globally-scaled-down prediction, closing the
    "free under-suppression" loophole described above. Keep the weight on
    this small (~0.1-0.2) -- it's a corrective nudge, not the main driver."""
    return F.l1_loss(pred_wav, clean_wav)


def suppression_penalty(pred_spec, clean_spec, under_weight=2.5, over_weight=1.0):
    """Asymmetric penalty on the STFT magnitude: costs `under_weight` per
    unit of speech energy the model REMOVED that it shouldn't have
    (clean_mag > pred_mag, i.e. over-suppression), and `over_weight` per
    unit of noise energy it LEFT IN (pred_mag > clean_mag, i.e.
    under-suppression). Raising under_weight relative to over_weight
    directly counteracts a model that's learned to play it safe by
    suppressing speech along with the noise -- exactly the symptom of
    "speech disappears whenever there's gunfire.\""""
    pred_mag = torch.view_as_complex(pred_spec.contiguous()).abs()
    clean_mag = torch.view_as_complex(clean_spec.contiguous()).abs()
    diff = clean_mag - pred_mag
    under = torch.clamp(diff, min=0)   # model suppressed too much
    over = torch.clamp(-diff, min=0)   # model left too much noise in
    return (under_weight * under + over_weight * over).mean()


# --------------------------------------------------------------------------
# 6. Train / validate for one epoch
# --------------------------------------------------------------------------
def run_epoch(model, loader, stft, optimizer, device, train=True,
              spec_weight=1.0, wav_weight=1.0,
              wav_l1_weight=0.15,
              supp_weight=0.4, supp_under_weight=2.5, supp_over_weight=1.0):
    model.train(mode=train)
    totals = {"total": 0.0, "spec": 0.0, "si_snr": 0.0, "wav_l1": 0.0, "supp": 0.0}
    n_batches = 0

    for noisy_wav, clean_wav in loader:
        noisy_wav = noisy_wav.to(device)
        clean_wav = clean_wav.to(device)

        noisy_spec = stft.stft(noisy_wav)
        clean_spec = stft.stft(clean_wav)

        with torch.set_grad_enabled(train):
            pred_spec = model(noisy_spec)
            pred_wav = stft.istft(pred_spec, length=noisy_wav.shape[-1])

            loss_spec = compressed_spectral_loss(pred_spec, clean_spec)
            loss_si_snr = si_snr_loss(pred_wav, clean_wav)
            loss_wav_l1 = wav_l1_loss(pred_wav, clean_wav)
            loss_supp = suppression_penalty(pred_spec, clean_spec,
                                             under_weight=supp_under_weight,
                                             over_weight=supp_over_weight)

            loss = (spec_weight * loss_spec
                    + wav_weight * loss_si_snr
                    + wav_l1_weight * loss_wav_l1
                    + supp_weight * loss_supp)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        totals["total"] += loss.item()
        totals["spec"] += loss_spec.item()
        totals["si_snr"] += loss_si_snr.item()
        totals["wav_l1"] += loss_wav_l1.item()
        totals["supp"] += loss_supp.item()
        n_batches += 1

    n_batches = max(n_batches, 1)
    return {k: v / n_batches for k, v in totals.items()}


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="sample_data",
                   help="path to the dataset folder (contains mixed_dataset/, clean/, csvs/) "
                        "-- defaults to 'sample_data', matching the project layout")
    p.add_argument("--csv_name", type=str, default="csvs/mix_log.csv",
                   help="path to the mix log CSV, relative to --data_root")
    p.add_argument("--train_end", type=int, default=100,
                   help="files 1..train_end (by numeric filename) are used for training")
    p.add_argument("--val_end", type=int, default=150,
                   help="files train_end+1..val_end are used for validation; "
                        "everything after val_end becomes the held-out extreme set")
    p.add_argument("--clean_target_rms", type=float, default=0.1,
                   help="RMS loudness the clean target is normalized to -- match "
                        "sound_mixer.py's TARGET_INPUT_RMS")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--segment_seconds", type=float, default=4.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--spec_weight", type=float, default=1.0)
    p.add_argument("--wav_weight", type=float, default=1.0,
                   help="weight on the scale-invariant SI-SNR loss")
    p.add_argument("--wav_l1_weight", type=float, default=0.15,
                   help="EARLY-PHASE (epochs 1..schedule_epoch-1) weight on plain "
                        "(non-scale-invariant) waveform L1 loss -- counteracts SI-SNR's "
                        "blind spot for uniform under-suppression")
    p.add_argument("--supp_weight", type=float, default=0.4,
                   help="EARLY-PHASE weight on the asymmetric over-suppression penalty")
    p.add_argument("--supp_under_weight", type=float, default=2.5,
                   help="EARLY-PHASE cost per unit of speech energy removed that shouldn't "
                        "have been (raise this if the model still over-suppresses speech)")
    p.add_argument("--supp_over_weight", type=float, default=1.0,
                   help="EARLY-PHASE cost per unit of leftover noise energy (raise this if "
                        "the model starts letting too much noise/gunfire through)")
    p.add_argument("--wav_l1_weight_late", type=float, default=None,
                   help="LATE-PHASE (epochs >= schedule_epoch) value for --wav_l1_weight. "
                        "Defaults to the same value as --wav_l1_weight (no schedule).")
    p.add_argument("--supp_weight_late", type=float, default=None,
                   help="LATE-PHASE value for --supp_weight. Defaults to --supp_weight "
                        "(no schedule) if not given.")
    p.add_argument("--supp_under_weight_late", type=float, default=None,
                   help="LATE-PHASE value for --supp_under_weight. Defaults to "
                        "--supp_under_weight (no schedule) if not given.")
    p.add_argument("--supp_over_weight_late", type=float, default=None,
                   help="LATE-PHASE value for --supp_over_weight. Defaults to "
                        "--supp_over_weight (no schedule) if not given.")
    p.add_argument("--schedule_epoch", type=int, default=None,
                   help="epoch (1-indexed) at which loss weights switch from the "
                        "early-phase values to the *_late values. E.g. 15 means epochs "
                        "1-14 use the early values, 15+ use the late values. Leave unset "
                        "to use the early values for the whole run (no schedule).")
    p.add_argument("--tra_bands", type=int, default=4,
                   help="number of frequency bands each GTConvBlock's attention gate is "
                        "computed over independently. 1 = original TRA (one gate shared "
                        "across the whole spectrum -- gunfire bursts pull the gain down "
                        "for speech-only frequency bands in the same frame). Costs no "
                        "extra parameters at any value since the GRU is shared across bands.")
    p.add_argument("--base_channels", type=int, default=24,
                   help="width of every conv/GTConvBlock/DPGRNN layer. Paper value is 16 "
                        "(~24k trainable params); must be a multiple of 4. Main lever for "
                        "spectral representation capacity -- roughly quadratic in param cost.")
    p.add_argument("--n_dpgrnn", type=int, default=3,
                   help="number of stacked DPGRNN (dual-path grouped RNN) stages at the "
                        "bottleneck. Paper value is 2. Extra stages add cheap temporal/"
                        "cross-frequency modeling depth at the network's most downsampled "
                        "resolution.")
    p.add_argument("--mask_floor_db", type=float, default=None,
                   help="optional mask magnitude floor in dB, applied during training too "
                        "if set (usually leave this None during training and only use it "
                        "as an inference-time knob in infer.py instead)")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init_checkpoint", type=str, default=None,
                   help="path to a .pt checkpoint (e.g. checkpoints/best_model.pt) to "
                        "resume training FROM instead of random init. Must match the "
                        "current --base_channels/--n_dpgrnn/--tra_bands. Optimizer/"
                        "scheduler and best_val always start fresh -- only the model "
                        "weights are carried over.")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.data_root) / args.csv_name
    rows = load_csv_rows(csv_path)
    train_rows, val_rows, extreme_rows = split_rows(rows, args.train_end, args.val_end)
    print(f"Split: {len(train_rows)} train / {len(val_rows)} validation / "
          f"{len(extreme_rows)} held-out extreme (robustness) files")

    train_ds = MixedPairDataset(args.data_root, train_rows, args.sample_rate,
                                 args.segment_seconds, args.clean_target_rms, seed=args.seed)
    val_ds = MixedPairDataset(args.data_root, val_rows, args.sample_rate,
                               args.segment_seconds, args.clean_target_rms, seed=args.seed + 1000)
    extreme_ds = MixedPairDataset(args.data_root, extreme_rows, args.sample_rate,
                                   args.segment_seconds, args.clean_target_rms, seed=args.seed + 2000)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    extreme_loader = DataLoader(extreme_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = GTCRN(sample_rate=args.sample_rate, n_fft=args.n_fft,
                   tra_bands=args.tra_bands, mask_floor_db=args.mask_floor_db,
                   base_channels=args.base_channels, n_dpgrnn=args.n_dpgrnn).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    n_trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"GTCRN params: {n_params:,} total / {n_trainable:,} trainable "
          f"(base_channels={args.base_channels}, n_dpgrnn={args.n_dpgrnn}, tra_bands={args.tra_bands})")

    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device)
        init_cfg = init_ckpt.get("config", {})
        for key in ("base_channels", "n_dpgrnn", "tra_bands"):
            if key in init_cfg and init_cfg[key] != getattr(args, key):
                raise ValueError(
                    f"--{key}={getattr(args, key)} doesn't match the checkpoint's "
                    f"{key}={init_cfg[key]}. Pass matching architecture flags, or "
                    f"omit --init_checkpoint to train that architecture from scratch."
                )
        model.load_state_dict(init_ckpt["model_state_dict"])
        print(f"Resumed weights from {args.init_checkpoint} "
              f"(was epoch {init_ckpt.get('epoch')}, val_loss {init_ckpt.get('val_loss')}). "
              f"Optimizer/scheduler/best_val start fresh from here.")

    stft = STFTFrontEnd(args.n_fft, args.hop_length, args.n_fft, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # Early-phase loss weights: what every epoch uses until (and if) schedule_epoch hits.
    early_kwargs = dict(spec_weight=args.spec_weight, wav_weight=args.wav_weight,
                         wav_l1_weight=args.wav_l1_weight, supp_weight=args.supp_weight,
                         supp_under_weight=args.supp_under_weight, supp_over_weight=args.supp_over_weight)

    # Late-phase loss weights: any *_late arg left unset falls back to its early value,
    # so passing only some of the _late flags is fine.
    late_kwargs = dict(
        spec_weight=args.spec_weight,      # not scheduled -- same both phases
        wav_weight=args.wav_weight,        # not scheduled -- same both phases
        wav_l1_weight=args.wav_l1_weight_late if args.wav_l1_weight_late is not None else args.wav_l1_weight,
        supp_weight=args.supp_weight_late if args.supp_weight_late is not None else args.supp_weight,
        supp_under_weight=args.supp_under_weight_late if args.supp_under_weight_late is not None else args.supp_under_weight,
        supp_over_weight=args.supp_over_weight_late if args.supp_over_weight_late is not None else args.supp_over_weight,
    )

    if args.schedule_epoch is not None:
        if not (1 <= args.schedule_epoch <= args.epochs):
            raise ValueError(f"--schedule_epoch={args.schedule_epoch} must be between 1 and --epochs={args.epochs}")
        print(f"Loss schedule: epochs 1-{args.schedule_epoch - 1} use early weights "
              f"{ {k: v for k, v in early_kwargs.items() if v != late_kwargs[k]} }, "
              f"epochs {args.schedule_epoch}-{args.epochs} switch to "
              f"{ {k: v for k, v in late_kwargs.items() if v != early_kwargs[k]} }")
    else:
        print("No --schedule_epoch given: using the early-phase loss weights for the entire run.")

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        use_late = args.schedule_epoch is not None and epoch >= args.schedule_epoch
        loss_kwargs = late_kwargs if use_late else early_kwargs
        if use_late and epoch == args.schedule_epoch:
            print(f"  -> epoch {epoch}: switching to late-phase loss weights {late_kwargs}")

        train_stats = run_epoch(model, train_loader, stft, optimizer, device, train=True, **loss_kwargs)
        val_stats = run_epoch(model, val_loader, stft, optimizer, device, train=False, **loss_kwargs)
        scheduler.step(val_stats["total"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:03d}/{args.epochs} | train_loss {train_stats['total']:.4f} "
              f"| val_loss {val_stats['total']:.4f} | lr {current_lr:.2e} | "
              f"val breakdown: spec {val_stats['spec']:.4f} si_snr {val_stats['si_snr']:.4f} "
              f"wav_l1 {val_stats['wav_l1']:.4f} supp {val_stats['supp']:.4f}")
        history.append((epoch, train_stats["total"], val_stats["total"], current_lr,
                         val_stats["spec"], val_stats["si_snr"], val_stats["wav_l1"], val_stats["supp"]))

        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "config": {"sample_rate": args.sample_rate, "n_fft": args.n_fft,
                           "hop_length": args.hop_length, "tra_bands": args.tra_bands,
                           "mask_floor_db": args.mask_floor_db,
                           "base_channels": args.base_channels, "n_dpgrnn": args.n_dpgrnn,
                           "dilations": list(model.dilations)},
            }, out_dir / "best_model.pt")
            print(f"  -> saved new best checkpoint (val_loss={best_val:.4f})")

    torch.save({"model_state_dict": model.state_dict(),
                "epoch": args.epochs, "val_loss": val_stats["total"],
                "config": {"sample_rate": args.sample_rate, "n_fft": args.n_fft,
                           "hop_length": args.hop_length, "tra_bands": args.tra_bands,
                           "mask_floor_db": args.mask_floor_db,
                           "base_channels": args.base_channels, "n_dpgrnn": args.n_dpgrnn,
                           "dilations": list(model.dilations)}},
               out_dir / "last_model.pt")

    with open(out_dir / "history.csv", "w") as f:
        f.write("epoch,train_loss,val_loss,lr,val_spec,val_si_snr,val_wav_l1,val_supp\n")
        for row in history:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nDone. Best val_loss={best_val:.4f}. Checkpoints saved in {out_dir}/")

    # One-time robustness check on the held-out extreme (-5dB floor) set,
    # using the BEST checkpoint -- never used for training or model selection,
    # just a number for your report on how the model holds up under the
    # harshest noise conditions in the dataset.
    if len(extreme_ds) > 0:
        best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        final_kwargs = late_kwargs if args.schedule_epoch is not None else early_kwargs
        extreme_stats = run_epoch(model, extreme_loader, stft, optimizer, device, train=False, **final_kwargs)
        print(f"Extreme (robustness, -5dB floor) set loss, best checkpoint: {extreme_stats['total']:.4f}")
        best_ckpt["extreme_loss"] = extreme_stats["total"]
        torch.save(best_ckpt, out_dir / "best_model.pt")
    else:
        print("No files fell after --val_end, so no extreme/robustness set was evaluated.")


if __name__ == "__main__":
    main()