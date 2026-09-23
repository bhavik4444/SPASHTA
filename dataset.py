"""
Dynamic mixing dataset for the ANC system.

WHY THIS REPLACES PRE-MIXED FILES
----------------------------------
The previous pipeline wrote mixtures to disk with sound_mixer.py and then had
train.py guess the clean target by RMS-normalising the original clean file.
That guess was wrong by a random per-file factor, because the mixer scaled
clean by a random weight and then peak-normalised the sum by a factor it never
logged. So the "ground truth" amplitude the network was asked to reproduce was
off by several dB in an unpredictable direction, and the error was largest
exactly where noise was loudest — at low SNR. Every amplitude-sensitive loss
(waveform L1, the gain-matching term, the asymmetric suppression penalty) was
being driven by that corrupted target. A network trained that way cannot do
better than output some hedged average gain, which is what "speech becomes
unintelligible at low SNR" sounds like.

Mixing here happens in the dataloader, so the clean target is the exact signal
that was summed into the mixture — same scale, same samples, bit-for-bit. Every
loss term below becomes meaningful, and the effective dataset size stops being
the number of files on disk: a fresh SNR, a fresh crop, a fresh noise
placement and fresh burst positions are drawn every time an item is served.

WHAT EACH ITEM LOOKS LIKE
-------------------------
    mixture:  float32 [segment], RMS normalised to `mix_rms`
    clean:    float32 [segment], the exact speech component of that mixture
    snr:      float32 scalar, the SNR actually used (sentinel 40.0 for
              speech-only items, -40.0 for noise-only items)

The mixture is normalised to a fixed RMS and the clean target is scaled by the
SAME factor, so the pair's relationship is preserved exactly while every item
arrives at a comparable loudness. That last part matters for training
stability: without it a mixture whose peak was dominated by one gunshot ends up
with speech 40 dB below full scale, and its contribution to the loss all but
vanishes compared to an easy item.

FILE LAYOUT
-----------
    <data_root>/clean/   speech recordings (any length, any sample rate)
    <data_root>/noise/   impulsive / non-stationary defence noise (gunfire,
                         artillery, blasts, vehicle passes)
    <data_root>/bg/      stationary background (wind, static, hum, engine drone)

Subdirectories are scanned recursively, so you can organise these however you
like. Files are split into train/val by a hash of their path, so a given
recording is only ever in one of the two — validation then measures
generalisation to unseen speakers and unseen noise, not memorisation.

A 16 kHz mono cache (.npy, memory-mapped) is built once on first run. After
that, serving an item costs a couple of slice reads, so you can point this at
an arbitrarily large corpus without worrying about RAM.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".aif")


# ===========================================================================
# File discovery, caching, splitting
# ===========================================================================
def list_audio_files(folder, recursive=True):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(str(p) for p in it
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _cache_name(path):
    return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:20] + ".npy"


def build_cache(files, cache_dir, target_sr, rebuild=False, verbose=True):
    """Decode, downmix to mono and resample every file to `target_sr` once,
    storing float32 .npy alongside a manifest. Returns the list of cache paths
    that decoded successfully (files that fail are reported and dropped)."""
    import torchaudio

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists() and not rebuild:
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}

    out, made, failed = [], 0, []
    for src in files:
        dst = cache_dir / _cache_name(src)
        key = str(Path(src).resolve())
        if dst.exists() and not rebuild and key in manifest:
            out.append(str(dst))
            continue
        try:
            data, sr = sf.read(src, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if data.size == 0:
                raise ValueError("empty file")
            if sr != target_sr:
                t = torch.from_numpy(np.ascontiguousarray(data))
                t = torchaudio.functional.resample(t, sr, target_sr)
                data = t.numpy()
            data = np.ascontiguousarray(data, dtype=np.float32)
            peak = float(np.max(np.abs(data)))
            if not np.isfinite(peak) or peak < 1e-8:
                raise ValueError("silent file")
            np.save(dst, data)
            manifest[key] = {"cache": dst.name, "frames": int(data.shape[0])}
            out.append(str(dst))
            made += 1
        except Exception as exc:
            failed.append((src, str(exc)))

    manifest_path.write_text(json.dumps(manifest))
    if verbose:
        msg = f"cache: {len(out)} usable file(s)"
        if made:
            msg += f", {made} newly decoded"
        if failed:
            msg += f", {len(failed)} skipped"
        print("  " + msg)
        for src, why in failed[:5]:
            print(f"    skipped {Path(src).name}: {why}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")
    return out


def load_name_map(cache_dir):
    """{cache_path -> original filename}. Cache files are named by a hash of
    the source path, so anything that wants to report which recording it used
    (the mixer's CSV, for instance) needs this to get back to a human name."""
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return {}
    return {str(Path(cache_dir) / v["cache"]): Path(k).name
            for k, v in manifest.items() if isinstance(v, dict) and "cache" in v}


def hash_split(files, val_fraction=0.1, salt=""):
    """Deterministic split by path hash. Stable when files are added: an
    existing file never moves between splits just because the corpus grew."""
    train, val = [], []
    if not files:
        return train, val
    for f in files:
        h = hashlib.md5((salt + Path(f).name).encode()).hexdigest()
        if (int(h[:8], 16) % 10000) / 10000.0 < val_fraction:
            val.append(f)
        else:
            train.append(f)
    if not train:                 # tiny corpora: never leave training empty
        train, val = files, files[:1]
    if not val:
        val = train[:1]
    return train, val


# ===========================================================================
# Signal helpers
# ===========================================================================
def rms(x, eps=1e-12):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps))


def active_rms(x, sr, frame_ms=20.0, rel_db=-35.0, abs_floor=1e-5):
    """RMS over the speech-active portion only, roughly in the spirit of
    ITU-T P.56. Using plain whole-signal RMS to set SNR is a quiet but real
    bug: a clip that is 50% pauses reports a level ~3 dB below its actual
    speech level, so the mixture ends up several dB easier than its label
    claims, and the model is systematically under-trained at the SNRs you
    think you are training on."""
    n = max(1, int(sr * frame_ms / 1000.0))
    if x.shape[0] < n:
        return rms(x)
    usable = (x.shape[0] // n) * n
    frames = x[:usable].reshape(-1, n).astype(np.float64)
    fe = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    peak = fe.max()
    if peak < abs_floor:
        return rms(x)
    thr = max(peak * (10.0 ** (rel_db / 20.0)), abs_floor)
    sel = fe >= thr
    if not sel.any():
        return rms(x)
    return float(np.sqrt((frames[sel] ** 2).mean() + 1e-12))


def read_segment(cache_path, n_samples, rng, allow_tile=True):
    """Random contiguous crop from a cached 16 kHz mono .npy, memory-mapped."""
    arr = np.load(cache_path, mmap_mode="r")
    n = arr.shape[0]
    if n == 0:
        return np.zeros(n_samples, dtype=np.float32)
    if n >= n_samples:
        start = int(rng.integers(0, n - n_samples + 1))
        return np.array(arr[start:start + n_samples], dtype=np.float32)
    if not allow_tile:
        out = np.zeros(n_samples, dtype=np.float32)
        out[:n] = arr
        return out
    reps = int(math.ceil(n_samples / n)) + 1
    buf = np.tile(np.array(arr, dtype=np.float32), reps)
    start = int(rng.integers(0, max(1, buf.shape[0] - n_samples)))
    return buf[start:start + n_samples].astype(np.float32)


def tilt_filter(x, a):
    """One-pole spectral tilt: y[n] = x[n] - a*x[n-1]. a>0 brightens, a<0
    darkens. Cheap stand-in for microphone/channel colouration."""
    if abs(a) < 1e-4:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - a * x[:-1]
    return y


def place_bursts(source, out_len, rng, n_bursts, min_ms=120, max_ms=1200,
                 sr=16000, gain_jitter_db=6.0):
    """Scatter short excerpts of `source` at random positions in a silent
    buffer, with short fades and per-burst gain jitter.

    Real gunfire is sparse and peaky. Tiling a gunshot recording end to end —
    which is what the old mixer did — turns it into a continuous texture that
    behaves statistically like stationary noise, so a model trained on it never
    learns the thing it actually has to handle: a 200 ms event 20 dB above
    speech, followed by silence. Sparse placement also puts the hard decision
    (is this frame speech or blast?) at a different point in every sample."""
    out = np.zeros(out_len, dtype=np.float32)
    if source.shape[0] < 32 or n_bursts <= 0:
        return out
    for _ in range(n_bursts):
        dur = int(rng.integers(int(min_ms * sr / 1000), int(max_ms * sr / 1000) + 1))
        dur = min(dur, out_len, source.shape[0])
        if dur < 32:
            continue
        s_src = int(rng.integers(0, source.shape[0] - dur + 1))
        s_dst = int(rng.integers(0, out_len - dur + 1))
        seg = source[s_src:s_src + dur].copy()

        fade = min(64, dur // 8)
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]

        g = 10.0 ** (float(rng.uniform(-gain_jitter_db, gain_jitter_db)) / 20.0)
        out[s_dst:s_dst + dur] += seg * g
    return out


# ===========================================================================
# Dynamic mixing
# ===========================================================================
class DynamicMixDataset(Dataset):
    """Mixes a fresh (mixture, clean) pair on every __getitem__.

    Set deterministic=True for validation: the RNG is seeded from the item
    index, so the same index always yields the same mixture and validation
    loss is comparable across epochs and across runs.
    """

    def __init__(self, clean_files, noise_files, bg_files,
                 sample_rate=16000, segment_seconds=4.0, length=8000,
                 snr_min=-15.0, snr_max=20.0, hard_frac=0.45, hard_span=10.0,
                 p_speech_only=0.06, p_noise_only=0.06,
                 n_noise_range=(1, 2), n_bg_range=(0, 2),
                 p_burst=0.55, burst_range=(1, 5),
                 bg_rel_db=(-15.0, 0.0),
                 speech_level_db=(-28.0, -16.0), mix_rms=0.1,
                 p_speech_tilt=0.3, p_noise_tilt=0.5, p_clip=0.04,
                 deterministic=False, seed=1234, return_meta=False):
        self.clean_files = list(clean_files)
        self.noise_files = list(noise_files)
        self.bg_files = list(bg_files)
        if not self.clean_files:
            raise ValueError("no clean speech files found — check <data_root>/clean/")

        self.sr = sample_rate
        self.seg = int(round(segment_seconds * sample_rate))
        self.length = int(length)
        self.snr_min, self.snr_max = snr_min, snr_max
        self.hard_frac, self.hard_span = hard_frac, hard_span
        self.p_speech_only = p_speech_only
        self.p_noise_only = p_noise_only if (self.noise_files or self.bg_files) else 0.0
        self.n_noise_range = n_noise_range
        self.n_bg_range = n_bg_range
        self.p_burst = p_burst
        self.burst_range = burst_range
        self.bg_rel_db = bg_rel_db
        self.speech_level_db = speech_level_db
        self.mix_rms = mix_rms
        self.p_speech_tilt = p_speech_tilt
        self.p_noise_tilt = p_noise_tilt
        self.p_clip = p_clip
        self.deterministic = deterministic
        self.seed = seed
        # sound_mixer.py sets this so it can log exactly which source files went
        # into each written mixture. Keeping the mixer on this same code path is
        # deliberate: the previous pipeline had two independent implementations
        # of "what is the clean component of this mixture", and they disagreed.
        self.return_meta = return_meta

    def __len__(self):
        return self.length

    # ---------------------------------------------------------------- utils
    def _rng(self, idx):
        if self.deterministic:
            return np.random.default_rng([self.seed, idx])
        return np.random.default_rng()

    def _draw_snr(self, rng):
        """Uniform over the full range, but with `hard_frac` of the mass
        concentrated in the bottom `hard_span` dB. Uniform sampling alone
        spends most of its budget on SNRs the model already handles; the
        failures are all below 0 dB, so that is where the samples should go."""
        if rng.random() < self.hard_frac:
            return float(rng.uniform(self.snr_min, self.snr_min + self.hard_span))
        return float(rng.uniform(self.snr_min, self.snr_max))

    def _get_speech(self, rng):
        """Crop speech, retrying a few times to avoid landing on a pure pause.
        Returns (segment, source_path)."""
        best, best_lvl, best_path = None, -1.0, ""
        for _ in range(6):
            path = self.clean_files[int(rng.integers(0, len(self.clean_files)))]
            seg = read_segment(path, self.seg, rng)
            lvl = active_rms(seg, self.sr)
            if lvl > best_lvl:
                best, best_lvl, best_path = seg, lvl, path
            if lvl > 1e-3:
                break
        if best is None:
            return np.zeros(self.seg, dtype=np.float32), ""
        return best, best_path

    def _build_noise(self, rng):
        """Sum impulsive noise (sparse bursts or continuous) and stationary
        background into one interference signal at unit RMS.
        Returns (signal_or_None, provenance_dict)."""
        buf = np.zeros(self.seg, dtype=np.float32)
        used = False
        info = {"noise_files": [], "bg_files": [], "n_bursts": 0}

        if self.noise_files:
            lo, hi = self.n_noise_range
            k = int(rng.integers(lo, hi + 1))
            for _ in range(k):
                path = self.noise_files[int(rng.integers(0, len(self.noise_files)))]
                if rng.random() < self.p_burst:
                    src = read_segment(path, self.seg * 2, rng, allow_tile=True)
                    nb = int(rng.integers(self.burst_range[0], self.burst_range[1] + 1))
                    sig = place_bursts(src, self.seg, rng, nb, sr=self.sr)
                    info["n_bursts"] += nb
                else:
                    sig = read_segment(path, self.seg, rng)
                r = rms(sig)
                if r < 1e-8:
                    continue
                if rng.random() < self.p_noise_tilt:
                    sig = tilt_filter(sig, float(rng.uniform(-0.4, 0.4)))
                    r = max(rms(sig), 1e-8)
                buf += (sig / r).astype(np.float32)
                info["noise_files"].append(path)
                used = True

        if self.bg_files:
            lo, hi = self.n_bg_range
            k = int(rng.integers(lo, hi + 1))
            for _ in range(k):
                path = self.bg_files[int(rng.integers(0, len(self.bg_files)))]
                sig = read_segment(path, self.seg, rng)
                r = rms(sig)
                if r < 1e-8:
                    continue
                if rng.random() < self.p_noise_tilt:
                    sig = tilt_filter(sig, float(rng.uniform(-0.4, 0.4)))
                    r = max(rms(sig), 1e-8)
                g = 10.0 ** (float(rng.uniform(*self.bg_rel_db)) / 20.0)
                buf += (sig / r * g).astype(np.float32)
                info["bg_files"].append(path)
                used = True

        if not used:
            return None, info
        r = rms(buf)
        if r < 1e-8:
            return None, info
        return (buf / r).astype(np.float32), info

    # ------------------------------------------------------------ main path
    def __getitem__(self, idx):
        rng = self._rng(idx)
        mode_roll = rng.random()
        meta = {"kind": "mixed", "clean_file": "", "noise_files": [],
                "bg_files": [], "n_bursts": 0, "clipped": 0}

        # ---- noise-only item: target is exact silence ----
        if mode_roll < self.p_noise_only:
            noise, info = self._build_noise(rng)
            if noise is not None:
                meta.update(info)
                meta["kind"] = "noise_only"
                clean = np.zeros(self.seg, dtype=np.float32)
                mix = noise * 10.0 ** (float(rng.uniform(-26.0, -12.0)) / 20.0)
                return self._finalise(mix, clean, -40.0, meta)

        # ---- speech ----
        clean, clean_path = self._get_speech(rng)
        meta["clean_file"] = clean_path
        if rng.random() < self.p_speech_tilt:
            clean = tilt_filter(clean, float(rng.uniform(-0.3, 0.3)))
        lvl = active_rms(clean, self.sr)
        if lvl < 1e-6:
            clean = np.zeros(self.seg, dtype=np.float32)
        else:
            target_lvl = 10.0 ** (float(rng.uniform(*self.speech_level_db)) / 20.0)
            clean = (clean * (target_lvl / lvl)).astype(np.float32)
            lvl = target_lvl

        # ---- speech-only item: nothing to remove, everything to preserve ----
        if mode_roll < self.p_noise_only + self.p_speech_only or lvl < 1e-6:
            meta["kind"] = "speech_only"
            return self._finalise(clean.copy(), clean, 40.0, meta)

        noise, info = self._build_noise(rng)
        meta.update(info)
        if noise is None:
            meta["kind"] = "speech_only"
            return self._finalise(clean.copy(), clean, 40.0, meta)

        snr = self._draw_snr(rng)
        noise = noise * (lvl / (10.0 ** (snr / 20.0)))
        mix = (clean + noise).astype(np.float32)

        # occasional recording-chain clipping, applied to the mixture only:
        # the target stays the undistorted speech, which is what we want the
        # model to recover
        if rng.random() < self.p_clip:
            peak = np.max(np.abs(mix)) + 1e-9
            thr = peak * float(rng.uniform(0.35, 0.8))
            mix = np.clip(mix, -thr, thr).astype(np.float32)
            meta["clipped"] = 1

        return self._finalise(mix, clean, snr, meta)

    def _finalise(self, mix, clean, snr, meta=None):
        """Scale mixture and target by the SAME factor so every item arrives at
        a comparable loudness without disturbing their relationship.

        This joint scaling is the reason the target stays exact. Anything that
        touches the mixture after the sum must touch the target identically, or
        the pair stops being a valid (input, ground truth) example."""
        r = rms(mix)
        if r > 1e-8:
            s = self.mix_rms / r
            mix = (mix * s).astype(np.float32)
            clean = (clean * s).astype(np.float32)
        mix = np.nan_to_num(mix, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
        out = (torch.from_numpy(np.ascontiguousarray(mix)),
               torch.from_numpy(np.ascontiguousarray(clean)),
               torch.tensor(float(snr), dtype=torch.float32))
        if self.return_meta:
            meta = meta or {}
            meta["snr_db"] = float(snr)
            return out + (meta,)
        return out


# ===========================================================================
# Fixed pre-mixed evaluation set (written by sound_mixer.py)
# ===========================================================================
class PremixedPairDataset(Dataset):
    """Reads (mixed/N.wav, target/N.wav) pairs produced by sound_mixer.py.

    sound_mixer.py now writes the exact clean component next to every mixture,
    so no reconstruction or renormalisation happens here — whatever is in
    target/ is used verbatim. That is the whole reason this is trustworthy as
    an evaluation set.
    """

    def __init__(self, root, rows, sample_rate=16000, segment_seconds=None,
                 mix_rms=0.1):
        self.root = Path(root)
        self.rows = list(rows)
        self.sr = sample_rate
        self.seg = int(round(segment_seconds * sample_rate)) if segment_seconds else None
        self.mix_rms = mix_rms

    def __len__(self):
        return len(self.rows)

    def _read(self, path):
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.sr:
            import torchaudio
            data = torchaudio.functional.resample(
                torch.from_numpy(np.ascontiguousarray(data)), sr, self.sr).numpy()
        return np.ascontiguousarray(data, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.rows[idx]
        mix = self._read(self.root / "mixed" / row["filename"])
        clean = self._read(self.root / "target" / row["filename"])
        n = min(mix.shape[0], clean.shape[0])
        mix, clean = mix[:n], clean[:n]

        if self.seg is not None:
            if n < self.seg:
                pad = self.seg - n
                mix = np.pad(mix, (0, pad))
                clean = np.pad(clean, (0, pad))
            else:
                start = (n - self.seg) // 2          # deterministic centre crop
                mix = mix[start:start + self.seg]
                clean = clean[start:start + self.seg]

        r = rms(mix)
        if r > 1e-8:
            s = self.mix_rms / r
            mix, clean = mix * s, clean * s

        snr = float(row.get("snr_db", 0.0) or 0.0)
        return (torch.from_numpy(np.ascontiguousarray(mix.astype(np.float32))),
                torch.from_numpy(np.ascontiguousarray(clean.astype(np.float32))),
                torch.tensor(snr, dtype=torch.float32))


# ===========================================================================
# Convenience: prepare everything from a data root
# ===========================================================================
def prepare_pools(data_root, cache_dir, sample_rate=16000, val_fraction=0.1,
                  rebuild_cache=False, verbose=True):
    """Discover, cache and split the three source folders.

    Returns (train_pools, val_pools) where each is a dict with keys
    'clean', 'noise', 'bg' holding lists of cache paths.
    """
    data_root = Path(data_root)
    pools_train, pools_val = {}, {}
    for kind, frac in (("clean", val_fraction), ("noise", val_fraction), ("bg", val_fraction)):
        files = list_audio_files(data_root / kind)
        if verbose:
            print(f"{kind}/: found {len(files)} file(s)")
        cached = build_cache(files, Path(cache_dir) / kind, sample_rate,
                             rebuild=rebuild_cache, verbose=verbose) if files else []
        tr, va = hash_split(cached, frac, salt=kind)
        pools_train[kind], pools_val[kind] = tr, va
        if verbose and cached:
            print(f"  split: {len(tr)} train / {len(va)} val")
    return pools_train, pools_val