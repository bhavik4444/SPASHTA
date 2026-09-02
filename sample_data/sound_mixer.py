"""
Sound Mixer for Noisy-Speech Dataset Generation (v3)
-----------------------------------------------------
Scans folders of clean speech, noise, and background files, and builds
a dataset of mixed audio files in TWO phases:

  1. "normal" phase (NUM_NORMAL_SAMPLES files) -- clean speech stays clearly
     audible; noise/bg are present but don't dominate.
  2. "extreme" phase (NUM_EXTREME_SAMPLES files) -- noise/bg are deliberately
     loud (can exceed speech loudness) and near-guaranteed present, for
     training a model to be robust under harsh conditions.

Each sample can combine a DIFFERENT NUMBER of files from each folder -- e.g.
one sample might mix 1 clean + 1 noise, another 1 clean + 2 noise + 2 bg.

Certain bg files can be boosted to appear more often than others (see
BG_FILE_BOOST) -- e.g. making "static.wav" show up ~4x as often as other
background files, if that's a noise condition you want over-represented.

Folder structure expected (relative to this script):

    clean/     -> one or more clean speech files
    noise/     -> one or more noise files (e.g. gunshots, artillery)
    bg/        -> one or more background files (e.g. wind, static, engine hum)
    mixed_dataset/  -> (auto-created) output mixed .wav files, named 1.wav, 2.wav, ...
    csvs/           -> (auto-created) mix_log.csv listing every file's exact recipe

Two mixing styles, chosen randomly per sample (see MODE_WEIGHTS):
  - "ratio": every selected file gets a random weight, all summed, peak-normalized.
  - "snr":   selected clean files are summed into one reference signal; selected
             noise files are summed and scaled to a target SNR (dB) relative to
             that reference; same for bg files; then everything is summed.

Usage:
    python sound_mixer.py
"""

import os
import csv
import random
import numpy as np
import soundfile as sf

# ============================== CONFIG ===============================

CLEAN_DIR = "clean"
NOISE_DIR = "noise"
BG_DIR = "bg"

OUTPUT_DIR = "mixed_dataset"
CSV_DIR = "csvs"
CSV_LOG_PATH = os.path.join(CSV_DIR, "mix_log.csv")

TARGET_SR = 16000
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")

NUM_NORMAL_SAMPLES = 150
NUM_EXTREME_SAMPLES = 50
RANDOM_SEED = 42

# How many files to pull from each folder for a single sample (inclusive range).
CLEAN_COUNT_RANGE = (1, 1)   # always exactly 1 clean speech file per sample
NOISE_COUNT_RANGE = (0, 2)
BG_COUNT_RANGE = (0, 2)

# Extreme phase forces noise to (almost) always be present, since the point
# is robustness under harsh/loud interference.
NOISE_COUNT_RANGE_EXTREME = (1, 2)
BG_COUNT_RANGE_EXTREME = (0, 2)

# A sample must include at least one non-clean file so it isn't pure clean speech.
REQUIRE_AT_LEAST_ONE_NOISE_OR_BG = True

# All loaded clips are first normalized to this common RMS loudness before any
# mixing happens. Without this, a naturally loud/punchy clip (e.g. a gunshot
# burst) can drown out a naturally quieter clip (e.g. speech) regardless of the
# "weight" it's given below.
TARGET_INPUT_RMS = 0.1

# ---- Normal-phase weight ranges (ratio mode) -- speech stays clearly audible ----
CLEAN_WEIGHT_RANGE = (0.7, 1.0)
NOISE_WEIGHT_RANGE = (0.2, 0.6)
BG_WEIGHT_RANGE = (0.15, 0.5)

# ---- Extreme-phase weight ranges (ratio mode) -- tuned so worst-case speech
# suppression bottoms out around -5dB even when 2 noise + 2 bg files stack ----
CLEAN_WEIGHT_RANGE_EXTREME = (0.6, 0.85)
NOISE_WEIGHT_RANGE_EXTREME = (0.35, 0.55)
BG_WEIGHT_RANGE_EXTREME = (0.3, 0.5)

# ---- Normal-phase SNR ranges (dB), "snr" mode ----
# 0 dB = noise as loud as speech; kept >= 0 so speech is never buried.
NOISE_SNR_RANGE_DB = (0, 20)
BG_SNR_RANGE_DB = (0, 20)

# ---- Extreme-phase SNR ranges (dB) -- floor capped at -5dB so speech is
# stressed but never buried too deep ----
NOISE_SNR_RANGE_DB_EXTREME = (-5, 5)
BG_SNR_RANGE_DB_EXTREME = (-5, 8)

# Probability of using "ratio" vs "snr" mode for a given sample (same for both phases).
MODE_WEIGHTS = {"ratio": 0.5, "snr": 0.5}

# Boost specific files' odds of being picked when sampling a folder.
# Key = filename (case-insensitive, matched against the basename), value =
# relative weight multiplier (1.0 = normal odds). E.g. 4.0 means that file
# is ~4x more likely to be selected than an unboosted file in the same pull.
# Currently applied to the bg/ folder; extend BOOST_MAPS below to add more.
BG_FILE_BOOST = {
    "static.wav": 4.0,
}
CLEAN_FILE_BOOST = {}
NOISE_FILE_BOOST = {}

# =======================================================================


def list_audio_files(folder):
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(AUDIO_EXTS)
    ]


def load_audio(path, target_sr=TARGET_SR):
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = resample_audio(data, sr, target_sr)
    data = data.astype(np.float32)
    data = normalize_rms(data, TARGET_INPUT_RMS)
    return data


def resample_audio(data, orig_sr, target_sr):
    duration = len(data) / orig_sr
    target_len = int(duration * target_sr)
    orig_x = np.linspace(0, duration, num=len(data))
    target_x = np.linspace(0, duration, num=target_len)
    return np.interp(target_x, orig_x, data)


def fit_length(signal, target_len):
    if len(signal) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(signal) >= target_len:
        return signal[:target_len]
    reps = int(np.ceil(target_len / len(signal)))
    return np.tile(signal, reps)[:target_len]


def rms(signal):
    return np.sqrt(np.mean(signal ** 2) + 1e-12)


def normalize_rms(signal, target_rms=0.1):
    """Scale signal so its RMS loudness equals target_rms. Puts every source
    clip on an even loudness footing before any weighting/SNR logic runs."""
    current = rms(signal)
    if current < 1e-6:
        return signal
    return signal * (target_rms / current)


def normalize_peak(signal, peak=0.95):
    max_val = np.max(np.abs(signal)) + 1e-12
    if max_val > peak:
        signal = signal * (peak / max_val)
    return signal


def load_group(cache, paths, target_len):
    """Load each file in `paths`, fit to target_len, return list of arrays."""
    out = []
    for p in paths:
        if p not in cache:
            cache[p] = load_audio(p)
        out.append(fit_length(cache[p], target_len))
    return out


def file_weight(path, boost_map):
    """Return the sampling weight for a file given a {filename: multiplier}
    boost map (case-insensitive basename match). Unlisted files get weight 1.0."""
    base = os.path.basename(path).lower()
    for name, mult in boost_map.items():
        if base == name.lower():
            return mult
    return 1.0


def sample_subset(files, count_range, rng, boost_map=None):
    """Pick a random-sized subset of `files` within count_range (inclusive).
    If boost_map is given, files matching it are proportionally more likely
    to be picked (sampling without replacement, weighted)."""
    if not files:
        return []
    lo, hi = count_range
    hi = min(hi, len(files))
    lo = min(lo, hi)
    if hi <= 0:
        return []
    count = rng.randint(lo, hi)
    if count == 0:
        return []

    if not boost_map:
        return rng.sample(files, count)

    # Weighted sampling without replacement.
    remaining = list(files)
    weights = [file_weight(p, boost_map) for p in remaining]
    chosen = []
    for _ in range(count):
        total = sum(weights)
        r = rng.uniform(0, total)
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                chosen.append(remaining.pop(i))
                weights.pop(i)
                break
    return chosen


def mix_ratio_group(clean_files, noise_files, bg_files, cache, target_len, rng, weight_ranges):
    """Random-weight mixing: every selected file (regardless of folder) gets
    its own random weight drawn from its category's range, all are summed,
    then peak-normalized. Every clip was already RMS-normalized at load time,
    so weights here control relative loudness predictably."""
    all_paths = [(p, "clean") for p in clean_files] + \
                [(p, "noise") for p in noise_files] + \
                [(p, "bg") for p in bg_files]

    mix = np.zeros(target_len, dtype=np.float32)
    weight_log = []  # list of (path, category, weight)

    for path, category in all_paths:
        sig = load_group(cache, [path], target_len)[0]
        w = round(rng.uniform(*weight_ranges[category]), 3)
        mix += w * sig
        weight_log.append((path, category, w))

    mix = normalize_peak(mix)
    return mix, weight_log


def mix_snr_group(clean_files, noise_files, bg_files, cache, target_len, rng, noise_snr_range, bg_snr_range):
    """Composite-SNR mixing: sums each group into one reference/noise/bg signal,
    scales noise & bg groups to random target SNRs relative to the clean group."""
    clean_signals = load_group(cache, clean_files, target_len)
    noise_signals = load_group(cache, noise_files, target_len)
    bg_signals = load_group(cache, bg_files, target_len)

    clean_composite = np.sum(clean_signals, axis=0) if clean_signals else np.zeros(target_len, dtype=np.float32)
    noise_composite = np.sum(noise_signals, axis=0) if noise_signals else None
    bg_composite = np.sum(bg_signals, axis=0) if bg_signals else None

    mix = clean_composite.copy()
    noise_snr_db = None
    bg_snr_db = None

    if noise_composite is not None:
        noise_snr_db = round(rng.uniform(*noise_snr_range), 2)
        clean_rms = rms(clean_composite) if clean_signals else rms(noise_composite)
        target_noise_rms = clean_rms / (10 ** (noise_snr_db / 20))
        scale = target_noise_rms / rms(noise_composite)
        mix = mix + noise_composite * scale

    if bg_composite is not None:
        bg_snr_db = round(rng.uniform(*bg_snr_range), 2)
        clean_rms = rms(clean_composite) if clean_signals else rms(bg_composite)
        target_bg_rms = clean_rms / (10 ** (bg_snr_db / 20))
        scale = target_bg_rms / rms(bg_composite)
        mix = mix + bg_composite * scale

    mix = normalize_peak(mix)

    weight_log = [(p, "clean", "") for p in clean_files]
    weight_log += [(p, "noise", noise_snr_db) for p in noise_files]
    weight_log += [(p, "bg", bg_snr_db) for p in bg_files]
    return mix, weight_log


def generate_batch(difficulty, count, start_index, pools, cache, rng, rows,
                    noise_count_range, bg_count_range, weight_ranges,
                    noise_snr_range, bg_snr_range):
    """Generate `count` mixed samples of the given difficulty, appending rows
    to `rows` and files to disk. Returns the next available file index."""
    clean_pool, noise_pool, bg_pool = pools
    modes = list(MODE_WEIGHTS.keys())
    mode_probs = list(MODE_WEIGHTS.values())

    generated = 0
    attempts = 0
    max_attempts = count * 20
    idx = start_index

    while generated < count and attempts < max_attempts:
        attempts += 1

        clean_sel = sample_subset(clean_pool, CLEAN_COUNT_RANGE, rng, CLEAN_FILE_BOOST)
        noise_sel = sample_subset(noise_pool, noise_count_range, rng, NOISE_FILE_BOOST)
        bg_sel = sample_subset(bg_pool, bg_count_range, rng, BG_FILE_BOOST)

        if not clean_sel:
            continue
        if REQUIRE_AT_LEAST_ONE_NOISE_OR_BG and not noise_sel and not bg_sel:
            continue

        clean_signals_raw = []
        for p in clean_sel:
            if p not in cache:
                cache[p] = load_audio(p)
            clean_signals_raw.append(cache[p])
        target_len = max(len(s) for s in clean_signals_raw)

        mode = rng.choices(modes, weights=mode_probs, k=1)[0]

        if mode == "ratio":
            mix, weight_log = mix_ratio_group(clean_sel, noise_sel, bg_sel, cache, target_len, rng, weight_ranges)
        else:
            mix, weight_log = mix_snr_group(clean_sel, noise_sel, bg_sel, cache, target_len, rng,
                                             noise_snr_range, bg_snr_range)

        generated += 1
        fname = f"{idx}.wav"
        sf.write(os.path.join(OUTPUT_DIR, fname), mix, TARGET_SR)

        clean_used = [p for p, cat, w in weight_log if cat == "clean"]
        noise_used = [p for p, cat, w in weight_log if cat == "noise"]
        bg_used = [p for p, cat, w in weight_log if cat == "bg"]
        clean_w = [w for p, cat, w in weight_log if cat == "clean"]
        noise_w = [w for p, cat, w in weight_log if cat == "noise"]
        bg_w = [w for p, cat, w in weight_log if cat == "bg"]

        rows.append({
            "filename": fname,
            "difficulty": difficulty,
            "mode": mode,
            "num_clean": len(clean_used),
            "num_noise": len(noise_used),
            "num_bg": len(bg_used),
            "clean_files": ";".join(os.path.basename(p) for p in clean_used),
            "noise_files": ";".join(os.path.basename(p) for p in noise_used),
            "bg_files": ";".join(os.path.basename(p) for p in bg_used),
            "clean_weights": ";".join(str(w) for w in clean_w),
            "noise_weights_or_snr_db": ";".join(str(w) for w in noise_w),
            "bg_weights_or_snr_db": ";".join(str(w) for w in bg_w),
            "duration_sec": round(len(mix) / TARGET_SR, 3),
        })

        idx += 1
        if generated % 10 == 0 or generated == count:
            print(f"  [{difficulty}] generated {generated}/{count}")

    if generated < count:
        print(f"  Note: [{difficulty}] stopped early after {attempts} attempts "
              f"-- check folder contents / count ranges if this is fewer than expected.")

    return idx


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    clean_pool = list_audio_files(CLEAN_DIR)
    noise_pool = list_audio_files(NOISE_DIR)
    bg_pool = list_audio_files(BG_DIR)

    print(f"Found {len(clean_pool)} clean, {len(noise_pool)} noise, {len(bg_pool)} bg file(s)")
    if not clean_pool:
        print(f"No clean speech files found in '{CLEAN_DIR}/'. Nothing to do.")
        return

    audio_cache = {}
    rows = []
    pools = (clean_pool, noise_pool, bg_pool)

    normal_weight_ranges = {"clean": CLEAN_WEIGHT_RANGE, "noise": NOISE_WEIGHT_RANGE, "bg": BG_WEIGHT_RANGE}
    extreme_weight_ranges = {"clean": CLEAN_WEIGHT_RANGE_EXTREME, "noise": NOISE_WEIGHT_RANGE_EXTREME, "bg": BG_WEIGHT_RANGE_EXTREME}

    next_idx = 1
    next_idx = generate_batch(
        "normal", NUM_NORMAL_SAMPLES, next_idx, pools, audio_cache, rng, rows,
        NOISE_COUNT_RANGE, BG_COUNT_RANGE, normal_weight_ranges,
        NOISE_SNR_RANGE_DB, BG_SNR_RANGE_DB,
    )
    next_idx = generate_batch(
        "extreme", NUM_EXTREME_SAMPLES, next_idx, pools, audio_cache, rng, rows,
        NOISE_COUNT_RANGE_EXTREME, BG_COUNT_RANGE_EXTREME, extreme_weight_ranges,
        NOISE_SNR_RANGE_DB_EXTREME, BG_SNR_RANGE_DB_EXTREME,
    )

    fieldnames = ["filename", "difficulty", "mode", "num_clean", "num_noise", "num_bg",
                  "clean_files", "noise_files", "bg_files",
                  "clean_weights", "noise_weights_or_snr_db", "bg_weights_or_snr_db",
                  "duration_sec"]
    with open(CSV_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"\nDone. Generated {total} files in '{OUTPUT_DIR}/'")
    print(f"Log written to '{CSV_LOG_PATH}'")


if __name__ == "__main__":
    main()