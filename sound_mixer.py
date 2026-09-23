"""
Sound Mixer v5 — fixed held-out evaluation set generator.

WHAT CHANGED, AND WHY IT MATTERS
================================
This script no longer generates the training set. Training now mixes on the fly
in the dataloader (see dataset.py), which gives unlimited effective data and,
more importantly, an exact clean target. This script exists for one job: to
write a FIXED, reproducible evaluation set you can point at, listen to, and
quote numbers from in a report.

Two substantive changes from v4:

1. IT WRITES THE CLEAN TARGET NEXT TO EVERY MIXTURE.
   v4 wrote only the mixture, and train.py reconstructed the target by
   RMS-normalising the original clean file. That reconstruction was wrong.
   v4 scaled clean by a random weight (0.55-1.0 in the gunfire phase) and then
   peak-normalised the whole sum by a factor it never logged, so the true clean
   component differed from the assumed target by a random, unrecoverable factor
   of several dB — worst precisely when noise was loudest, because that is when
   peak normalisation bites hardest. Now target/N.wav is the actual speech
   component of mixed/N.wav, sample for sample. Nothing has to be inferred.

2. IT SHARES ONE MIXING IMPLEMENTATION WITH TRAINING.
   All the mixing logic lives in dataset.DynamicMixDataset. This script just
   configures it per phase and writes the results out. Having two
   implementations of "what is in this mixture" is exactly how the target drift
   above went unnoticed for so long, so there is now only one.

Output layout (under OUTPUT_ROOT):
    mixed/   1.wav, 2.wav, ...   the noisy mixtures
    target/  1.wav, 2.wav, ...   the exact clean component of each mixture
    csvs/mix_log.csv             per-file recipe, including the true SNR

Input layout (under DATA_ROOT, scanned recursively):
    clean/  speech
    noise/  impulsive defence noise (gunfire, artillery, blasts)
    bg/     stationary background (wind, static, hum, engine drone)

Usage:
    python sound_mixer.py                          # defaults below
    python sound_mixer.py --data_root sample_data --out_root eval_set --n_per_phase 80

Then point training at it for a held-out number:
    python train.py ... --eval_root eval_set
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from dataset import DynamicMixDataset, load_name_map, prepare_pools

# Phases, chosen to span the operating range rather than to be "difficulties".
# Each is a slice of SNR, so the per-bucket numbers train.py prints line up with
# something you can describe in a report.
PHASES = [
    # name            snr range        noise ct   bg ct   burst prob
    ("easy",          (8.0, 20.0),     (0, 2),    (0, 2),  0.5),
    ("moderate",      (0.0, 8.0),      (1, 2),    (0, 2),  0.55),
    ("hard",          (-8.0, 0.0),     (1, 2),    (0, 2),  0.6),
    ("extreme",       (-20.0, -8.0),   (1, 3),    (0, 2),  0.65),
]


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root", type=str, default="sample_data",
                   help="folder holding clean/, noise/, bg/")
    p.add_argument("--out_root", type=str, default="eval_set")
    p.add_argument("--cache_dir", type=str, default=".cache_16k")
    p.add_argument("--n_per_phase", type=int, default=60)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--segment_seconds", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--mix_rms", type=float, default=0.1)
    p.add_argument("--peak_limit", type=float, default=0.95,
                   help="written files are scaled down if they would clip; the "
                        "target is scaled by the SAME factor so the pair stays exact")
    p.add_argument("--use_val_pool", action="store_true", default=True,
                   help="build the evaluation set from the held-out file split, "
                        "so evaluation speakers and noise recordings are ones "
                        "training never saw")
    p.add_argument("--use_all_files", dest="use_val_pool", action="store_false")
    p.add_argument("--val_fraction", type=float, default=0.08,
                   help="must match train.py's --val_fraction for the split to line up")
    args = p.parse_args()

    out_root = Path(args.out_root)
    (out_root / "mixed").mkdir(parents=True, exist_ok=True)
    (out_root / "target").mkdir(parents=True, exist_ok=True)
    (out_root / "csvs").mkdir(parents=True, exist_ok=True)

    print("Preparing source pools")
    pools_tr, pools_va = prepare_pools(args.data_root, args.cache_dir,
                                       args.sample_rate, args.val_fraction)
    pools = pools_va if args.use_val_pool else {
        k: pools_tr[k] + pools_va[k] for k in pools_tr}
    print(f"\nUsing the {'held-out' if args.use_val_pool else 'full'} pool: "
          f"{len(pools['clean'])} clean, {len(pools['noise'])} noise, "
          f"{len(pools['bg'])} bg")
    if not pools["clean"]:
        print("No clean speech available. Nothing to generate.")
        return
    if not pools["noise"] and not pools["bg"]:
        print("WARNING: no noise or bg files — every mixture would be clean speech.")

    # cache filenames are path hashes; map them back for a readable log
    name_map = {}
    for kind in ("clean", "noise", "bg"):
        name_map.update(load_name_map(Path(args.cache_dir) / kind))

    def pretty(path):
        return name_map.get(str(path), os.path.basename(str(path)))

    rows = []
    idx = 1
    # phase_i, not hash(phase): Python randomises string hashing per process
    # unless PYTHONHASHSEED is pinned, which would silently make this script
    # non-reproducible across runs.
    for phase_i, (phase, (snr_lo, snr_hi), n_range, bg_range, p_burst) in enumerate(PHASES):
        if args.n_per_phase <= 0:
            continue
        ds = DynamicMixDataset(
            pools["clean"], pools["noise"], pools["bg"],
            sample_rate=args.sample_rate, segment_seconds=args.segment_seconds,
            length=args.n_per_phase, snr_min=snr_lo, snr_max=snr_hi,
            hard_frac=0.0,                     # phases already define the range
            p_speech_only=0.0, p_noise_only=0.0,
            n_noise_range=n_range, n_bg_range=bg_range, p_burst=p_burst,
            mix_rms=args.mix_rms, p_clip=0.0,  # keep the eval set clean of artefacts
            deterministic=True, seed=args.seed + 1000 * (phase_i + 1),
            return_meta=True)

        for i in range(len(ds)):
            mix_t, clean_t, snr_t, meta = ds[i]
            mix = mix_t.numpy()
            clean = clean_t.numpy()

            # one joint rescale so nothing clips on disk; applying it to both
            # keeps target/ an exact description of mixed/
            peak = float(np.max(np.abs(mix))) + 1e-12
            if peak > args.peak_limit:
                s = args.peak_limit / peak
                mix, clean = mix * s, clean * s

            name = f"{idx}.wav"
            sf.write(out_root / "mixed" / name, mix, args.sample_rate)
            sf.write(out_root / "target" / name, clean, args.sample_rate)

            rows.append({
                "filename": name,
                "phase": phase,
                "snr_db": round(float(snr_t), 2),
                "kind": meta.get("kind", ""),
                "clean_file": pretty(meta.get("clean_file", "")),
                "noise_files": ";".join(pretty(x) for x in meta.get("noise_files", [])),
                "bg_files": ";".join(pretty(x) for x in meta.get("bg_files", [])),
                "n_bursts": meta.get("n_bursts", 0),
                "duration_sec": round(len(mix) / args.sample_rate, 3),
            })
            idx += 1

        print(f"  [{phase}] wrote {args.n_per_phase} file(s), "
              f"SNR {snr_lo:+.0f}..{snr_hi:+.0f} dB")

    csv_path = out_root / "csvs" / "mix_log.csv"
    fields = ["filename", "phase", "snr_db", "kind", "clean_file", "noise_files",
              "bg_files", "n_bursts", "duration_sec"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. {len(rows)} mixture/target pairs in {out_root}/")
    print(f"Log: {csv_path}")
    print(f"\nEvery target/N.wav is the exact speech component of mixed/N.wav, "
          f"so you can verify a pair by subtracting one from the other and "
          f"listening to what is left — it should be pure noise.")
    print(f"\nUse it as a held-out set:")
    print(f"  python train.py --data_root {args.data_root} --eval_root {args.out_root}")


if __name__ == "__main__":
    main()