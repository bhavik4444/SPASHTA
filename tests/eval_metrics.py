#!/usr/bin/env python3
"""
eval_metrics.py -- PESQ, STOI/eSTOI, SI-SDR for the enhancement model.

These are evaluation metrics, not loss terms -- nothing here touches training.
What they add over the SI-SDR verify_export.py already prints:

  PESQ (ITU-T P.862)   models a human listener's quality rating: distortion,
                       residual noise, artifacts. -0.5..4.5 (wb mode), higher
                       is better. Two modes:
                         'wb' (wideband, 16 kHz) -- use this, it's what the
                              model runs at
                         'nb' (narrowband, 8 kHz) -- only if you need to
                              compare against an 8 kHz baseline elsewhere
  STOI / eSTOI         intelligibility -- correlates with word-recognition
                       accuracy in listening tests far better than SI-SDR
                       does. 0..1, higher is better. eSTOI is the
                       "extended" variant, slightly more sensitive at low SNR.
  SI-SDR               energy-domain fidelity, scale-invariant. dB, higher is
                       better. Already used elsewhere in this project; kept
                       here too so one table has everything.

For a defence-comms use case, STOI is arguably the metric that matters most:
a clear-but-imperfect signal beats a quieter one a listener can't parse.

Install:
    pip install pesq pystoi soundfile numpy

Usage -- three ways to feed it audio:

  1. Explicit pair:
       python eval_metrics.py --clean target.wav --enhanced clean_out.wav

  2. Three-way (also scores the untouched noisy signal, so you can see how
     much the model actually bought you):
       python eval_metrics.py --clean target.wav --noisy noisy.wav \\
                              --enhanced clean_out.wav

  3. Folders, matched by filename -- this is what you want for a batch report
     over an eval set (same layout sound_mixer.py produces: mixed/, target/).
     Pass --enhanced_dir pointing at wherever you rendered your model's
     output for those same files (e.g. from gtcrn_ref.py or host_test):
       python eval_metrics.py --clean_dir eval_set/target \\
                              --noisy_dir eval_set/mixed \\
                              --enhanced_dir eval_set/enhanced \\
                              --csv report.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _require(pkg, pip_name=None):
    try:
        return __import__(pkg)
    except ImportError:
        sys.exit(f"missing dependency '{pkg}'. install it with:\n"
                 f"    pip install {pip_name or pkg}")


def load_mono(path, target_sr):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        # PESQ and STOI both require a specific rate; resample rather than
        # silently comparing at the wrong scale.
        import scipy.signal as sig
        n = int(round(len(x) * target_sr / sr))
        x = sig.resample(x, n).astype(np.float32)
    return x.astype(np.float32)


def si_sdr(est, ref):
    ref = ref - ref.mean()
    est = est - est.mean()
    denom = float(np.dot(ref, ref)) + 1e-12
    t = (float(np.dot(est, ref)) / denom) * ref
    e = est - t
    return 10.0 * np.log10((float(np.dot(t, t)) + 1e-12) / (float(np.dot(e, e)) + 1e-12))


def compute_all(clean, degraded, sr, pesq_mode="wb"):
    """clean, degraded: float32 arrays at `sr` Hz, same length.
    Returns a dict; a metric is None if it could not be computed (PESQ is
    picky about near-silent or degenerate input and will raise)."""
    n = min(len(clean), len(degraded))
    clean, degraded = clean[:n], degraded[:n]

    out = {"si_sdr": si_sdr(degraded, clean)}

    pesq_mod = _require("pesq")
    pesq_sr = 16000 if pesq_mode == "wb" else 8000
    try:
        if sr != pesq_sr:
            import scipy.signal as sig
            m = int(round(n * pesq_sr / sr))
            c = sig.resample(clean, m).astype(np.float32)
            d = sig.resample(degraded, m).astype(np.float32)
        else:
            c, d = clean, degraded
        out["pesq"] = float(pesq_mod.pesq(pesq_sr, c, d, pesq_mode))
    except Exception as e:                       # noqa: BLE001
        out["pesq"] = None
        out["pesq_error"] = str(e)

    stoi_mod = _require("pystoi", "pystoi")
    try:
        out["stoi"] = float(stoi_mod.stoi(clean, degraded, sr, extended=False))
        out["estoi"] = float(stoi_mod.stoi(clean, degraded, sr, extended=True))
    except Exception as e:                        # noqa: BLE001
        out["stoi"] = out["estoi"] = None
        out["stoi_error"] = str(e)

    return out


def fmt(v, width=7, nd=3):
    return f"{v:>{width}.{nd}f}" if v is not None else f"{'n/a':>{width}}"


def print_row(name, m):
    print(f"  {name:<20s} SI-SDR {fmt(m['si_sdr'])}  PESQ {fmt(m.get('pesq'))}  "
          f"STOI {fmt(m.get('stoi'))}  eSTOI {fmt(m.get('estoi'))}")

def save_metric_card(output_path, filename, noisy_metrics, enhanced_metrics):
    import matplotlib.pyplot as plt

    metrics = [
        ("PESQ", noisy_metrics.get("pesq"), enhanced_metrics.get("pesq"), 2),
        ("STOI", noisy_metrics.get("stoi"), enhanced_metrics.get("stoi"), 3),
        ("SI-SDR (dB)", noisy_metrics.get("si_sdr"), enhanced_metrics.get("si_sdr"), 2),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    ax.text(
        0.5, 0.94,
        f"ANC PERFORMANCE — {filename}",
        ha="center", va="center",
        fontsize=20, fontweight="bold"
    )

    ax.text(
        0.42, 0.84, "NOISY INPUT",
        ha="center", va="center",
        fontsize=13, fontweight="bold"
    )

    ax.text(
        0.64, 0.84, "PROCESSED OUTPUT",
        ha="center", va="center",
        fontsize=13, fontweight="bold"
    )

    ax.text(
        0.84, 0.84, "GAIN",
        ha="center", va="center",
        fontsize=13, fontweight="bold"
    )

    y_positions = [0.68, 0.50, 0.32]

    for (label, noisy, enhanced, decimals), y in zip(metrics, y_positions):

        if noisy is None or enhanced is None:
            noisy_text = "N/A"
            enhanced_text = "N/A"
            gain_text = "N/A"
        else:
            noisy_text = f"{noisy:.{decimals}f}"
            enhanced_text = f"{enhanced:.{decimals}f}"
            gain = enhanced - noisy
            gain_text = f"{gain:+.{decimals}f}"

        ax.text(
            0.14, y,
            label,
            ha="left", va="center",
            fontsize=17, fontweight="bold"
        )

        ax.text(
            0.42, y,
            noisy_text,
            ha="center", va="center",
            fontsize=22
        )

        ax.text(
            0.64, y,
            enhanced_text,
            ha="center", va="center",
            fontsize=22
        )

        ax.text(
            0.84, y,
            gain_text,
            ha="center", va="center",
            fontsize=22, fontweight="bold"
        )

    ax.text(
        0.5, 0.13,
        "Positive gain = improvement over noisy input",
        ha="center", va="center",
        fontsize=12
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"saved metric card: {output_path}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description=__doc__.split("Usage")[0])
    p.add_argument("--sample_rate", type=int, default=16000,
                   help="rate the model runs at; STOI is computed at this "
                        "rate, PESQ is resampled to 16k (wb) or 8k (nb)")
    p.add_argument("--pesq_mode", choices=["wb", "nb"], default="wb")

    p.add_argument("--clean")
    p.add_argument("--noisy")
    p.add_argument("--enhanced")

    p.add_argument("--clean_dir")
    p.add_argument("--noisy_dir")
    p.add_argument("--enhanced_dir")
    p.add_argument("--csv", help="write a per-file CSV report here")
    p.add_argument(
        "--plot",
        nargs="?",
        const="auto",
        default=None,
        help="generate a PNG metric card; optionally provide output path"
    )
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    rows = []

    if args.clean and args.enhanced:
        clean = load_mono(args.clean, args.sample_rate)
        enh = load_mono(args.enhanced, args.sample_rate)

        print(f"\n{Path(args.enhanced).name}")

        mn = None

        if args.noisy:
            noisy = load_mono(args.noisy, args.sample_rate)
            mn = compute_all(clean, noisy, args.sample_rate, args.pesq_mode)

            print_row("noisy (baseline)", mn)
            rows.append({
                "file": Path(args.enhanced).name,
                "stage": "noisy",
                **mn
            })

        me = compute_all(clean, enh, args.sample_rate, args.pesq_mode)

        print_row("enhanced", me)

        rows.append({
            "file": Path(args.enhanced).name,
            "stage": "enhanced",
            **me
        })

        # Generate visual metric card
        if args.plot and mn is not None:

            if args.plot == "auto":
                output_path = (
                    Path(args.enhanced).parent
                    / f"metrics_comparison_{Path(args.enhanced).stem}.png"
                )
            else:
                output_path = Path(args.plot)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            save_metric_card(
                output_path,
                Path(args.enhanced).name,
                mn,
                me
            )

    elif args.clean_dir and args.enhanced_dir:
        clean_dir = Path(args.clean_dir)
        enh_dir = Path(args.enhanced_dir)
        noisy_dir = Path(args.noisy_dir) if args.noisy_dir else None

        files = sorted(p_.name for p_ in enh_dir.glob("*.wav"))
        if args.limit:
            files = files[:args.limit]
        if not files:
            sys.exit(f"no .wav files found in {enh_dir}")

        agg = {"noisy": [], "enhanced": []}
        for name in files:
            cpath = clean_dir / name
            epath = enh_dir / name
            if not cpath.exists():
                print(f"  skip {name}: no matching file in {clean_dir}")
                continue
            clean = load_mono(cpath, args.sample_rate)
            enh = load_mono(epath, args.sample_rate)

            print(f"\n{name}")
            if noisy_dir and (noisy_dir / name).exists():
                noisy = load_mono(noisy_dir / name, args.sample_rate)
                mn = compute_all(clean, noisy, args.sample_rate, args.pesq_mode)
                print_row("noisy", mn)
                rows.append({"file": name, "stage": "noisy", **mn})
                agg["noisy"].append(mn)
            me = compute_all(clean, enh, args.sample_rate, args.pesq_mode)
            print_row("enhanced", me)
            rows.append({"file": name, "stage": "enhanced", **me})
            agg["enhanced"].append(me)

        print(f"\n{'=' * 60}")
        for stage, ms in agg.items():
            if not ms:
                continue
            def avg(key):
                vals = [m[key] for m in ms if m.get(key) is not None]
                return float(np.mean(vals)) if vals else None
            print(f"  {stage:<10s} mean  SI-SDR {fmt(avg('si_sdr'))}  "
                  f"PESQ {fmt(avg('pesq'))}  STOI {fmt(avg('stoi'))}  "
                  f"eSTOI {fmt(avg('estoi'))}   (n={len(ms)})")
        if agg["noisy"] and agg["enhanced"]:
            def avg(ms, key):
                vals = [m[key] for m in ms if m.get(key) is not None]
                return float(np.mean(vals)) if vals else None
            for key, label in (("si_sdr", "SI-SDR"), ("pesq", "PESQ"),
                               ("stoi", "STOI"), ("estoi", "eSTOI")):
                a, b = avg(agg["noisy"], key), avg(agg["enhanced"], key)
                if a is not None and b is not None:
                    print(f"  gain over noisy input, {label}: {b - a:+.3f}")

                # Generate dataset-level metric summary image
        if args.plot and agg["noisy"] and agg["enhanced"]:

            def avg(ms, key):
                vals = [m[key] for m in ms if m.get(key) is not None]
                return float(np.mean(vals)) if vals else None

            noisy_mean = {
                "pesq": avg(agg["noisy"], "pesq"),
                "stoi": avg(agg["noisy"], "stoi"),
                "si_sdr": avg(agg["noisy"], "si_sdr"),
            }

            enhanced_mean = {
                "pesq": avg(agg["enhanced"], "pesq"),
                "stoi": avg(agg["enhanced"], "stoi"),
                "si_sdr": avg(agg["enhanced"], "si_sdr"),
            }

            if args.plot == "auto":
                output_path = Path("eval_set/metrics_summary.png")
            else:
                output_path = Path(args.plot)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            save_metric_card(
                output_path,
                f"Dataset Mean ({len(agg['enhanced'])} files)",
                noisy_mean,
                enhanced_mean
            )
    else:
        p.error("pass either --clean/--enhanced (one file) or "
                "--clean_dir/--enhanced_dir (a folder)")

    if args.csv and rows:
        import csv
        keys = ["file", "stage", "si_sdr", "pesq", "stoi", "estoi"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
