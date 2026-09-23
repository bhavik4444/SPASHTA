#!/usr/bin/env python3
"""
verify_export.py -- prove the int8 blob still sounds like the float model.

This is the gate. Run it on the held-out set that sound_mixer.py writes, look at
the SI-SDR column, and only flash a blob whose median delta is small. My
threshold: under 0.3 dB is noise, 0.3-0.8 dB is worth a QAT pass, over 1 dB
means something is structurally wrong (almost always a layout bug in the
exporter, not quantisation itself -- quantisation degrades gracefully, layout
bugs do not).

    python verify_export.py --checkpoint checkpoints_v1/best_model.pt \
                            --blob gtcrn_int8.bin --eval_root eval_set

It also dumps on-device test vectors, so the firmware can prove on boot that it
computes the same numbers as your laptop:

    python verify_export.py --checkpoint ... --blob ... --dump_vectors \
                            --vectors_out ../firmware/main/gtcrn_testvec.bin
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

from gtcrn_ref import Blob, GtcrnRef, enhance_wav, sqrt_hann


def si_sdr(est, ref):
    ref = ref - ref.mean()
    est = est - est.mean()
    denom = float(np.dot(ref, ref)) + 1e-12
    t = (float(np.dot(est, ref)) / denom) * ref
    e = est - t
    return 10.0 * np.log10((float(np.dot(t, t)) + 1e-12) / (float(np.dot(e, e)) + 1e-12))


def torch_enhance(model, stft, wav, device, mix_rms):
    import torch
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-12))
    s = mix_rms / rms if rms > 1e-9 else 1.0
    with torch.no_grad():
        t = torch.from_numpy((wav * s).astype(np.float32)).unsqueeze(0).to(device)
        spec = stft.stft(t)
        y = stft.istft(model(spec).float(), length=t.shape[-1])
    return (y.squeeze(0).cpu().numpy() / s).astype(np.float32)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--repo", default=".", help="folder containing model.py / train.py")
    p.add_argument("--eval_root", default="eval_set",
                   help="output of sound_mixer.py: mixed/ and target/")
    p.add_argument("--limit", type=int, default=24, help="files to score (0 = all)")
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--dump_vectors", action="store_true")
    p.add_argument("--vectors_out", default="gtcrn_testvec.bin")
    p.add_argument("--vector_frames", type=int, default=32)
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    import soundfile as sf
    from model import build_model_from_config
    from train import STFTFrontEnd

    blob = Blob(args.blob)
    cfg = blob.cfg
    ck = torch.load(args.checkpoint, map_location="cpu")
    tmodel = build_model_from_config(ck.get("config", {}))
    tmodel.load_state_dict(ck["model_state_dict" if args.weights == "ema" else "raw_state_dict"])
    tmodel.eval()
    stft = STFTFrontEnd(cfg["n_fft"], cfg["hop_length"])

    # ---------------- on-device test vectors ----------------
    if args.dump_vectors:
        rng = np.random.default_rng(7)
        n = cfg["hop_length"] * args.vector_frames + cfg["n_fft"]
        # speech-ish: a few harmonics plus noise, so the noise-floor tracker and
        # the mask both see something non-degenerate
        t = np.arange(n) / cfg["sample_rate"]
        wav = (0.05 * np.sin(2 * np.pi * 180 * t) + 0.03 * np.sin(2 * np.pi * 540 * t)
               + 0.02 * np.sin(2 * np.pi * 1450 * t)
               + 0.02 * rng.standard_normal(n)).astype(np.float32)
        ref = GtcrnRef(blob)
        y = enhance_wav(ref, wav, cfg)
        with open(args.vectors_out, "wb") as f:
            f.write(b"GTVEC\x00\x00\x01")
            f.write(struct.pack("<2I", n, cfg["sample_rate"]))
            f.write(wav.astype("<f4").tobytes())
            f.write(y.astype("<f4").tobytes())
        print(f"test vectors: {args.vectors_out} ({n} samples in + out)")

    # ---------------- scoring ----------------
    root = Path(args.eval_root)
    mixed = sorted((root / "mixed").glob("*.wav"))
    if not mixed:
        print(f"no mixtures under {root}/mixed -- generate them with sound_mixer.py")
        return
    if args.limit:
        mixed = mixed[:args.limit]

    rows = []
    for m in mixed:
        tgt = root / "target" / m.name
        if not tgt.exists():
            continue
        x, _ = sf.read(m, dtype="float32")
        ref_wav, _ = sf.read(tgt, dtype="float32")
        n = min(len(x), len(ref_wav))
        x, ref_wav = x[:n], ref_wav[:n]

        yf = torch_enhance(tmodel, stft, x, "cpu", cfg["mix_rms"])[:n]
        rmodel = GtcrnRef(blob)
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
        s = cfg["mix_rms"] / rms if rms > 1e-9 else 1.0
        yq = (enhance_wav(rmodel, (x * s).astype(np.float32), cfg) / s)[:n]

        a = si_sdr(yf, ref_wav)
        b = si_sdr(yq, ref_wav)
        rows.append((m.name, si_sdr(x, ref_wav), a, b, b - a))
        print(f"  {m.name:<12s} in {rows[-1][1]:+6.2f}  float {a:+6.2f}  "
              f"int8 {b:+6.2f}  delta {b - a:+5.2f} dB")

    if rows:
        d = np.array([r[4] for r in rows])
        print(f"\n{len(rows)} files | mean delta {d.mean():+.3f} dB | "
              f"median {np.median(d):+.3f} dB | worst {d.min():+.3f} dB")
        verdict = ("ship it" if np.median(d) > -0.3 else
                   "run qat_finetune.py" if np.median(d) > -0.8 else
                   "check the exporter -- this is too large for quantisation alone")
        print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
