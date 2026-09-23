#!/usr/bin/env python3
"""
make_synthetic_blob.py -- build a structurally valid blob full of random weights.

The point is not audio quality (the output is noise). The point is that it lets
you run the C runtime against gtcrn_ref.py on a laptop, with no PyTorch and no
board, and confirm that the two agree to float precision. That check catches the
bugs that actually bite in a port of this shape: a transposed weight layout, a
ring buffer off by one dilation step, the tap parity in the strided transposed
convolution, a GRU gate in the wrong order. Quantisation error is graceful;
those are not.

    python make_synthetic_blob.py --out /tmp/fake.bin --frames 8
    gcc -O2 -std=c99 -I ../firmware/components/gtcrn/include \
        -o /tmp/host_test host_test.c \
        ../firmware/components/gtcrn/gtcrn_ops.c \
        ../firmware/components/gtcrn/gtcrn_net.c -lm
    /tmp/host_test /tmp/fake.bin /tmp/fake_in.f32 /tmp/out_c.f32
    python make_synthetic_blob.py --out /tmp/fake.bin --frames 8 --run_ref
    python -c "import numpy as np; a=np.fromfile('/tmp/out_c.f32',np.float32); \
b=np.fromfile('/tmp/out_ref.f32',np.float32); \
print('max abs diff', np.abs(a-b).max())"
"""
import argparse

import numpy as np

from export_int8 import Blob

CFG = dict(
    sample_rate=16000, n_fft=512, hop_length=256, n_freqs=257,
    erb_subband_1=65, erb_subband_2=64, width=129, bn_width=33,
    base_channels=32, n_dpgrnn=3, tra_bands=8, n_dil=5, dil=[1, 2, 4, 8, 16],
    df_order=5, df_bins=64, n_out=13,
    mask_max=1.5, mask_min=0.0, compress=0.3,
    level_frames=192, smooth_frames=4, nf_frames=96,
    nf_bias=1.6, snr_lo=-1.0, snr_hi=2.5, mix_rms=0.1,
)


def add_gru(blob, rng, pfx, n_in, H):
    blob.i8_rows(pfx + "wih", rng.standard_normal((3 * H, n_in)) * 0.3)
    blob.i8_rows(pfx + "whh", rng.standard_normal((3 * H, H)) * 0.3)
    blob.f32(pfx + "bih", rng.standard_normal(3 * H) * 0.05)
    blob.f32(pfx + "bhh", rng.standard_normal(3 * H) * 0.05)


def add_block(blob, rng, pfx, C):
    half = C // 2
    blob.i8_rows(pfx + "pc1.w", rng.standard_normal((C, 3 * half)) * 0.15)
    blob.f32(pfx + "pc1.b", rng.standard_normal(C) * 0.02)
    blob.f32(pfx + "pc1.a", [0.25])
    blob.f32(pfx + "dw.w", rng.standard_normal((C, 3, 3)) * 0.2)
    blob.f32(pfx + "dw.b", rng.standard_normal(C) * 0.02)
    blob.f32(pfx + "dw.a", [0.25])
    blob.i8_rows(pfx + "pc2.w", rng.standard_normal((half, C)) * 0.15)
    blob.f32(pfx + "pc2.b", rng.standard_normal(half) * 0.02)
    add_gru(blob, rng, pfx + "tra.", half, 2 * half)
    blob.i8_rows(pfx + "tra.fc.w", rng.standard_normal((half, 2 * half)) * 0.2)
    blob.f32(pfx + "tra.fc.b", rng.standard_normal(half) * 0.02)


def add_dp(blob, rng, k, C, Wb):
    h2, Hb = C // 2, C // 4
    for r in (1, 2):
        for d in ("f.", "b."):
            add_gru(blob, rng, f"dp{k}.ia{r}.{d}", h2, Hb)
    blob.i8_rows(f"dp{k}.ia.fc.w", rng.standard_normal((C, C)) * 0.2)
    blob.f32(f"dp{k}.ia.fc.b", rng.standard_normal(C) * 0.02)
    blob.f32(f"dp{k}.ia.ln.w", np.ones((Wb, C)) + rng.standard_normal((Wb, C)) * 0.05)
    blob.f32(f"dp{k}.ia.ln.b", rng.standard_normal((Wb, C)) * 0.02)
    for r in (1, 2):
        add_gru(blob, rng, f"dp{k}.ie{r}.", h2, h2)
    blob.i8_rows(f"dp{k}.ie.fc.w", rng.standard_normal((C, C)) * 0.2)
    blob.f32(f"dp{k}.ie.fc.b", rng.standard_normal(C) * 0.02)
    blob.f32(f"dp{k}.ie.ln.w", np.ones((Wb, C)) + rng.standard_normal((Wb, C)) * 0.05)
    blob.f32(f"dp{k}.ie.ln.b", rng.standard_normal((Wb, C)) * 0.02)


def triangular_erb(n_bins, n_bands):
    """Overlapping triangles, one contiguous run per band -- the same sparsity
    pattern a real ERB filterbank has."""
    edges = np.linspace(0, n_bins, n_bands + 2)
    fb = np.zeros((n_bands, n_bins), dtype=np.float64)
    for b in range(n_bands):
        lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
        for i in range(n_bins):
            if lo <= i < mid:
                fb[b, i] = (i - lo) / max(mid - lo, 1e-9)
            elif mid <= i < hi:
                fb[b, i] = (hi - i) / max(hi - mid, 1e-9)
    fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-9)
    return fb


def add_erb(blob, n_hi, n_bands):
    fb = triangular_erb(n_hi, n_bands)
    for tag, mat in (("bm", fb), ("bs", fb.T)):
        starts, lens, vals = [], [], []
        for r in range(mat.shape[0]):
            nz = np.nonzero(np.abs(mat[r]) > 1e-9)[0]
            if len(nz) == 0:
                starts.append(0); lens.append(0); continue
            a, b = int(nz[0]), int(nz[-1]) + 1
            starts.append(a); lens.append(b - a)
            vals.extend(mat[r, a:b].tolist())
        blob.i32(f"erb.{tag}.start", np.array(starts, np.int32))
        blob.i32(f"erb.{tag}.len", np.array(lens, np.int32))
        blob.f32(f"erb.{tag}.w", np.array(vals, np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/fake.bin")
    p.add_argument("--input", default="/tmp/fake_in.f32")
    p.add_argument("--ref_out", default="/tmp/out_ref.f32")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--run_ref", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(1234)
    C, Wb = CFG["base_channels"], CFG["bn_width"]
    b = Blob()

    b.i8_rows("enc.c1.w", rng.standard_normal((C, 5 * 12)) * 0.12)
    b.f32("enc.c1.b", rng.standard_normal(C) * 0.02)
    b.f32("enc.c1.a", [0.25])
    b.i8_rows("enc.c2.w", rng.standard_normal((C, 5 * (C // 2))) * 0.12)
    b.f32("enc.c2.b", rng.standard_normal(C) * 0.02)
    b.f32("enc.c2.a", [0.25])
    for i in range(CFG["n_dil"]):
        add_block(b, rng, f"enc.g{i}.", C)
    for k in range(CFG["n_dpgrnn"]):
        add_dp(b, rng, k, C, Wb)
    for i in range(CFG["n_dil"]):
        add_block(b, rng, f"dec.g{i}.", C)
    b.i8_rows("dec.c1.w", rng.standard_normal((C, 5 * (C // 2))) * 0.12)
    b.f32("dec.c1.b", rng.standard_normal(C) * 0.02)
    b.f32("dec.c1.a", [0.25])
    b.i8_rows("dec.head.w", rng.standard_normal((CFG["n_out"], 5 * C)) * 0.1)
    b.f32("dec.head.b", rng.standard_normal(CFG["n_out"]) * 0.02)
    add_erb(b, CFG["n_freqs"] - CFG["erb_subband_1"], CFG["erb_subband_2"])

    size = b.write(args.out, CFG)
    print(f"blob: {args.out} ({size / 1024:.1f} kB, {len(b.entries)} tensors)")

    n = CFG["hop_length"] * args.frames
    t = np.arange(n) / CFG["sample_rate"]
    sig = (0.08 * np.sin(2 * np.pi * 180 * t)
           + 0.04 * np.sin(2 * np.pi * 620 * t)
           + 0.02 * rng.standard_normal(n)).astype(np.float32)
    sig.tofile(args.input)
    print(f"input: {args.input} ({n} samples)")

    if args.run_ref:
        from gtcrn_ref import Blob as RB, GtcrnRef
        m = GtcrnRef(RB(args.out))
        cfg = m.c
        out = np.zeros(n, np.float32)
        hop, N = cfg["hop_length"], cfg["n_fft"]
        win = np.sqrt(np.maximum(
            0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N), 0.0)).astype(np.float32)
        ana = np.zeros(N, np.float32)
        ola = np.zeros(N, np.float32)
        for h in range(args.frames):
            ana[:N - hop] = ana[hop:]
            ana[N - hop:] = sig[h * hop:(h + 1) * hop]
            S = np.fft.rfft(ana * win)
            er, ei = m.frame(S.real.astype(np.float32), S.imag.astype(np.float32))
            y = np.fft.irfft(er + 1j * ei, N).astype(np.float32)
            ola += y * win
            out[h * hop:(h + 1) * hop] = ola[:hop]
            ola[:N - hop] = ola[hop:]
            ola[N - hop:] = 0.0
            print(f"  frame {h + 1}/{args.frames}", end="\r", flush=True)
        out.tofile(args.ref_out)
        print(f"\nreference output: {args.ref_out}")


if __name__ == "__main__":
    main()
