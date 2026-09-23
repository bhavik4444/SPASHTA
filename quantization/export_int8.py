#!/usr/bin/env python3
"""
export_int8.py -- GTCRN-DF checkpoint  ->  gtcrn_int8.bin

Why this exists instead of a .tflite
====================================
This model cannot survive a PyTorch -> ONNX -> TF -> TFLite-Micro pipeline, and
that is not a bug in your export command. TFLite Micro has no complex dtype
(the deep filter is a complex FIR), no streaming GRU state (BandTRA and both
DPGRNN passes carry hidden state across frames), no bidirectional-over-frequency
RNN primitive, and no way to express the causal left-pad + dilated depthwise
conv history that GTConvBlock needs. Converters "succeed" by unrolling the whole
utterance into a static graph, which then needs hundreds of kilobytes of tensor
arena per second of audio and cannot run frame-by-frame at all.

So we skip the converter entirely. This script writes the weights into a flat
binary and the C runtime in firmware/components/gtcrn implements the same
forward pass directly. That gives exact control over numerics, O(1) memory per
frame, and no dependency on op coverage in someone else's interpreter.

Quantisation scheme (mirrored exactly by gtcrn_ref.py and by the C runtime)
==========================================================================
Weight-and-activation int8, but only where it is safe:

  int8, per-output-row scale   all matmul-shaped weights: the (1,5) convs, every
                               1x1 point conv, every GRU weight_ih/weight_hh,
                               every Linear. ~120k of the ~125k parameters.
  float32                      depthwise 3x3 kernels (288 params each -- 9 MACs
                               per output, int8 buys nothing and costs accuracy),
                               all biases, PReLU alphas, LayerNorm affine, and
                               the ERB filterbank.

Activations are quantised *dynamically*: each time a quantised matmul runs, the
input vector's own max-abs sets the scale for that frame. No calibration set, no
calibration drift, and no risk of an unseen loud frame clipping into a scale
chosen on quiet data. This is the same scheme torch.quantization.quantize_dynamic
uses for GRUs, which is exactly where the accuracy risk lives.

Everything that is numerically delicate stays in float32 on device: the
FeatureFront (level tracking, minimum statistics, log), the sigmoid mask, the
phase normalisation, the deep filter, the STFT/ISTFT and the overlap-add.

BatchNorm is folded into the preceding convolution here, so the runtime never
sees a BN layer.

Usage
-----
    python export_int8.py --checkpoint checkpoints_v1/best_model.pt \
                          --out ../firmware/main/gtcrn_int8.bin

    # export the non-EMA weights instead
    python export_int8.py --checkpoint ... --weights raw

Run tools/verify_export.py afterwards. Do not flash a blob you have not verified.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"GTCRNQ8\x00"
VERSION = 1
NAME_LEN = 48
DIR_ENTRY = 88          # bytes per directory entry, must match gtcrn_blob.h
HEADER_BYTES = 176      # 32 header + 132 config + pad, must match gtcrn_blob.h

DT_F32, DT_I8, DT_I32 = 0, 1, 2


# ---------------------------------------------------------------------------
# quantisation helpers -- the C runtime and gtcrn_ref.py use identical maths
# ---------------------------------------------------------------------------
def round_half_away(x):
    """Match C's  (int)(v + (v>=0 ? 0.5f : -0.5f)).  np.round() is banker's
    rounding and would disagree with the device on exact .5 ties."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quant_rows_i8(w):
    """w: (rows, k) float -> (int8 (rows,k), float32 scales (rows,)).
    Symmetric, per output row, 127 levels (never -128, so negation is safe)."""
    w = np.asarray(w, dtype=np.float64)
    amax = np.max(np.abs(w), axis=1)
    scale = np.where(amax < 1e-12, 1.0, amax / 127.0)
    q = round_half_away(w / scale[:, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


# ---------------------------------------------------------------------------
# layer surgery
# ---------------------------------------------------------------------------
def fold_bn_conv(w, b, bn_w, bn_b, bn_mean, bn_var, eps):
    """Standard fold for Conv2d weights (out, in/g, kH, kW)."""
    s = bn_w / np.sqrt(bn_var + eps)
    w = w * s[:, None, None, None]
    b = (b - bn_mean) * s + bn_b
    return w, b


def deconv_to_conv(w, groups):
    """ConvTranspose2d stores (in, out/groups, kH, kW).  Re-index to the Conv2d
    convention (out, in/groups, kH, kW) so one C kernel serves both."""
    cin, out_pg, kh, kw = w.shape
    in_pg = cin // groups
    out = out_pg * groups
    c = np.zeros((out, in_pg, kh, kw), dtype=w.dtype)
    for g in range(groups):
        for j in range(out_pg):
            for i in range(in_pg):
                c[g * out_pg + j, i] = w[g * in_pg + i, j]
    return c


def fold_bn_deconv(w, b, bn_w, bn_b, bn_mean, bn_var, eps, groups):
    """Same fold, but the output-channel axis of a ConvTranspose weight is dim 1
    (and is offset per group), so convert first and fold after."""
    c = deconv_to_conv(w, groups)
    return fold_bn_conv(c, b, bn_w, bn_b, bn_mean, bn_var, eps)


def perm_conv_okci(w):
    """(out, in/g, 1, kW) -> flat [out][kW][in/g].

    The C kernels walk input channels in the innermost loop so the int8 dot
    product is over contiguous memory; that means the frequency-tap index has
    to sit outside it."""
    out, inpg, kh, kw = w.shape
    assert kh == 1, f"expected a (1,k) kernel, got {(kh, kw)}"
    return np.ascontiguousarray(w[:, :, 0, :].transpose(0, 2, 1)).reshape(out, kw * inpg)


# ---------------------------------------------------------------------------
# blob writer
# ---------------------------------------------------------------------------
class Blob:
    def __init__(self):
        self.entries = []          # (name, dtype, dims, payload, scales|None)

    def _add(self, name, dtype, arr, scales=None):
        assert len(name) < NAME_LEN, name
        assert not any(e[0] == name for e in self.entries), f"duplicate tensor {name}"
        dims = list(arr.shape) + [0] * (4 - arr.ndim)
        assert arr.ndim <= 4, name
        self.entries.append((name, dtype, dims, np.ascontiguousarray(arr), scales))

    def f32(self, name, arr):
        self._add(name, DT_F32, np.asarray(arr, dtype=np.float32))

    def i32(self, name, arr):
        self._add(name, DT_I32, np.asarray(arr, dtype=np.int32))

    def i8_rows(self, name, w2d):
        """Quantise a (rows, k) float matrix per row and store q + scales."""
        q, s = quant_rows_i8(w2d)
        self._add(name, DT_I8, q, s)

    def write(self, path, cfg):
        blocks, data, off = [], bytearray(), 0

        def push(buf):
            nonlocal off
            while off % 16:
                data.append(0)
                off += 1
            start = off
            data.extend(buf)
            off += len(buf)
            return start, len(buf)

        for name, dtype, dims, arr, scales in self.entries:
            d_off, d_len = push(arr.tobytes())
            if scales is None:
                s_off, n_s = 0, 0
            else:
                s_off, _ = push(np.asarray(scales, dtype=np.float32).tobytes())
                n_s = len(scales)
            blocks.append((name, dtype, arr.ndim, dims, d_off, d_len, s_off, n_s))

        directory = bytearray()
        for name, dtype, ndim, dims, d_off, d_len, s_off, n_s in blocks:
            e = bytearray()
            e += name.encode("ascii").ljust(NAME_LEN, b"\x00")
            e += struct.pack("<BBH", dtype, ndim, 0)
            e += struct.pack("<4I", *dims)
            e += struct.pack("<4I", d_off, d_len, s_off, n_s)
            e += struct.pack("<I", 0)
            assert len(e) == DIR_ENTRY, len(e)
            directory += e

        head = bytearray()
        head += MAGIC
        head += struct.pack("<I", VERSION)
        head += struct.pack("<I", len(blocks))
        head += struct.pack("<I", HEADER_BYTES)                      # dir_off
        dir_bytes = len(directory)
        data_off = HEADER_BYTES + dir_bytes
        data_off += (-data_off) % 16
        head += struct.pack("<I", data_off)
        head += struct.pack("<I", data_off + len(data))              # total size
        head += struct.pack("<I", 0)
        assert len(head) == 32

        c = cfg
        conf = struct.pack(
            "<12I", c["sample_rate"], c["n_fft"], c["hop_length"], c["n_freqs"],
            c["erb_subband_1"], c["erb_subband_2"], c["width"], c["bn_width"],
            c["base_channels"], c["n_dpgrnn"], c["tra_bands"], c["n_dil"])
        conf += struct.pack("<8i", *(list(c["dil"]) + [0] * (8 - len(c["dil"]))))
        conf += struct.pack("<3I", c["df_order"], c["df_bins"], c["n_out"])
        conf += struct.pack("<3f", c["mask_max"], c["mask_min"], c["compress"])
        conf += struct.pack("<3I", c["level_frames"], c["smooth_frames"], c["nf_frames"])
        conf += struct.pack("<4f", c["nf_bias"], c["snr_lo"], c["snr_hi"], c["mix_rms"])
        assert len(conf) == 132, len(conf)

        blob = bytearray()
        blob += head + conf
        blob += b"\x00" * (HEADER_BYTES - len(blob))
        blob += directory
        blob += b"\x00" * (data_off - len(blob))
        blob += data

        Path(path).write_bytes(bytes(blob))
        return len(blob)


# ---------------------------------------------------------------------------
# per-module exporters
# ---------------------------------------------------------------------------
def np_(t):
    return t.detach().cpu().float().numpy().astype(np.float64)


def export_convblock(blob, prefix, cb, deconv, groups, with_act=True):
    """ConvBlock = conv/deconv -> BatchNorm -> PReLU, all folded into one op."""
    w, b = np_(cb.conv.weight), np_(cb.conv.bias)
    bn = cb.bn
    if deconv:
        w, b = fold_bn_deconv(w, b, np_(bn.weight), np_(bn.bias),
                              np_(bn.running_mean), np_(bn.running_var), bn.eps, groups)
    else:
        w, b = fold_bn_conv(w, b, np_(bn.weight), np_(bn.bias),
                            np_(bn.running_mean), np_(bn.running_var), bn.eps)
    blob.i8_rows(prefix + "w", perm_conv_okci(w))
    blob.f32(prefix + "b", b)
    if with_act:
        blob.f32(prefix + "a", np_(cb.act.weight).reshape(-1))


def export_gru(blob, prefix, gru, suffix=""):
    """One direction of an nn.GRU. PyTorch gate order is [r, z, n]; the runtime
    keeps that order, so no reshuffling here."""
    blob.i8_rows(prefix + "wih", np_(getattr(gru, "weight_ih_l0" + suffix)))
    blob.i8_rows(prefix + "whh", np_(getattr(gru, "weight_hh_l0" + suffix)))
    blob.f32(prefix + "bih", np_(getattr(gru, "bias_ih_l0" + suffix)))
    blob.f32(prefix + "bhh", np_(getattr(gru, "bias_hh_l0" + suffix)))


def export_linear(blob, prefix, lin):
    blob.i8_rows(prefix + "w", np_(lin.weight))
    blob.f32(prefix + "b", np_(lin.bias))


def export_gtconv(blob, prefix, blk, deconv):
    """GTConvBlock: SFE -> pw1+BN+PReLU -> causal dilated DW+BN+PReLU -> pw2+BN
    -> BandTRA.  For the decoder variant every conv is a ConvTranspose2d; with
    stride 1 and the padding this block uses, a transposed conv is exactly a
    convolution with both kernel axes flipped, so we flip here and the runtime
    has a single code path."""
    hid = blk.point_conv1.weight.shape[1] if deconv else blk.point_conv1.weight.shape[0]

    # ---- point_conv1 (1x1) ----
    w, b = np_(blk.point_conv1.weight), np_(blk.point_conv1.bias)
    bn = blk.point_bn1
    if deconv:
        w, b = fold_bn_deconv(w, b, np_(bn.weight), np_(bn.bias),
                              np_(bn.running_mean), np_(bn.running_var), bn.eps, 1)
    else:
        w, b = fold_bn_conv(w, b, np_(bn.weight), np_(bn.bias),
                            np_(bn.running_mean), np_(bn.running_var), bn.eps)
    blob.i8_rows(prefix + "pc1.w", w[:, :, 0, 0])
    blob.f32(prefix + "pc1.b", b)
    blob.f32(prefix + "pc1.a", np_(blk.point_act.weight).reshape(-1))

    # ---- depthwise 3x3, dilated along time, causal ----
    w, b = np_(blk.depth_conv.weight), np_(blk.depth_conv.bias)
    bn = blk.depth_bn
    if deconv:
        w, b = fold_bn_deconv(w, b, np_(bn.weight), np_(bn.bias),
                              np_(bn.running_mean), np_(bn.running_var), bn.eps, hid)
        w = w[:, :, ::-1, ::-1].copy()            # transposed conv == flipped conv
    else:
        w, b = fold_bn_conv(w, b, np_(bn.weight), np_(bn.bias),
                            np_(bn.running_mean), np_(bn.running_var), bn.eps)
    blob.f32(prefix + "dw.w", w[:, 0])             # (C,3,3) float, stays float
    blob.f32(prefix + "dw.b", b)
    blob.f32(prefix + "dw.a", np_(blk.depth_act.weight).reshape(-1))

    # ---- point_conv2 (1x1), no activation, BN folded in ----
    w, b = np_(blk.point_conv2.weight), np_(blk.point_conv2.bias)
    bn = blk.point_bn2
    if deconv:
        w, b = fold_bn_deconv(w, b, np_(bn.weight), np_(bn.bias),
                              np_(bn.running_mean), np_(bn.running_var), bn.eps, 1)
    else:
        w, b = fold_bn_conv(w, b, np_(bn.weight), np_(bn.bias),
                            np_(bn.running_mean), np_(bn.running_var), bn.eps)
    blob.i8_rows(prefix + "pc2.w", w[:, :, 0, 0])
    blob.f32(prefix + "pc2.b", b)

    # ---- BandTRA ----
    export_gru(blob, prefix + "tra.", blk.tra.att_gru)
    export_linear(blob, prefix + "tra.fc.", blk.tra.att_fc)


def export_dpgrnn(blob, prefix, dp):
    # intra: bidirectional over frequency, two grouped GRUs
    export_gru(blob, prefix + "ia1.f.", dp.intra_rnn.rnn1, "")
    export_gru(blob, prefix + "ia1.b.", dp.intra_rnn.rnn1, "_reverse")
    export_gru(blob, prefix + "ia2.f.", dp.intra_rnn.rnn2, "")
    export_gru(blob, prefix + "ia2.b.", dp.intra_rnn.rnn2, "_reverse")
    export_linear(blob, prefix + "ia.fc.", dp.intra_fc)
    blob.f32(prefix + "ia.ln.w", np_(dp.intra_ln.weight))
    blob.f32(prefix + "ia.ln.b", np_(dp.intra_ln.bias))

    # inter: unidirectional over time
    export_gru(blob, prefix + "ie1.", dp.inter_rnn.rnn1, "")
    export_gru(blob, prefix + "ie2.", dp.inter_rnn.rnn2, "")
    export_linear(blob, prefix + "ie.fc.", dp.inter_fc)
    blob.f32(prefix + "ie.ln.w", np_(dp.inter_ln.weight))
    blob.f32(prefix + "ie.ln.b", np_(dp.inter_ln.bias))


def export_erb_sparse(blob, erb):
    """The ERB matrices are triangular: each of the 64 bands touches one
    contiguous run of bins. Dense they are 2 x 192 x 64 floats (98 kB) and
    86k MACs per frame; stored as runs they are ~3 kB and ~800 MACs."""
    bm = np_(erb.erb_fc.weight)      # (bands, 192)
    bs = np_(erb.ierb_fc.weight)     # (192, bands)

    def runs(mat, tag):
        starts, lens, vals = [], [], []
        for r in range(mat.shape[0]):
            nz = np.nonzero(np.abs(mat[r]) > 1e-9)[0]
            if len(nz) == 0:
                starts.append(0); lens.append(0); continue
            a, b = int(nz[0]), int(nz[-1]) + 1
            starts.append(a); lens.append(b - a)
            vals.extend(mat[r, a:b].tolist())
        blob.i32(f"erb.{tag}.start", np.array(starts, dtype=np.int32))
        blob.i32(f"erb.{tag}.len", np.array(lens, dtype=np.int32))
        blob.f32(f"erb.{tag}.w", np.array(vals, dtype=np.float32))
        return len(vals)

    n1 = runs(bm, "bm")
    n2 = runs(bs, "bs")
    return n1, n2


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="gtcrn_int8.bin")
    p.add_argument("--repo", default=".",
                   help="folder containing model.py (the training repo)")
    p.add_argument("--weights", choices=["ema", "raw"], default="ema",
                   help="ema = model_state_dict (what infer.py uses)")
    p.add_argument("--mask_min", type=float, default=None,
                   help="bake a mask floor into the blob; leave unset to keep "
                        "the trained value (it is runtime-adjustable anyway)")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch                                                   # noqa: E402
    from model import build_model_from_config                      # noqa: E402

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = dict(ck.get("config", {}))
    if args.mask_min is not None:
        cfg["mask_min"] = args.mask_min
    model = build_model_from_config(cfg)
    key = "model_state_dict" if args.weights == "ema" else "raw_state_dict"
    model.load_state_dict(ck[key])
    model.eval()

    n_fft = model.n_fft
    n_freqs = n_fft // 2 + 1
    erb1 = model.erb.erb_subband_1
    erb2 = model.erb.erb_fc.weight.shape[0]
    width = erb1 + erb2
    bn_width = ((width + 4 - 5) // 2 + 1)
    bn_width = ((bn_width + 4 - 5) // 2 + 1)
    C = model.base_channels
    dil = list(model.dilations)

    meta = {
        "sample_rate": model.sample_rate, "n_fft": n_fft,
        "hop_length": int(cfg.get("hop_length", n_fft // 2)),
        "n_freqs": n_freqs, "erb_subband_1": erb1, "erb_subband_2": erb2,
        "width": width, "bn_width": bn_width, "base_channels": C,
        "n_dpgrnn": model.n_dpgrnn, "tra_bands": model.encoder.en_convs[2].tra.num_bands,
        "n_dil": len(dil), "dil": dil,
        "df_order": model.df_order, "df_bins": model.df_bins, "n_out": model.n_out,
        "mask_max": float(model.mask_max), "mask_min": float(model.mask_min),
        "compress": float(model.front.compress),
        "level_frames": model.front.level_frames,
        "smooth_frames": model.front.smooth_frames,
        "nf_frames": model.front.nf_frames,
        "nf_bias": float(model.front.nf_bias),
        "snr_lo": float(model.front.snr_lo), "snr_hi": float(model.front.snr_hi),
        "mix_rms": float(cfg.get("mix_rms", 0.1)),
    }

    blob = Blob()

    # ---- encoder ----
    export_convblock(blob, "enc.c1.", model.encoder.en_convs[0], deconv=False, groups=1)
    export_convblock(blob, "enc.c2.", model.encoder.en_convs[1], deconv=False, groups=2)
    for i in range(len(dil)):
        export_gtconv(blob, f"enc.g{i}.", model.encoder.en_convs[2 + i], deconv=False)

    # ---- bottleneck ----
    for k in range(model.n_dpgrnn):
        export_dpgrnn(blob, f"dp{k}.", model.dpgrnns[k])

    # ---- decoder (dilations are consumed in reverse) ----
    for i in range(len(dil)):
        export_gtconv(blob, f"dec.g{i}.", model.decoder.de_convs[i], deconv=True)
    export_convblock(blob, "dec.c1.", model.decoder.de_convs[len(dil)],
                     deconv=True, groups=2)
    hw = np_(model.decoder.head.weight)                 # (in, out, 1, 5)
    hb = np_(model.decoder.head.bias)
    blob.i8_rows("dec.head.w", perm_conv_okci(deconv_to_conv(hw, 1)))
    blob.f32("dec.head.b", hb)

    # ---- ERB ----
    n1, n2 = export_erb_sparse(blob, model.erb)

    size = blob.write(args.out, meta)

    n_par = sum(q.numel() for q in model.parameters())
    n_i8 = sum(int(np.prod(e[3].shape)) for e in blob.entries if e[1] == DT_I8)
    n_f32 = sum(int(np.prod(e[3].shape)) for e in blob.entries if e[1] == DT_F32)
    print(f"checkpoint : {args.checkpoint}  (epoch {ck.get('epoch')}, "
          f"val {ck.get('val_loss', float('nan')):.4f}, weights={args.weights})")
    print(f"model      : {n_par:,} params, C={C}, dpgrnn={model.n_dpgrnn}, "
          f"dil={dil}, DF={model.df_order}x{model.df_bins}")
    print(f"bottleneck : width {width} -> {bn_width}")
    print(f"int8       : {n_i8:,} values   float32: {n_f32:,} values")
    print(f"ERB runs   : bm {n1} nonzeros, bs {n2} nonzeros (dense would be "
          f"{erb2 * (n_freqs - erb1) * 2:,})")
    print(f"written    : {args.out}  ({size / 1024:.1f} kB, {len(blob.entries)} tensors)")

    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=2))
    print(f"config     : {args.out}.json")


if __name__ == "__main__":
    main()
