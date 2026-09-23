#!/usr/bin/env python3
"""
gtcrn_ref.py -- frame-by-frame NumPy reference for the ESP32 runtime.

This file is the specification. Every operation here has a one-to-one
counterpart in firmware/components/gtcrn/*.c, in the same order, with the same
quantisation, the same state layout and the same buffer semantics. If the C and
this disagree, the C is wrong.

Its practical job is to let you answer "did quantisation hurt the audio?" on a
laptop, in seconds, before you spend an afternoon on a board. Run
verify_export.py, look at the SI-SDR delta, and only then flash.

It is deliberately written in a streaming style -- one STFT frame in, one frame
out, all history in explicit state objects -- even though NumPy would let you
batch the whole utterance. Batching would hide exactly the bugs this is meant to
catch (ring-buffer off-by-ones, GRU state that resets, causal padding at the
stream start).

    python gtcrn_ref.py --blob gtcrn_int8.bin --input noisy.wav --output clean.wav
"""
import argparse
import struct
from pathlib import Path

import numpy as np

MAGIC = b"GTCRNQ8\x00"
NAME_LEN, DIR_ENTRY, HEADER_BYTES = 48, 88, 176
DT_F32, DT_I8, DT_I32 = 0, 1, 2


# ===========================================================================
# blob reader
# ===========================================================================
class Blob:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        assert raw[:8] == MAGIC, "not a GTCRNQ8 blob"
        (ver, n_tensors, dir_off, data_off, total, _) = struct.unpack_from("<6I", raw, 8)
        assert ver == 1, ver
        c = struct.unpack_from("<12I", raw, 32)
        dil = struct.unpack_from("<8i", raw, 32 + 48)
        d2 = struct.unpack_from("<3I", raw, 32 + 80)
        f1 = struct.unpack_from("<3f", raw, 32 + 92)
        d3 = struct.unpack_from("<3I", raw, 32 + 104)
        f2 = struct.unpack_from("<4f", raw, 32 + 116)
        self.cfg = dict(
            sample_rate=c[0], n_fft=c[1], hop_length=c[2], n_freqs=c[3],
            erb1=c[4], erb2=c[5], width=c[6], bn_width=c[7], base_channels=c[8],
            n_dpgrnn=c[9], tra_bands=c[10], n_dil=c[11], dil=list(dil[:c[11]]),
            df_order=d2[0], df_bins=d2[1], n_out=d2[2],
            mask_max=f1[0], mask_min=f1[1], compress=f1[2],
            level_frames=d3[0], smooth_frames=d3[1], nf_frames=d3[2],
            nf_bias=f2[0], snr_lo=f2[1], snr_hi=f2[2], mix_rms=f2[3])

        self.t = {}
        for i in range(n_tensors):
            o = dir_off + i * DIR_ENTRY
            name = raw[o:o + NAME_LEN].split(b"\x00")[0].decode()
            dtype, ndim, _ = struct.unpack_from("<BBH", raw, o + NAME_LEN)
            dims = struct.unpack_from("<4I", raw, o + NAME_LEN + 4)[:ndim]
            d_off, d_len, s_off, n_s = struct.unpack_from("<4I", raw, o + NAME_LEN + 20)
            base = data_off + d_off
            if dtype == DT_F32:
                arr = np.frombuffer(raw, np.float32, d_len // 4, base).reshape(dims)
                self.t[name] = np.array(arr, dtype=np.float32)
            elif dtype == DT_I32:
                arr = np.frombuffer(raw, np.int32, d_len // 4, base).reshape(dims)
                self.t[name] = np.array(arr)
            else:
                q = np.frombuffer(raw, np.int8, d_len, base).reshape(dims)
                s = np.frombuffer(raw, np.float32, n_s, data_off + s_off)
                self.t[name] = (np.array(q, dtype=np.int32), np.array(s, dtype=np.float32))

    def __getitem__(self, k):
        return self.t[k]


# ===========================================================================
# quantised primitives -- identical maths to gtcrn_ops.c
# ===========================================================================
def _round_half_away(x):
    """C's (int)(v + (v>=0 ? 0.5f : -0.5f)). NumPy's round() is banker's
    rounding and disagrees on exact ties."""
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def quantize(x):
    """Dynamic symmetric per-tensor int8. Returns (int32 codes, scale).

    Deliberately float32 and multiply-by-reciprocal, because that is exactly
    what gtcrn_ops.c does. Doing it in float64, or dividing instead of
    multiplying, differs in the last bit and flips the occasional value that
    lands precisely on a .5 boundary -- which then propagates through a GRU and
    shows up as a 1 % spike on one frame. Matching the device's arithmetic here
    is what makes this file usable as a debugging oracle."""
    x = np.asarray(x, dtype=np.float32)
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    s = np.float32(1.0) if amax < 1e-12 else np.float32(amax) / np.float32(127.0)
    v = x * (np.float32(1.0) / s)
    q = np.clip(_round_half_away(v), -127, 127)
    return q.astype(np.int32), s


def qlinear(W, x, bias=None):
    """W = (int8 codes (out,in), per-row scales). x float (in,) or (n,in)."""
    wq, ws = W
    xq, sx = quantize(x)
    acc = xq @ wq.T
    y = acc.astype(np.float32) * (ws.astype(np.float32) * sx)
    if bias is not None:
        y = y + bias
    return y.astype(np.float32)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))).astype(np.float32)


def prelu(x, a):
    return np.where(x >= 0, x, a * x).astype(np.float32)


def gru_step(W, x, h):
    """PyTorch GRU, gate order [r, z, n]. W = dict of the four tensors."""
    H = h.shape[0]
    gi = qlinear(W["wih"], x, W["bih"])
    gh = qlinear(W["whh"], h, W["bhh"])
    r = sigmoid(gi[:H] + gh[:H])
    z = sigmoid(gi[H:2 * H] + gh[H:2 * H])
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    return ((1.0 - z) * n + z * h).astype(np.float32)


def layernorm(x, w, b, eps=1e-8):
    """Over the whole (F, C) plane jointly, as nn.LayerNorm((width, hidden))."""
    xd = x.astype(np.float64)
    mu = np.float32(xd.mean())
    var = np.float32(((xd - float(mu)) ** 2).mean())
    inv = np.float32(1.0) / np.sqrt(var + np.float32(eps))
    return ((x - mu) * inv * w + b).astype(np.float32)


def sfe3(x):
    """Subband feature extraction: channel c*3+j holds x[c, f+j-1], zero outside."""
    C, F = x.shape
    out = np.zeros((C * 3, F), dtype=np.float32)
    out[0::3, 1:] = x[:, :-1]
    out[1::3, :] = x
    out[2::3, :-1] = x[:, 1:]
    return out


def conv1x5_s2(x, W, bias, groups):
    """Conv2d((1,5), stride (1,2), padding (0,2)). x (Cin,Fin) -> (Cout,Fout)."""
    wq, ws = W
    Cin, Fin = x.shape
    Cout = wq.shape[0]
    Fout = (Fin + 4 - 5) // 2 + 1
    in_pg, out_pg = Cin // groups, Cout // groups
    xq, sx = quantize(x)
    xp = np.zeros((Fin + 4, Cin), dtype=np.int32)
    xp[2:2 + Fin] = xq.T
    y = np.empty((Cout, Fout), dtype=np.float32)
    for g in range(groups):
        sl = slice(g * in_pg, (g + 1) * in_pg)
        # patch[f] = [k0 ci..., k1 ci..., ...] matching the exported row layout
        patch = np.stack([xp[2 * f:2 * f + 5, sl].reshape(-1) for f in range(Fout)])
        w = wq[g * out_pg:(g + 1) * out_pg]
        acc = patch @ w.T                                     # (Fout, out_pg)
        sc = ws[g * out_pg:(g + 1) * out_pg].astype(np.float32) * sx
        y[g * out_pg:(g + 1) * out_pg] = (acc.astype(np.float32) * sc).T + \
            bias[g * out_pg:(g + 1) * out_pg, None]
    return y


def deconv1x5_s2(x, W, bias, groups):
    """ConvTranspose2d((1,5), stride (1,2), padding (0,2)).  Output position j
    only sees taps k with (j + 2 - k) even, so even bins get 3 taps and odd bins
    get 2 -- that asymmetry is the upsampling, and it has to be exact."""
    wq, ws = W
    Cin, Fin = x.shape
    Cout = wq.shape[0]
    Fout = 2 * Fin - 1
    in_pg, out_pg = Cin // groups, Cout // groups
    xq, sx = quantize(x)
    xt = xq.T                                                 # (Fin, Cin)
    y = np.empty((Cout, Fout), dtype=np.float32)
    for g in range(groups):
        sl = slice(g * in_pg, (g + 1) * in_pg)
        w = wq[g * out_pg:(g + 1) * out_pg].reshape(out_pg, 5, in_pg)
        sc = ws[g * out_pg:(g + 1) * out_pg].astype(np.float32) * sx
        for j in range(Fout):
            acc = np.zeros(out_pg, dtype=np.int64)
            for k in range(5):
                num = j + 2 - k
                if num % 2:
                    continue
                idx = num // 2
                if 0 <= idx < Fin:
                    acc += w[:, k, :] @ xt[idx, sl]
            y[g * out_pg:(g + 1) * out_pg, j] = acc.astype(np.float32) * sc + \
                bias[g * out_pg:(g + 1) * out_pg]
    return y


def conv1x1(x, W, bias):
    """(Cin,F) -> (Cout,F), one dynamic scale for the whole frame."""
    return qlinear(W, x.T, bias).T.astype(np.float32)


def depthwise33(taps, w, bias):
    """taps: list of 3 arrays (C,F) at times t-2d, t-d, t (already in that
    order for both encoder and decoder -- the exporter flipped the decoder
    kernels so this routine is shared). w: (C,3,3) float, freq padded with 0."""
    C, F = taps[0].shape
    y = np.zeros((C, F), dtype=np.float32)
    for kt in range(3):
        src = taps[kt]
        for kf in range(3):
            sh = kf - 1
            if sh == 0:
                y += w[:, kt, kf][:, None] * src
            elif sh < 0:
                y[:, 1:] += w[:, kt, kf][:, None] * src[:, :-1]
            else:
                y[:, :-1] += w[:, kt, kf][:, None] * src[:, 1:]
    return (y + bias[:, None]).astype(np.float32)


# ===========================================================================
# streaming state
# ===========================================================================
class Ring:
    """Fixed-length frame history with replicate-on-first-write, which is what
    F.pad(mode='replicate') does at the left edge of a sequence."""

    def __init__(self, n, shape):
        self.n = n
        self.buf = np.zeros((n,) + shape, dtype=np.float32)
        self.pos = 0
        self.primed = False

    def push(self, x, replicate_first=False):
        if not self.primed:
            self.primed = True
            if replicate_first:
                self.buf[:] = x
                self.pos = 0
                return
        self.buf[self.pos] = x
        self.pos = (self.pos + 1) % self.n

    def at(self, back):
        """back=0 is the most recently written frame."""
        return self.buf[(self.pos - 1 - back) % self.n]


class FeatureFront:
    def __init__(self, cfg):
        self.cfg = cfg
        F = cfg["n_freqs"]
        self.lvl = np.zeros(cfg["level_frames"], dtype=np.float32)
        self.smo = np.zeros((cfg["smooth_frames"], F), dtype=np.float32)
        self.nf = np.zeros((cfg["nf_frames"], F), dtype=np.float32)
        self.i_lvl = self.i_smo = self.i_nf = 0
        self.primed = False

    def __call__(self, mag, real, imag):
        c = self.cfg
        power = mag.astype(np.float64) ** 2
        bp = float(power.mean())

        if not self.primed:
            self.primed = True
            self.lvl[:] = bp
            self.smo[:] = power
            # nf ring is filled once p_smooth exists, below
            level = np.sqrt(max(bp, 1e-12))
            p_smooth = power.copy()
            self.nf[:] = p_smooth
        else:
            self.lvl[self.i_lvl] = bp
            self.i_lvl = (self.i_lvl + 1) % len(self.lvl)
            self.smo[self.i_smo] = power
            self.i_smo = (self.i_smo + 1) % len(self.smo)
            level = np.sqrt(max(float(self.lvl.mean()), 1e-12))
            p_smooth = self.smo.mean(axis=0)
            self.nf[self.i_nf] = p_smooth
            self.i_nf = (self.i_nf + 1) % len(self.nf)

        noise_floor = self.nf.min(axis=0) * c["nf_bias"]
        snr = np.log10(np.maximum(p_smooth, 1e-12) / np.maximum(noise_floor, 1e-12))
        snr = np.clip(snr * 0.5, c["snr_lo"], c["snr_hi"])

        mag_n = np.maximum(mag / level, 1e-6)
        comp = mag_n ** c["compress"]
        gain = comp / mag_n
        feat = np.stack([comp, (real / level) * gain, (imag / level) * gain, snr])
        return feat.astype(np.float32)


# ===========================================================================
# the network
# ===========================================================================
class GtcrnRef:
    def __init__(self, blob):
        self.b = blob
        self.c = blob.cfg
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self):
        c, C = self.c, self.c["base_channels"]
        W, Fb = self.c["bn_width"], self.c["n_freqs"]
        self.front = FeatureFront(c)
        n_blocks = c["n_dil"]
        self.enc_hist = [Ring(2 * d + 1, (C, W)) for d in c["dil"]]
        self.dec_hist = [Ring(2 * d + 1, (C, W)) for d in reversed(c["dil"])]
        self.enc_tra = [np.zeros((c["tra_bands"], C), np.float32) for _ in range(n_blocks)]
        self.dec_tra = [np.zeros((c["tra_bands"], C), np.float32) for _ in range(n_blocks)]
        self.inter_h = [[np.zeros(C // 2, np.float32), np.zeros(C // 2, np.float32)]
                        for _ in range(c["n_dpgrnn"] * W)]
        self.df_hist_r = np.zeros((max(c["df_order"], 1), c["df_bins"]), np.float32)
        self.df_hist_i = np.zeros((max(c["df_order"], 1), c["df_bins"]), np.float32)
        self.df_pos = 0

    def _gru(self, pfx):
        b = self.b
        return {"wih": b[pfx + "wih"], "whh": b[pfx + "whh"],
                "bih": b[pfx + "bih"], "bhh": b[pfx + "bhh"]}

    # -- pieces -----------------------------------------------------------
    def erb_bm(self, x):
        """(4, n_freqs) -> (4, width): low bins verbatim, high bins into bands."""
        c, b = self.c, self.b
        e1, e2 = c["erb1"], c["erb2"]
        st, ln, w = b["erb.bm.start"], b["erb.bm.len"], b["erb.bm.w"]
        out = np.zeros((x.shape[0], c["width"]), dtype=np.float32)
        out[:, :e1] = x[:, :e1]
        hi = x[:, e1:]
        o = 0
        for i in range(e2):
            n = int(ln[i])
            if n:
                out[:, e1 + i] = hi[:, int(st[i]):int(st[i]) + n] @ w[o:o + n]
            o += n
        return out

    def erb_bs(self, x):
        """(3, width) -> (3, n_freqs)."""
        c, b = self.c, self.b
        e1 = c["erb1"]
        st, ln, w = b["erb.bs.start"], b["erb.bs.len"], b["erb.bs.w"]
        out = np.zeros((x.shape[0], c["n_freqs"]), dtype=np.float32)
        out[:, :e1] = x[:, :e1]
        bands = x[:, e1:]
        o = 0
        for i in range(c["n_freqs"] - e1):
            n = int(ln[i])
            if n:
                out[:, e1 + i] = bands[:, int(st[i]):int(st[i]) + n] @ w[o:o + n]
            o += n
        return out

    def band_tra(self, pfx, x, state):
        c = self.c
        C, F = x.shape
        nb = c["tra_bands"]
        pad = (-F) % nb
        Fp, bw = F + pad, (F + pad) // nb
        xp = np.zeros((C, Fp), dtype=np.float32)
        xp[:, :F] = x
        zt = (xp.astype(np.float64) ** 2).reshape(C, nb, bw).mean(axis=2)
        gw = self._gru(pfx + "tra.")
        fcw, fcb = self.b[pfx + "tra.fc.w"], self.b[pfx + "tra.fc.b"]
        gains = np.empty((C, nb), dtype=np.float32)
        for bi in range(nb):
            state[bi] = gru_step(gw, zt[:, bi].astype(np.float32), state[bi])
            gains[:, bi] = sigmoid(qlinear(fcw, state[bi], fcb))
        at = np.repeat(gains, bw, axis=1)[:, :F]
        return (x * at).astype(np.float32)

    def gtconv(self, pfx, x, dilation, hist, tra_state):
        C = x.shape[0]
        half = C // 2
        x1, x2 = x[:half], x[half:]
        h = conv1x1(sfe3(x1), self.b[pfx + "pc1.w"], self.b[pfx + "pc1.b"])
        h = prelu(h, float(self.b[pfx + "pc1.a"][0]))

        hist.push(h)
        taps = [hist.at(2 * dilation), hist.at(dilation), h]
        h = depthwise33(taps, self.b[pfx + "dw.w"], self.b[pfx + "dw.b"])
        h = prelu(h, float(self.b[pfx + "dw.a"][0]))

        h = conv1x1(h, self.b[pfx + "pc2.w"], self.b[pfx + "pc2.b"])
        h = self.band_tra(pfx, h, tra_state)

        out = np.empty((C, x.shape[1]), dtype=np.float32)
        out[0::2] = h                                   # channel shuffle
        out[1::2] = x2
        return out

    def dpgrnn(self, k, x):
        c = self.c
        C, F = x.shape
        h2 = C // 2
        xt = x.T.copy()                                 # (F, C)
        pfx = f"dp{k}."

        # ---- intra: bidirectional over frequency, fresh state every frame ----
        Hb = C // 4                                     # 8 for C=32
        y = np.empty((F, C), dtype=np.float32)
        for tag, rnn in (("ia1.", 0), ("ia2.", 1)):
            sl = slice(rnn * h2, (rnn + 1) * h2)
            gf, gb = self._gru(pfx + tag + "f."), self._gru(pfx + tag + "b.")
            hf = np.zeros(Hb, np.float32)
            hb = np.zeros(Hb, np.float32)
            fwd = np.empty((F, Hb), np.float32)
            bwd = np.empty((F, Hb), np.float32)
            for f in range(F):
                hf = gru_step(gf, xt[f, sl], hf)
                fwd[f] = hf
            for f in range(F - 1, -1, -1):
                hb = gru_step(gb, xt[f, sl], hb)
                bwd[f] = hb
            y[:, rnn * h2:rnn * h2 + Hb] = fwd
            y[:, rnn * h2 + Hb:(rnn + 1) * h2] = bwd

        z = qlinear(self.b[pfx + "ia.fc.w"], y, self.b[pfx + "ia.fc.b"])
        z = layernorm(z, self.b[pfx + "ia.ln.w"], self.b[pfx + "ia.ln.b"])
        intra = (xt + z).astype(np.float32)

        # ---- inter: one GRU step per frequency, state carried across time ----
        y = np.empty((F, C), dtype=np.float32)
        g1, g2 = self._gru(pfx + "ie1."), self._gru(pfx + "ie2.")
        for f in range(F):
            st = self.inter_h[k * F + f]
            st[0] = gru_step(g1, intra[f, :h2], st[0])
            st[1] = gru_step(g2, intra[f, h2:], st[1])
            y[f, :h2] = st[0]
            y[f, h2:] = st[1]
        z = qlinear(self.b[pfx + "ie.fc.w"], y, self.b[pfx + "ie.fc.b"])
        z = layernorm(z, self.b[pfx + "ie.ln.w"], self.b[pfx + "ie.ln.b"])
        return (intra + z).T.astype(np.float32)

    # -- one frame --------------------------------------------------------
    def frame(self, re, im):
        c, b = self.c, self.b
        mag = np.sqrt(re.astype(np.float64) ** 2 + im.astype(np.float64) ** 2 + 1e-12)
        feat = self.front(mag.astype(np.float32), re, im)

        x = sfe3(self.erb_bm(feat))
        e = []
        x = prelu(conv1x5_s2(x, b["enc.c1.w"], b["enc.c1.b"], 1),
                  float(b["enc.c1.a"][0])); e.append(x)
        x = prelu(conv1x5_s2(x, b["enc.c2.w"], b["enc.c2.b"], 2),
                  float(b["enc.c2.a"][0])); e.append(x)
        for i, d in enumerate(c["dil"]):
            x = self.gtconv(f"enc.g{i}.", x, d, self.enc_hist[i], self.enc_tra[i])
            e.append(x)

        for k in range(c["n_dpgrnn"]):
            x = self.dpgrnn(k, x)

        n = len(e)
        for i, d in enumerate(reversed(c["dil"])):
            x = self.gtconv(f"dec.g{i}.", x + e[n - 1 - i], d,
                            self.dec_hist[i], self.dec_tra[i])
        x = prelu(deconv1x5_s2(x + e[1], b["dec.c1.w"], b["dec.c1.b"], 2),
                  float(b["dec.c1.a"][0]))
        out = deconv1x5_s2(x + e[0], b["dec.head.w"], b["dec.head.b"], 1)

        # ---- bounded, phase-decoupled mask ----
        m = self.erb_bs(out[:3])
        span = c["mask_max"] - c["mask_min"]
        m_mag = c["mask_min"] + span * sigmoid(m[0])
        pr, pi = 1.0 + m[1], m[2]
        pn = np.sqrt(pr * pr + pi * pi + 1e-8)
        mr, mi = m_mag * pr / pn, m_mag * pi / pn
        er = re * mr - im * mi
        ei = re * mi + im * mr

        # ---- deep filter residual on the low band ----
        K, Fd = c["df_order"], c["df_bins"]
        if K > 0 and Fd > 0:
            coef = out[3:, :Fd]
            cr, ci = coef[0::2], coef[1::2]
            self.df_hist_r[self.df_pos] = er[:Fd]
            self.df_hist_i[self.df_pos] = ei[:Fd]
            self.df_pos = (self.df_pos + 1) % K
            dr = np.zeros(Fd, np.float32)
            di = np.zeros(Fd, np.float32)
            for k in range(K):
                idx = (self.df_pos - 1 - k) % K
                lr, li = self.df_hist_r[idx], self.df_hist_i[idx]
                dr += cr[k] * lr - ci[k] * li
                di += cr[k] * li + ci[k] * lr
            er[:Fd] += dr
            ei[:Fd] += di
        return er.astype(np.float32), ei.astype(np.float32)


# ===========================================================================
# STFT / ISTFT -- sqrt-Hann, hop = n_fft/2, plain overlap-add
# ===========================================================================
def sqrt_hann(n):
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)      # periodic
    return np.sqrt(np.maximum(w, 0.0)).astype(np.float32)


def enhance_wav(model, wav, cfg):
    """Zero-pad the head by n_fft/2 (the device cannot reflect-pad the future).
    The analysis and synthesis windows multiply to a periodic Hann, which sums
    to exactly 1.0 at 50 % overlap, so no window normalisation is needed."""
    n_fft, hop = cfg["n_fft"], cfg["hop_length"]
    win = sqrt_hann(n_fft)
    pad = n_fft // 2
    x = np.concatenate([np.zeros(pad, np.float32), wav,
                        np.zeros(n_fft, np.float32)]).astype(np.float32)
    n_frames = 1 + (len(x) - n_fft) // hop
    out = np.zeros(len(x), np.float32)
    for t in range(n_frames):
        seg = x[t * hop:t * hop + n_fft] * win
        S = np.fft.rfft(seg)
        er, ei = model.frame(S.real.astype(np.float32), S.imag.astype(np.float32))
        y = np.fft.irfft(er + 1j * ei, n_fft).astype(np.float32)
        out[t * hop:t * hop + n_fft] += y * win
    return out[pad:pad + len(wav)]


def main():
    import soundfile as sf
    p = argparse.ArgumentParser()
    p.add_argument("--blob", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    blob = Blob(args.blob)
    cfg = blob.cfg
    wav, sr = sf.read(args.input, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == cfg["sample_rate"], f"expected {cfg['sample_rate']} Hz, got {sr}"

    model = GtcrnRef(blob)
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-12))
    scale = cfg["mix_rms"] / rms if rms > 1e-9 else 1.0
    y = enhance_wav(model, (wav * scale).astype(np.float32), cfg) / scale
    sf.write(args.output, y, sr)
    print(f"wrote {args.output}  ({len(y) / sr:.2f} s)")


if __name__ == "__main__":
    main()
