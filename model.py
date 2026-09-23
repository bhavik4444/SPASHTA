"""
GTCRN-DF — low-complexity speech enhancement for adaptive noise cancellation.

Based on GTCRN (Xiaobin Rong et al., ICASSP 2024, MIT License,
https://github.com/Xiaobin-Rong/gtcrn) with four changes aimed specifically
at the low-SNR failure mode (speech becomes unintelligible when noise
dominates):

1. LEVEL-INVARIANT INPUT FEATURES (FeatureFront)
   The network no longer sees raw STFT magnitudes. It sees magnitudes divided
   by a causal running level estimate, plus an explicit per-bin
   "signal-to-noise-floor" feature computed by minimum statistics. At -15 dB
   input SNR the raw magnitude of a speech bin is meaningless in absolute
   terms; its ratio to the local stationary noise floor is not. This is the
   single most useful piece of information you can hand a masking network at
   low SNR, and previously the network had to infer it from scratch through
   BatchNorm.

2. THE STATIONARY PRE-FILTER IS GONE
   The old StationaryNoiseGate ran spectral subtraction BEFORE the network and
   permanently removed energy the network could never recover. That is
   survivable at +10 dB and destructive at -15 dB, where much of the speech
   sits at or below the tracked floor. The same noise-floor estimate is now
   fed in as a FEATURE instead of being subtracted, so the network gets all
   the information and loses none of the signal. (It was also a Python loop
   over frames — the vectorised min-statistics tracker here is far faster.)

3. BOUNDED, DECOUPLED MASK, INITIALISED TO IDENTITY
   The old head produced an unbounded complex mask through a Tanh. The new
   head predicts magnitude (sigmoid, bounded to [0, mask_max]) and phase
   rotation separately. At initialisation the mask is exactly 1.0 with zero
   phase rotation, so training starts from "pass the input through unchanged"
   rather than from a random mask. This matters a lot when strong suppression
   terms are in the loss: from a random start the cheapest early descent
   direction is "output near-silence", and models that fall into that basin
   produce exactly the muffled, unintelligible speech being reported.

4. DEEP FILTERING ON THE LOW BAND (DeepFilter)
   A per-bin multiplicative mask can only ever attenuate a T-F bin; when noise
   is 15 dB above speech in a bin, there is no gain that recovers the speech,
   because the bin's phase is wrong too. Deep filtering predicts a short
   complex FIR filter across the last `df_order` frames per bin, so the model
   can reconstruct a bin from its temporal neighbours instead of merely gating
   it. This is the main mechanism for actually recovering intelligibility (as
   opposed to just suppressing) below 0 dB. It is applied as a RESIDUAL,
   zero-initialised, over the lowest `df_bins` bins where F0 and the first two
   formants live.

Everything else (ERB filterbank, SFE, ShuffleNetV2-style grouped conv blocks,
band-wise temporal recurrent attention, dual-path grouped RNN) is structurally
the paper's, with more width and depth.

The model is fully causal: no frame ever sees the future, so the algorithmic
latency is one STFT frame regardless of the dilation schedule.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# ERB filterbank
# ===========================================================================
class ERB(nn.Module):
    """Compresses the linear STFT frequency axis into fewer, perceptually
    spaced bands (and back again). The lowest `erb_subband_1` bins are kept at
    full linear resolution — that is deliberate, and the deep filter below
    relies on it: those positions are still literal FFT bins, not bands."""

    def __init__(self, erb_subband_1, erb_subband_2, nfft, high_lim, fs):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    def erb2hz(self, erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft, high_lim, fs):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        bins = np.clip(bins, 0, nfft // 2)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
            / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) \
                / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) \
                / (bins[i + 2] - bins[i + 1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1] + 1] = 1 - erb_filters[-2, bins[-2]:bins[-1] + 1]
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """Band merge. x: (B,C,T,F) -> (B,C,T,erb_subband_1+erb_subband_2)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """Band split (inverse of bm)."""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


# ===========================================================================
# Input feature front-end
# ===========================================================================
class FeatureFront(nn.Module):
    """Turns a raw complex spectrogram into level-invariant network inputs.

    Produces 4 channels, all of which are unchanged if you scale the whole
    input signal by any constant:
        0: compressed normalised magnitude      (mag/level)**compress
        1: compressed normalised real part
        2: compressed normalised imaginary part
        3: local signal-to-noise-floor in dB/20 (minimum statistics)

    Channels 1 and 2 carry phase; channel 3 is the one that matters most at
    low SNR. All operators here are causal (left-padded pooling only) and
    vectorised — there is no per-frame Python loop anywhere.

    `level_frames` sets how far back the loudness estimate looks (default
    ~3 s). `nf_frames` sets the minimum-statistics window (default ~1.5 s):
    long enough that a speech utterance cannot pull the floor up with it,
    short enough to track a genuinely drifting background.
    """

    def __init__(self, compress=0.3, level_frames=192, smooth_frames=4,
                 nf_frames=96, nf_bias=1.6, snr_clamp=(-1.0, 2.5)):
        super().__init__()
        self.compress = compress
        self.level_frames = int(level_frames)
        self.smooth_frames = int(smooth_frames)
        self.nf_frames = int(nf_frames)
        self.nf_bias = nf_bias
        self.snr_lo, self.snr_hi = snr_clamp

    @staticmethod
    def _causal_avg(x, k):
        """x: (B,C,T) -> causal moving average of width k, same length."""
        if k <= 1:
            return x
        return F.avg_pool1d(F.pad(x, (k - 1, 0), mode="replicate"), k, stride=1)

    @staticmethod
    def _causal_min(x, k):
        """x: (B,C,T) -> causal running minimum of width k, same length."""
        if k <= 1:
            return x
        return -F.max_pool1d(F.pad(-x, (k - 1, 0), mode="replicate"), k, stride=1)

    def forward(self, mag, real, imag):
        """mag/real/imag: (B,F,T). Returns features (B,4,T,F) and the level
        estimate (B,1,T) in case a caller wants it."""
        power = mag.pow(2)

        # --- causal broadband level, used only to normalise the features ---
        band_power = power.mean(dim=1, keepdim=True)                  # (B,1,T)
        level = self._causal_avg(band_power, self.level_frames)
        level = level.clamp_min(1e-12).sqrt()                          # (B,1,T)

        # --- per-bin stationary noise floor by minimum statistics ---
        p_smooth = self._causal_avg(power, self.smooth_frames)         # (B,F,T)
        noise_floor = self._causal_min(p_smooth, self.nf_frames) * self.nf_bias

        snr = torch.log10(p_smooth.clamp_min(1e-12) / noise_floor.clamp_min(1e-12))
        snr = (snr * 10.0 / 20.0).clamp(self.snr_lo, self.snr_hi)      # dB/20

        # --- level-normalised, power-law compressed complex spectrum ---
        mag_n = (mag / level).clamp_min(1e-6)
        comp = mag_n.pow(self.compress)
        gain = comp / mag_n                                            # mag_n**(c-1)
        real_c = (real / level) * gain
        imag_c = (imag / level) * gain

        feat = torch.stack([comp, real_c, imag_c, snr], dim=1)         # (B,4,F,T)
        return feat.permute(0, 1, 3, 2).contiguous(), level            # (B,4,T,F)


# ===========================================================================
# Building blocks (structurally as in GTCRN)
# ===========================================================================
class SFE(nn.Module):
    """Subband Feature Extraction: hands each frequency bin a small window of
    its neighbours so 1x1 convs can see local frequency context."""

    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride),
                                padding=(0, (kernel_size - 1) // 2))

    def forward(self, x):
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1] * self.kernel_size,
                                    x.shape[2], x.shape[3])
        return xs


class BandTRA(nn.Module):
    """Band-wise Temporal Recurrent Attention.

    A GRU tracks each band's energy trajectory over time and emits a sigmoid
    gain. Splitting the frequency axis into `num_bands` independent chunks
    (rather than averaging over all of it, as the original TRA does) means a
    transient concentrated in the upper bands can be gated there without
    dragging down the gain in the bands carrying speech formants in the same
    frame. The GRU is shared across bands — bands are folded into the batch
    dimension — so num_bands costs no parameters at all.
    """

    def __init__(self, channels, num_bands=4):
        super().__init__()
        self.num_bands = num_bands
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        B, C, T, Fq = x.shape
        pad = (-Fq) % self.num_bands
        xp = F.pad(x, [0, pad]) if pad else x
        Fp = xp.shape[-1]
        band_w = Fp // self.num_bands

        xb = xp.view(B, C, T, self.num_bands, band_w)
        zt = xb.pow(2).mean(dim=-1)                                       # (B,C,T,bands)
        zt = zt.permute(0, 3, 2, 1).reshape(B * self.num_bands, T, C)

        at = self.att_gru(zt)[0]
        at = self.att_act(self.att_fc(at))
        at = at.reshape(B, self.num_bands, T, C).permute(0, 3, 2, 1)      # (B,C,T,bands)
        at = at.unsqueeze(-1).expand(-1, -1, -1, -1, band_w).reshape(B, C, T, Fp)
        if pad:
            at = at[..., :Fq]
        return x * at


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 groups=1, use_deconv=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride,
                                padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Grouped temporal convolution block (ShuffleNetV2-style split + shuffle),
    causal along time: the dilated conv is left-padded only."""

    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding,
                 dilation, use_deconv=False, tra_bands=4):
        super().__init__()
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.sfe = SFE(kernel_size=3, stride=1)

        self.point_conv1 = conv_module(in_channels // 2 * 3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                      stride=stride, padding=padding, dilation=dilation,
                                      groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels // 2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels // 2)

        self.tra = BandTRA(in_channels // 2, num_bands=tra_bands)

    @staticmethod
    def shuffle(x1, x2):
        B, C, T, Fq = x1.shape
        return torch.stack([x1, x2], dim=2).reshape(B, 2 * C, T, Fq)

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = F.pad(h1, [0, 0, self.pad_size, 0])          # causal: past only
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))
        h1 = self.tra(h1)

        return self.shuffle(h1, x2)


class GRNN(nn.Module):
    """Grouped RNN: channels split in half, two independent GRUs."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True,
                 bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        if h is None:
            n_dir = 2 if self.bidirectional else 1
            h = torch.zeros(self.num_layers * n_dir, x.shape[0], self.hidden_size,
                            device=x.device, dtype=x.dtype)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        return torch.cat([y1, y2], dim=-1), torch.cat([h1, h2], dim=-1)


class DPGRNN(nn.Module):
    """Dual-path grouped RNN. The intra pass sweeps across frequency and is
    bidirectional (frequency has no causality constraint); the inter pass
    sweeps across time and is strictly unidirectional."""

    def __init__(self, input_size, width, hidden_size):
        super().__init__()
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2,
                              bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size,
                              bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x):
        """x: (B,C,T,F)"""
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = self.intra_fc(self.intra_rnn(intra_x)[0])
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        x = intra_out.permute(0, 2, 1, 3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_fc(self.inter_rnn(inter_x)[0])
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)
        inter_x = inter_x.permute(0, 2, 1, 3)
        inter_x = self.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)

        return inter_out.permute(0, 3, 1, 2)  # (B,C,T,F)


class Encoder(nn.Module):
    def __init__(self, in_channels, tra_bands=4, channels=32, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        assert channels % 4 == 0, "channels must be a multiple of 4"
        blocks = [
            ConvBlock(in_channels, channels, (1, 5), stride=(1, 2), padding=(0, 2)),
            ConvBlock(channels, channels, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
        ]
        for d in dilations:
            blocks.append(GTConvBlock(channels, channels, (3, 3), stride=(1, 1),
                                      padding=(0, 1), dilation=(d, 1), tra_bands=tra_bands))
        self.en_convs = nn.ModuleList(blocks)

    def forward(self, x):
        en_outs = []
        for conv in self.en_convs:
            x = conv(x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    """Mirrors Encoder. The final layer is a plain linear transposed conv with
    no BatchNorm and no activation — the head applies the right nonlinearity to
    each output group itself (sigmoid for mask magnitude, none for the deep
    filter coefficients)."""

    def __init__(self, out_channels, tra_bands=4, channels=32,
                 dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        assert channels % 4 == 0, "channels must be a multiple of 4"
        blocks = []
        for d in reversed(dilations):
            blocks.append(GTConvBlock(channels, channels, (3, 3), stride=(1, 1),
                                      padding=(2 * d, 1), dilation=(d, 1),
                                      use_deconv=True, tra_bands=tra_bands))
        blocks.append(ConvBlock(channels, channels, (1, 5), stride=(1, 2),
                                padding=(0, 2), groups=2, use_deconv=True))
        self.head = nn.ConvTranspose2d(channels, out_channels, (1, 5),
                                       stride=(1, 2), padding=(0, 2))
        blocks.append(self.head)
        self.de_convs = nn.ModuleList(blocks)

    def forward(self, x, en_outs):
        n = len(self.de_convs)
        for i in range(n):
            x = self.de_convs[i](x + en_outs[n - 1 - i])
        return x


class DeepFilter(nn.Module):
    """Causal complex FIR filtering across time, per frequency bin.

    A multiplicative mask applies one complex number per T-F bin, so once noise
    exceeds speech in a bin there is no gain that recovers it. Deep filtering
    instead predicts `order` complex coefficients per bin and forms

        y[t,f] = sum_{k=0..order-1} c_k[t,f] * x[t-k,f]

    letting the model rebuild a bin from recent frames — which is how it can
    recover harmonic structure that a mask alone would have to throw away.
    Applied as a residual on top of the masked spectrum and zero-initialised,
    so it starts as a no-op and can only add refinement.
    """

    def __init__(self, order=5):
        super().__init__()
        self.order = order

    def forward(self, e_real, e_imag, coef):
        """e_real/e_imag: (B,T,Fd). coef: (B,2*order,T,Fd). Returns residuals."""
        K = self.order
        cr = coef[:, 0::2]                       # (B,K,T,Fd)
        ci = coef[:, 1::2]

        T = e_real.shape[1]
        pr = F.pad(e_real, (0, 0, K - 1, 0))     # pad the time axis on the left
        pi = F.pad(e_imag, (0, 0, K - 1, 0))

        acc_r = torch.zeros_like(e_real)
        acc_i = torch.zeros_like(e_imag)
        for k in range(K):
            s = K - 1 - k                        # k=0 is the current frame
            lr = pr[:, s:s + T]
            li = pi[:, s:s + T]
            acc_r = acc_r + cr[:, k] * lr - ci[:, k] * li
            acc_i = acc_i + cr[:, k] * li + ci[:, k] * lr
        return acc_r, acc_i


# ===========================================================================
# Full model
# ===========================================================================
class GTCRN(nn.Module):
    """
    Args
        sample_rate / n_fft: must match the STFT actually feeding this model.
        erb_subband_1: number of low bins kept at full linear resolution (65 by
            default = 0..2031 Hz at 16 kHz / 512). df_bins must not exceed it.
        erb_subband_2: number of compressed ERB bands above that.
        base_channels: network width. Must be a multiple of 4. The paper uses
            16; 32 is the default here because the low-SNR job (telling a
            speech harmonic apart from a noise partial at negative SNR) needs
            more spectral capacity than the +5 dB job does.
        n_dpgrnn: stacked dual-path RNN stages at the bottleneck. Cheap, since
            they run at the most downsampled resolution.
        dilations: one causal GTConvBlock per entry. (1,2,4,8,16) gives ~62
            frames (~1 s at 16 kHz / hop 256) of past-only context.
        tra_bands: independent attention bands per GTConvBlock (free).
        df_order / df_bins: deep filter taps, and how many low linear bins it
            covers. df_bins <= erb_subband_1.
        mask_max: ceiling on mask magnitude. sigmoid(0)*mask_max must equal 1
            for the identity initialisation to hold, so leave this at 2.0
            unless you also change the head init.
        mask_min: floor on mask magnitude. 0.0 during training. Setting a small
            value at inference (e.g. 0.02) leaves a little noise floor in,
            which some listeners prefer to absolute silence between words.
    """

    def __init__(self, sample_rate=16000, n_fft=512, erb_subband_1=65, erb_subband_2=64,
                 high_lim=8000, tra_bands=4, base_channels=32, n_dpgrnn=3,
                 dilations=(1, 2, 4, 8, 16), df_order=5, df_bins=64,
                 mask_max=2.0, mask_min=0.0, compress=0.3,
                 level_frames=192, nf_frames=96):
        super().__init__()
        nfreqs = n_fft // 2 + 1
        assert nfreqs > erb_subband_1, (
            f"n_fft={n_fft} gives {nfreqs} bins, must exceed erb_subband_1={erb_subband_1}")
        assert base_channels % 4 == 0, "base_channels must be a multiple of 4"
        assert df_bins <= erb_subband_1, (
            f"df_bins={df_bins} must be <= erb_subband_1={erb_subband_1}; above that the "
            f"decoder's frequency positions are ERB bands, not linear FFT bins")
        high_lim = min(high_lim, sample_rate // 2 - 1)

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.base_channels = base_channels
        self.n_dpgrnn = n_dpgrnn
        self.dilations = tuple(dilations)
        self.df_order = df_order
        self.df_bins = df_bins
        self.mask_max = mask_max
        self.mask_min = mask_min
        C = base_channels

        self.front = FeatureFront(compress=compress, level_frames=level_frames,
                                  nf_frames=nf_frames)
        self.n_feat = 4

        self.erb = ERB(erb_subband_1, erb_subband_2, nfft=n_fft, high_lim=high_lim,
                       fs=sample_rate)
        self.sfe = SFE(3, 1)

        self.encoder = Encoder(in_channels=self.n_feat * 3, tra_bands=tra_bands,
                               channels=C, dilations=dilations)

        width = erb_subband_1 + erb_subband_2
        dpgrnn_width = ((width + 4 - 5) // 2 + 1)
        dpgrnn_width = ((dpgrnn_width + 4 - 5) // 2 + 1)
        self.dpgrnns = nn.ModuleList([DPGRNN(C, dpgrnn_width, C) for _ in range(n_dpgrnn)])

        self.n_out = 3 + 2 * df_order          # mask: mag, phase_r, phase_i | then DF taps
        self.decoder = Decoder(out_channels=self.n_out, tra_bands=tra_bands,
                               channels=C, dilations=dilations)
        self.deep_filter = DeepFilter(order=df_order)

        self._init_head()

    def _init_head(self):
        """Start from identity. The mask channels get a tiny random init so the
        first backward pass still carries signal into the decoder body, and the
        deep filter coefficients start at exactly zero so DF contributes
        nothing until it has learned something useful."""
        head = self.decoder.head
        with torch.no_grad():
            nn.init.normal_(head.weight[:, :3], std=0.01)   # ConvTranspose: dim 1 is out_ch
            nn.init.zeros_(head.weight[:, 3:])
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, spec):
        """
        spec: (B, F, T, 2) real/imag STFT, F = n_fft//2 + 1
        returns: enhanced spec, same shape
        """
        real = spec[..., 0]
        imag = spec[..., 1]
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)     # (B,F,T)

        feat, _ = self.front(mag, real, imag)               # (B,4,T,F)

        x = self.erb.bm(feat)
        x = self.sfe(x)
        x, en_outs = self.encoder(x)
        for dpgrnn in self.dpgrnns:
            x = dpgrnn(x)
        out = self.decoder(x, en_outs)                      # (B,n_out,T,W)

        # ---- bounded, phase-decoupled mask over the full spectrum ----
        m = self.erb.bs(out[:, :3])                         # (B,3,T,F)
        span = self.mask_max - self.mask_min
        m_mag = self.mask_min + span * torch.sigmoid(m[:, 0])
        pr = 1.0 + m[:, 1]                                  # identity phase at init
        pi = m[:, 2]
        pnorm = torch.sqrt(pr * pr + pi * pi + 1e-8)
        mr = m_mag * pr / pnorm
        mi = m_mag * pi / pnorm

        sr = real.permute(0, 2, 1)                          # (B,T,F)
        si = imag.permute(0, 2, 1)
        e_r = sr * mr - si * mi
        e_i = sr * mi + si * mr

        # ---- deep filtering residual on the low band ----
        Fd = self.df_bins
        if Fd > 0 and self.df_order > 0:
            coef = out[:, 3:, :, :Fd]                       # (B,2K,T,Fd)
            dr, di = self.deep_filter(e_r[..., :Fd], e_i[..., :Fd], coef)
            e_r = torch.cat([e_r[..., :Fd] + dr, e_r[..., Fd:]], dim=-1)
            e_i = torch.cat([e_i[..., :Fd] + di, e_i[..., Fd:]], dim=-1)

        return torch.stack([e_r, e_i], dim=-1).permute(0, 2, 1, 3)   # (B,F,T,2)


def build_model_from_config(cfg):
    """Reconstruct a model from a checkpoint's saved config dict, ignoring any
    keys that are training-only (sample rates, hop length, loss weights)."""
    keys = ("sample_rate", "n_fft", "erb_subband_1", "erb_subband_2", "high_lim",
            "tra_bands", "base_channels", "n_dpgrnn", "dilations", "df_order",
            "df_bins", "mask_max", "mask_min", "compress", "level_frames", "nf_frames")
    kwargs = {k: cfg[k] for k in keys if k in cfg}
    if "dilations" in kwargs:
        kwargs["dilations"] = tuple(kwargs["dilations"])
    return GTCRN(**kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)

    for label, kw in [
        ("paper-ish  (16/2, no DF)", dict(base_channels=16, n_dpgrnn=2,
                                          dilations=(1, 2, 5), df_order=0, df_bins=0)),
        ("default    (32/3, DF-5)", dict()),
        ("large      (40/4, DF-5)", dict(base_channels=40, n_dpgrnn=4)),
    ]:
        m = GTCRN(**kw).eval()
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{label:28s} total={total:>9,}  trainable={trainable:>9,}")

    print()
    model = GTCRN().eval()
    spec = torch.randn(2, 257, 101, 2)
    with torch.no_grad():
        out = model(spec)
    print(f"shapes: in={tuple(spec.shape)} out={tuple(out.shape)}  match={out.shape == spec.shape}")

    # identity init: an untrained model should pass the input through almost
    # unchanged, which is the whole point of the head initialisation.
    err = (out - spec).abs().mean() / spec.abs().mean()
    print(f"relative deviation from identity at init: {err:.4f} (should be well under 0.1)")

    # causality: changing the last frame must not alter any earlier frame.
    spec2 = spec.clone()
    spec2[:, :, -1] += 10.0
    with torch.no_grad():
        out2 = model(spec2)
    print(f"causal (past frames unchanged): "
          f"{torch.allclose(out[:, :, :-1], out2[:, :, :-1], atol=1e-5)}")

    # scale invariance of the features: 20x louder input -> 20x louder output.
    with torch.no_grad():
        out3 = model(spec * 20.0)
    rel = ((out3 / 20.0) - out).abs().mean() / out.abs().mean()
    print(f"level invariance (20x input): relative error {rel:.4f} (should be small)")