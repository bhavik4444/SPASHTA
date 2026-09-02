"""
GTCRN (Grouped Temporal Convolutional Recurrent Network) — speech enhancement model.

Adapted from the official implementation:
    Xiaobin Rong et al., "GTCRN: A Speech Enhancement Model Requiring Ultralow
    Computational Resources", ICASSP 2024.
    Official repo: https://github.com/Xiaobin-Rong/gtcrn  (MIT License)

Changes made for this project:
    - `sample_rate` and `n_fft` are now constructor arguments instead of being
      hard-coded to 16000 Hz / 512 inside the ERB module. This lets you train
      at whatever sampling rate your dataset actually uses.
    - `high_lim` (top edge of the ERB filterbank) auto-clamps to the Nyquist
      frequency of whatever sample_rate you pass in, so it never breaks.

The model itself (ERB filterbank, ShuffleNetV2-style grouped conv blocks,
temporal recurrent attention, dual-path grouped RNN) is otherwise unchanged
from the paper.
"""
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class ERB(nn.Module):
    """Equivalent Rectangular Bandwidth filterbank: compresses the linear
    frequency axis produced by the STFT into fewer, perceptually-spaced bands
    (and back again on the way out)."""

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
        bins = np.clip(bins, 0, nfft // 2)  # guard against rounding past Nyquist bin
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
        """Band split (inverse of bm). x_erb: (B,C,T,erb_subband_1+erb_subband_2)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction: gives each frequency bin a small window of
    its neighbours so 1x1 convs can see local frequency context."""

    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride),
                                 padding=(0, (kernel_size - 1) // 2))

    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1] * self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention — learns a per-frame, per-channel gain
    from the energy trajectory over time. This is what helps the model react
    fast to sudden (impulsive) noise bursts like gunfire.

    NOTE: this gate is broadcast across the ENTIRE frequency axis (see
    `At = at[..., None]` below) -- a burst that spikes the frame's average
    energy pulls the gain down for every frequency bin in that frame,
    including bins where speech (not the burst) dominates. Kept here for
    reference / A-B testing; the model now uses BandTRA by default, which
    fixes this by gating each frequency band independently. See BandTRA
    for the frequency-selective replacement.
    """

    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)
        return x * At


class BandTRA(nn.Module):
    """Frequency-aware Temporal Recurrent Attention.

    Same idea as TRA (a GRU tracks the energy trajectory over time and
    produces a sigmoid gain gate), but instead of averaging energy over the
    WHOLE frequency axis and broadcasting one gate to every bin, the
    frequency axis is split into `num_bands` contiguous chunks and each
    chunk gets its own independent gate. This is the key fix for the
    "gunfire suppresses speech" problem: a burst that's concentrated in the
    upper bands can now be suppressed there without dragging down the gain
    in lower bands where speech formants live in the same time frame.

    num_bands=1 recovers the exact original TRA behaviour.

    Weight-sharing note: the SAME GRU is applied to every band independently
    (bands are folded into the batch dimension, not concatenated into the
    feature dimension). This keeps parameter count essentially identical to
    the original TRA regardless of num_bands -- you get frequency
    selectivity from each band getting its own hidden state / energy
    trajectory, not from learning separate per-band weights, which keeps
    this cheap enough to fit GTCRN's low-compute design goal.
    """

    def __init__(self, channels, num_bands=4):
        super().__init__()
        self.num_bands = num_bands
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        B, C, T, F = x.shape
        pad = (-F) % self.num_bands
        xp = nn.functional.pad(x, [0, pad]) if pad else x
        Fp = xp.shape[-1]
        band_w = Fp // self.num_bands

        xb = xp.view(B, C, T, self.num_bands, band_w)
        zt = xb.pow(2).mean(dim=-1)                                        # (B,C,T,bands)
        # fold (batch, band) together so every band runs through the SAME
        # shared GRU independently, instead of growing the GRU's own size
        zt = zt.permute(0, 3, 2, 1).reshape(B * self.num_bands, T, C)      # (B*bands,T,C)

        at = self.att_gru(zt)[0]
        at = self.att_act(self.att_fc(at))                                 # (B*bands,T,C)
        at = at.reshape(B, self.num_bands, T, C).permute(0, 3, 2, 1)       # (B,C,T,bands)
        at = at.unsqueeze(-1).expand(-1, -1, -1, -1, band_w).reshape(B, C, T, Fp)
        if pad:
            at = at[..., :F]
        return x * at


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Grouped Temporal Convolution block (ShuffleNetV2-style split + shuffle)."""

    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False,
                 tra_bands=4):
        super().__init__()
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.sfe = SFE(kernel_size=3, stride=1)

        self.point_conv1 = conv_module(in_channels // 2 * 3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels // 2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels // 2)

        # tra_bands=1 recovers the original (frequency-broadcast) TRA behaviour;
        # tra_bands>1 gates each frequency chunk independently -- see BandTRA docstring.
        self.tra = BandTRA(in_channels // 2, num_bands=tra_bands)

    def shuffle(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        x = rearrange(x, 'b c g t f -> b (c g) t f')
        return x

    def forward(self, x):
        """x: (B,C,T,F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))
        h1 = self.tra(h1)

        return self.shuffle(h1, x2)


class GRNN(nn.Module):
    """Grouped RNN: splits channels in half and runs two independent GRUs
    (cheaper than one big GRU over all channels)."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        if h is None:
            n_dir = 2 if self.bidirectional else 1
            h = torch.zeros(self.num_layers * n_dir, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        return torch.cat([y1, y2], dim=-1), torch.cat([h1, h2], dim=-1)


class DPGRNN(nn.Module):
    """Dual-path grouped RNN: one RNN sweeps across frequency (intra), the
    other across time (inter), letting the model model both axes cheaply."""

    def __init__(self, input_size, width, hidden_size):
        super().__init__()
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
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
    """
    channels: width (C) of every conv/GTConvBlock in the stack. Original
        paper value is 16; must be a multiple of 4 (the grouped-RNN /
        channel-split design chunks channels in half twice over).
    dilations: one causal GTConvBlock per entry, in this order. Each block's
        causal receptive field is (kernel_size-1)*dilation = 2*dilation
        frames, and they stack additively (dilated causal TCN). The default
        (1,2,4,8,16) gives ~62 frames = ~992ms of PAST-only context at the
        model's default 16kHz/hop256 STFT settings -- enough for the model
        to "remember" what speech sounded like right before a brief noise
        burst (e.g. gunfire) and use that to keep separating speech from
        noise during and just after it, instead of only reacting to the
        current frame's energy. Purely causal (padding is prepended, never
        appended) so this adds no extra algorithmic latency, only memory/
        compute -- the original 3-block (1,2,5) schedule (~256ms) is the
        paper's version, kept reachable by passing dilations=(1,2,5).
    """

    def __init__(self, tra_bands=4, channels=16, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        assert channels % 4 == 0, "channels must be a multiple of 4"
        blocks = [
            ConvBlock(3 * 3, channels, (1, 5), stride=(1, 2), padding=(0, 2)),
            ConvBlock(channels, channels, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
        ]
        for d in dilations:
            blocks.append(GTConvBlock(channels, channels, (3, 3), stride=(1, 1), padding=(0, 1),
                                       dilation=(d, 1), tra_bands=tra_bands))
        self.en_convs = nn.ModuleList(blocks)

    def forward(self, x):
        en_outs = []
        for conv in self.en_convs:
            x = conv(x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    """Mirrors Encoder: same channels, dilations applied in reverse order.
    padding=2*dilation for each deconv GTConvBlock is the pattern the
    original architecture uses to keep the transposed conv's output the
    same length as its input; verified empirically here for every dilation
    used (including the new 8 and 16)."""

    def __init__(self, tra_bands=4, channels=16, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        assert channels % 4 == 0, "channels must be a multiple of 4"
        blocks = []
        for d in reversed(dilations):
            blocks.append(GTConvBlock(channels, channels, (3, 3), stride=(1, 1), padding=(2 * d, 1),
                                       dilation=(d, 1), use_deconv=True, tra_bands=tra_bands))
        blocks += [
            ConvBlock(channels, channels, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True),
            ConvBlock(channels, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True),
        ]
        self.de_convs = nn.ModuleList(blocks)

    def forward(self, x, en_outs):
        n = len(self.de_convs)
        for i in range(n):
            x = self.de_convs[i](x + en_outs[n - 1 - i])
        return x


class Mask(nn.Module):
    """Applies the predicted complex ratio mask to the noisy spectrogram.

    floor_db: optional inference-time knob (leave None for training). If
    set, clamps the mask magnitude to never go below `floor_db` (e.g. -9.0),
    which puts a hard ceiling on how much any bin can be suppressed --
    a cheap safety net against over-suppression that needs no retraining.
    No learnable parameters either way, so it's safe to flip on/off against
    an existing checkpoint.
    """

    def __init__(self, floor_db=None):
        super().__init__()
        self.floor = None if floor_db is None else 10 ** (floor_db / 20)

    def forward(self, mask, spec):
        if self.floor is not None:
            mag = torch.sqrt(mask[:, 0] ** 2 + mask[:, 1] ** 2 + 1e-12)
            scale = torch.clamp(self.floor / mag, min=1.0)  # only ever boosts bins below the floor
            mask = torch.stack([mask[:, 0] * scale, mask[:, 1] * scale], dim=1)
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        return torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)


class GTCRN(nn.Module):
    """
    Args:
        sample_rate: audio sampling rate in Hz your dataset actually uses
                     (e.g. 8000, 16000, 24000...). NOT hard-coded — this is
                     the parameter the ERB filterbank is built from.
        n_fft:       FFT size used for the STFT that produces the spectrogram
                     fed into this model. Must satisfy n_fft // 2 + 1 > 65
                     (so n_fft >= 256 in practice).
        erb_subband_1 / erb_subband_2: number of raw low-frequency bins kept
                     as-is, and number of compressed high-frequency ERB bands.
                     Leave at the paper's defaults unless you know you want
                     to retune the filterbank resolution.
        high_lim:    top frequency (Hz) covered by the ERB filterbank.
                     Automatically clamped to sample_rate/2 (Nyquist) if you
                     pass something too high for a low sample rate.
        tra_bands:   number of frequency bands each GTConvBlock's attention
                     gate is computed over independently (see BandTRA).
                     1 recovers the original TRA behaviour (one gate shared
                     across the whole spectrum -- prone to letting loud
                     broadband transients like gunfire suppress speech in
                     unrelated frequency bands within the same frame). 4 is
                     a reasonable default; higher = more frequency
                     selectivity, and it's free (weight-shared GRU, so this
                     doesn't change the parameter count at all).
        mask_floor_db: optional inference-time-only mask magnitude floor in
                     dB (e.g. -9.0). Leave None for training. See Mask.

        base_channels: width (C) of every conv/GTConvBlock and of the
                     DPGRNN stages. Original paper value is 16 (~24k
                     trainable params); must stay a multiple of 4. This is
                     the main lever for giving the model more capacity to
                     represent complex spectral shapes (distinguishing
                     speech harmonics from noise) -- roughly quadratic in
                     param cost, so raise it gradually and check params_.
        n_dpgrnn:    how many DPGRNN (dual-path grouped RNN) stages to
                     stack back-to-back at the network's bottleneck. Each
                     one alternately sweeps across frequency then time.
                     Original paper value is 2. Extra stages add temporal/
                     cross-frequency modeling depth exactly where the
                     network integrates the widest context, which is cheap
                     because it happens at the network's most downsampled
                     resolution.
        dilations:   see Encoder/Decoder. Extends the model's causal (past-
                     only) memory so it can use pre-burst speech content to
                     keep separating speech during/after a brief noise
                     burst, without adding latency.

        The paper-faithful configuration (~24k trainable params) is
        base_channels=16, n_dpgrnn=2, dilations=(1,2,5). The default below
        (base_channels=24, n_dpgrnn=3, dilations=(1,2,4,8,16)) lands at
        ~73k trainable / ~98k total params -- still tiny in absolute terms
        (RNNoise-scale), but with meaningfully more spectral capacity, more
        temporal/frequency integration depth, and ~4x the causal receptive
        field, aimed specifically at the "speech disappears under bursts"
        intelligibility failure mode rather than at raw parameter count.
    """

    def __init__(self, sample_rate=16000, n_fft=512, erb_subband_1=65, erb_subband_2=64, high_lim=8000,
                 tra_bands=4, mask_floor_db=None,
                 base_channels=24, n_dpgrnn=3, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        nfreqs = n_fft // 2 + 1
        assert nfreqs > erb_subband_1, (
            f"n_fft={n_fft} gives only {nfreqs} freq bins, which must be > erb_subband_1={erb_subband_1}. "
            f"Use n_fft >= 256."
        )
        assert base_channels % 4 == 0, "base_channels must be a multiple of 4"
        high_lim = min(high_lim, sample_rate // 2 - 1)

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.base_channels = base_channels
        self.n_dpgrnn = n_dpgrnn
        self.dilations = tuple(dilations)
        C = base_channels

        self.erb = ERB(erb_subband_1, erb_subband_2, nfft=n_fft, high_lim=high_lim, fs=sample_rate)
        self.sfe = SFE(3, 1)

        self.encoder = Encoder(tra_bands=tra_bands, channels=C, dilations=dilations)

        width = erb_subband_1 + erb_subband_2  # frequency width entering the encoder
        # after two stride-2 (kernel 5, pad 2) downsamples in the encoder:
        dpgrnn_width = ((width + 4 - 5) // 2 + 1)
        dpgrnn_width = ((dpgrnn_width + 4 - 5) // 2 + 1)
        self.dpgrnns = nn.ModuleList([DPGRNN(C, dpgrnn_width, C) for _ in range(n_dpgrnn)])

        self.decoder = Decoder(tra_bands=tra_bands, channels=C, dilations=dilations)
        self.mask = Mask(floor_db=mask_floor_db)

    def forward(self, spec):
        """
        spec: (B, F, T, 2) real/imag STFT, F = n_fft//2 + 1
        returns: enhanced spec, same shape as input
        """
        spec_ref = spec  # (B,F,T,2)

        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real ** 2 + spec_imag ** 2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,F)

        feat = self.erb.bm(feat)
        feat = self.sfe(feat)

        feat, en_outs = self.encoder(feat)
        for dpgrnn in self.dpgrnns:
            feat = dpgrnn(feat)

        m_feat = self.decoder(feat, en_outs)
        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))  # (B,2,T,F)
        return spec_enh.permute(0, 3, 2, 1)  # (B,F,T,2)


if __name__ == "__main__":
    # quick shape/sanity check at a couple of sample rates, and a size
    # comparison between the paper-faithful config and the new default
    for sr, nfft in [(16000, 512), (8000, 256)]:
        model = GTCRN(sample_rate=sr, n_fft=nfft).eval()
        n_params = sum(p.numel() for p in model.parameters())
        spec = torch.randn(2, nfft // 2 + 1, 63, 2)
        with torch.no_grad():
            out = model(spec)
        print(f"sample_rate={sr:>6} n_fft={nfft:>4} -> params={n_params:,} in={tuple(spec.shape)} out={tuple(out.shape)}")

    print()
    for label, kwargs in [
        ("paper-faithful (16/2/1,2,5)", dict(base_channels=16, n_dpgrnn=2, dilations=(1, 2, 5))),
        ("new default (24/3/1,2,4,8,16)", dict(base_channels=24, n_dpgrnn=3, dilations=(1, 2, 4, 8, 16))),
    ]:
        model = GTCRN(**kwargs).eval()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{label:32s} total={total:>7,}  trainable={trainable:>7,}")