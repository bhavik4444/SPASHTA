#!/usr/bin/env python3
"""
qat_finetune.py -- short quantisation-aware fine-tune, only if you need it.

Run verify_export.py first. If the int8 SI-SDR delta is already inside a few
tenths of a dB, skip this entirely -- a fine-tune that starts from a converged
model and a fresh optimiser can just as easily lose you something.

What it does: monkey-patches the model so that every tensor the ESP32 will
quantise is rounded to int8 in the forward pass, with a straight-through
estimator on the backward pass, and then hands control to your existing
train.py. Same data pipeline, same losses, same curriculum -- the only
difference is that the network now feels the rounding error while it learns, so
the weights drift into a configuration where that error does not matter.

Quantised here, exactly as on device:
  - every (1,5) conv and every 1x1 point conv: weights per output row, inputs
    per tensor
  - every nn.Linear: same
  - every GRU weight matrix, and the GRU input
Left in float here, exactly as on device:
  - the depthwise 3x3 kernels, all biases, PReLU, LayerNorm, ERB, the mask
    arithmetic, the deep filter

Usage -- pass through every flag you trained with, then override the schedule:

    python qat_finetune.py \
        --data_root sample_data --init_checkpoint checkpoints_v1/best_model.pt \
        --epochs 12 --steps_per_epoch 500 --lr 1.5e-4 --warmup_epochs 0 \
        --curriculum_epochs 0 --supp_ramp_epochs 1 \
        --snr_min -15 --snr_max 20 --hard_frac 0.45 --p_burst 0.55 \
        --base_channels 32 --n_dpgrnn 3 --tra_bands 8 --df_order 5 --df_bins 64 \
        --keep_weight 3.0 --supp_active 1.0 --supp_silent_start 8.0 \
        --supp_silent_end 8.0 --ema_decay 0.999 --out_dir checkpoints_qat

Note the schedule overrides: low LR, no warmup, no SNR curriculum, and the
suppression weight pinned at its FINAL value. You are polishing a trained model,
not training one, and re-running the curriculum would undo it.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(".").resolve()))


# ---------------------------------------------------------------------------
# fake quantisation, straight-through
# ---------------------------------------------------------------------------
def fq_tensor(x):
    """Dynamic symmetric per-tensor int8, as the runtime does for activations."""
    amax = x.detach().abs().amax().clamp_min(1e-12)
    s = amax / 127.0
    q = torch.clamp(torch.round(x / s), -127, 127) * s
    return x + (q - x).detach()


def fq_rows(w, ch_axis):
    """Per-output-channel int8. ch_axis is 0 for Conv2d/Linear/GRU weights and
    1 for ConvTranspose2d, whose output channels live on dim 1."""
    perm = list(range(w.dim()))
    perm[0], perm[ch_axis] = perm[ch_axis], perm[0]
    v = w.permute(*perm)
    amax = v.reshape(v.shape[0], -1).abs().amax(dim=1).clamp_min(1e-12)
    s = amax.view(-1, *([1] * (v.dim() - 1))) / 127.0
    q = torch.clamp(torch.round(v / s), -127, 127) * s
    q = q.permute(*perm)
    return w + (q - w).detach()


def is_quantised_conv(m):
    """Depthwise 3x3 stays float on device, so it stays float here."""
    if not isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return False
    kh, kw = m.kernel_size
    if kh == 3 and kw == 3:
        return False
    return True


def patch(model):
    n_conv = n_lin = n_gru = 0

    for m in model.modules():
        if is_quantised_conv(m):
            if isinstance(m, nn.Conv2d):
                def cfwd(x, _m=m):
                    return _m._conv_forward(fq_tensor(x), fq_rows(_m.weight, 0), _m.bias)
                m.forward = cfwd
            else:
                def tfwd(x, _m=m):
                    return nn.functional.conv_transpose2d(
                        fq_tensor(x), fq_rows(_m.weight, 1), _m.bias, _m.stride,
                        _m.padding, _m.output_padding, _m.groups, _m.dilation)
                m.forward = tfwd
            n_conv += 1

        elif isinstance(m, nn.Linear):
            of = m.forward

            def lfwd(x, _m=m):
                return nn.functional.linear(fq_tensor(x), fq_rows(_m.weight, 0), _m.bias)
            m.forward = lfwd
            n_lin += 1

        elif isinstance(m, nn.GRU):
            # nn.GRU reads its weights through the cached _flat_weights list, so
            # a parametrisation on the attributes would be silently ignored.
            # Rebuilding that list on every forward is the supported workaround.
            def pre(mod, inp):
                mod._flat_weights = [
                    fq_rows(getattr(mod, nm), 0) if nm.startswith("weight")
                    else getattr(mod, nm)
                    for nm in mod._flat_weights_names]
                return (fq_tensor(inp[0]),) + tuple(inp[1:])
            m.register_forward_pre_hook(pre)
            n_gru += 1

    print(f"[qat] fake-quantised {n_conv} convs, {n_lin} linears, {n_gru} GRUs")
    return model


def main():
    import train as train_mod

    if "--init_checkpoint" not in sys.argv:
        print("refusing to run: QAT must start from a trained checkpoint.\n"
              "pass --init_checkpoint checkpoints_v1/best_model.pt")
        sys.exit(2)

    from model import GTCRN
    orig_init = GTCRN.__init__

    def patched_init(self, *a, **k):
        orig_init(self, *a, **k)
        patch(self)
    GTCRN.__init__ = patched_init

    # cuDNN's fused GRU wants contiguous flattened weights; our per-forward
    # rebuild defeats that and prints a warning on every call. The fallback path
    # is correct, just slower, which is fine for a dozen epochs.
    torch.backends.cudnn.enabled = False
    import warnings
    warnings.filterwarnings("ignore", message=".*flatten_parameters.*")

    train_mod.main()


if __name__ == "__main__":
    main()
