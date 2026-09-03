"""
Run a trained GTCRN checkpoint on a noisy audio file and save the enhanced result.

Usage (run from the project root, next to sample_data/):
    python infer.py --input sample_data/mixed_dataset/152.wav --output enhanced.wav

--checkpoint defaults to checkpoints/best_model.pt (train.py's default output
location), so you only need to pass it if you're pointing at a different
checkpoint, e.g. checkpoints/last_model.pt or a renamed/moved file.
"""
import argparse

import torch

from model import GTCRN
from train import load_audio, save_audio, STFTFrontEnd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                   help="defaults to checkpoints/best_model.pt (train.py's default output)")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--mask_floor_db", type=float, default=None,
                   help="optional inference-time override for the mask magnitude floor "
                        "(e.g. -9.0). Overrides whatever the checkpoint was trained with -- "
                        "handy for A/B testing suppression aggressiveness on an existing "
                        "checkpoint without retraining. Leave unset to use the checkpoint's "
                        "own setting (usually None).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    # tra_bands / mask_floor_db / base_channels / n_dpgrnn are newer fields -- .get() with
    # the old (paper-faithful) defaults keeps this working against checkpoints saved before
    # each of these updates
    tra_bands = cfg.get("tra_bands", 1)
    mask_floor_db = args.mask_floor_db if args.mask_floor_db is not None else cfg.get("mask_floor_db", None)
    base_channels = cfg.get("base_channels", 16)
    n_dpgrnn = cfg.get("n_dpgrnn", 2)
    dilations = tuple(cfg["dilations"]) if "dilations" in cfg else (1, 2, 5)

    model = GTCRN(sample_rate=cfg["sample_rate"], n_fft=cfg["n_fft"],
                   tra_bands=tra_bands, mask_floor_db=mask_floor_db,
                   base_channels=base_channels, n_dpgrnn=n_dpgrnn, dilations=dilations).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    stft = STFTFrontEnd(cfg["n_fft"], cfg["hop_length"], cfg["n_fft"], device=device)

    noisy_wav = load_audio(args.input, cfg["sample_rate"]).to(device)
    noisy_wav = noisy_wav.unsqueeze(0)  # add batch dim -> (1, samples)

    with torch.no_grad():
        noisy_spec = stft.stft(noisy_wav)
        pred_spec = model(noisy_spec)
        enhanced_wav = stft.istft(pred_spec, length=noisy_wav.shape[-1])

    save_audio(args.output, enhanced_wav.squeeze(0), cfg["sample_rate"])
    print(f"Saved enhanced audio to {args.output}")


if __name__ == "__main__":
    main()