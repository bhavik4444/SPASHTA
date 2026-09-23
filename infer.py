"""
Inference for GTCRN-DF.

    python infer.py --checkpoint checkpoints/best_model.pt --input noisy.wav --output clean.wav
    python infer.py --checkpoint checkpoints/best_model.pt --input in_dir/ --output out_dir/

The model reconstructs itself from the config saved in the checkpoint, so you
do not have to remember which architecture flags you trained with.

Two knobs you can turn without retraining, because neither has learnable
parameters:

  --mask_min      floor on mask magnitude (0 = full suppression). A small value
                  such as 0.02 (-34 dB) leaves a faint noise bed between words.
                  Total silence in the pauses is objectively more suppression
                  but some listeners find it unnatural and it can make the
                  speech sound like it is cutting in and out. Try 0.0 first.
  --output_gain_db  a flat gain on the output. The model is trained to
                  reproduce the clean speech at the level it had inside the
                  mixture, which at -15 dB input SNR is genuinely quiet. If you
                  want a hotter output for a headset, add it here rather than
                  training the model to hallucinate energy.

Long files are processed in overlapping chunks so memory stays bounded. The
model is causal, so the chunk boundary handling only needs enough left context
to prime the recurrent state and the noise-floor tracker — hence --context_sec.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from model import build_model_from_config
from train import STFTFrontEnd

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def load_model(checkpoint, device, mask_min=None):
    ck = torch.load(checkpoint, map_location=device)
    cfg = ck.get("config", {})
    if mask_min is not None:
        cfg = dict(cfg, mask_min=mask_min)
    model = build_model_from_config(cfg).to(device).eval()
    model.load_state_dict(ck["model_state_dict"])
    return model, cfg, ck


def read_audio(path, target_sr):
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        data = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(data)), sr, target_sr).numpy()
    return np.ascontiguousarray(data, dtype=np.float32)


@torch.no_grad()
def enhance(model, stft, wav, device, chunk_sec=20.0, context_sec=3.0,
            sample_rate=16000, norm_rms=0.1):
    """Normalise to the training loudness, enhance, then restore the original
    level. The model's features are level-invariant by construction, so this is
    belt and braces — but it also guarantees the output sits at the same level
    as the input, which is what anyone downstream expects."""
    n = wav.shape[0]
    in_rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-12))
    scale = (norm_rms / in_rms) if in_rms > 1e-9 else 1.0
    x = wav * scale

    chunk = int(chunk_sec * sample_rate)
    ctx = int(context_sec * sample_rate)
    out = np.zeros(n, dtype=np.float32)

    pos = 0
    while pos < n:
        end = min(pos + chunk, n)
        start = max(0, pos - ctx)
        seg = torch.from_numpy(x[start:end]).unsqueeze(0).to(device)
        spec = stft.stft(seg)
        pred = model(spec).float()
        y = stft.istft(pred, length=seg.shape[-1]).squeeze(0).cpu().numpy()
        out[pos:end] = y[pos - start:]          # drop the priming context
        pos = end

    return (out / scale).astype(np.float32)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="a wav file or a folder of them")
    p.add_argument("--output", required=True, help="output file or folder")
    p.add_argument("--mask_min", type=float, default=None,
                   help="override the mask magnitude floor (see module docstring)")
    p.add_argument("--output_gain_db", type=float, default=0.0)
    p.add_argument("--chunk_sec", type=float, default=20.0)
    p.add_argument("--context_sec", type=float, default=3.0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, ck = load_model(args.checkpoint, device, args.mask_min)
    sr = cfg.get("sample_rate", 16000)
    stft = STFTFrontEnd(cfg.get("n_fft", 512), cfg.get("hop_length", 256)).to(device)

    n_par = sum(q.numel() for q in model.parameters())
    print(f"loaded {args.checkpoint} (epoch {ck.get('epoch')}, "
          f"val {ck.get('val_loss', float('nan')):.4f}) | {n_par:,} params | {sr} Hz")

    in_path = Path(args.input)
    out_path = Path(args.output)
    if in_path.is_dir():
        files = sorted(f for f in in_path.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        out_path.mkdir(parents=True, exist_ok=True)
        pairs = [(f, out_path / f.name) for f in files]
    else:
        if out_path.suffix == "":
            out_path.mkdir(parents=True, exist_ok=True)
            pairs = [(in_path, out_path / in_path.name)]
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pairs = [(in_path, out_path)]

    gain = 10.0 ** (args.output_gain_db / 20.0)
    for src, dst in pairs:
        wav = read_audio(src, sr)
        y = enhance(model, stft, wav, device, args.chunk_sec, args.context_sec,
                    sr, cfg.get("mix_rms", 0.1)) * gain
        peak = float(np.max(np.abs(y))) + 1e-12
        if peak > 0.999:                        # only ever attenuates
            y = y * (0.999 / peak)
        sf.write(str(dst), y, sr)
        print(f"  {src.name} -> {dst}")

    print("done")


if __name__ == "__main__":
    main()