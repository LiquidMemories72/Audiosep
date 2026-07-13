"""
infer_best.py  –  Load the best checkpoint and save separated audio for all mixtures.

Usage:
    python scripts/infer_best.py configs/tiny_overfit.yaml
"""

import os
import sys
import csv
import torch
import torchaudio

# Pre-import workaround for SpeechBrain / torch._dynamo / k2
import torch.optim  # noqa
try:
    import torch._dynamo  # noqa
except Exception:
    pass

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.logger import get_logger

from dexformer.models.dexformer import DExFormer
import glob
import json

logger = get_logger(__name__)


def find_best_ckpt_dir(save_dir):
    """Find the CKPT folder with the lowest OR-PIT_loss."""
    best_loss = float("inf")
    best_dir = None
    for meta_path in glob.glob(os.path.join(save_dir, "CKPT*", "CKPT.yaml")):
        with open(meta_path, encoding="utf-8") as f:
            content = f.read()
        # Parse the loss value from YAML-like meta file
        for line in content.splitlines():
            if "OR-PIT_loss" in line:
                try:
                    val = float(line.split(":")[-1].strip())
                    if val < best_loss:
                        best_loss = val
                        best_dir = os.path.dirname(meta_path)
                except ValueError:
                    pass
    return best_dir, best_loss


def load_model(hparams, device):
    """Build the DExFormer model and load only the model weights from the best CKPT."""
    model = hparams["dexformer"].to(device)

    save_dir = os.path.join(hparams["output_folder"], "save")
    best_dir, best_loss = find_best_ckpt_dir(save_dir)

    if best_dir is None:
        logger.warning("No checkpoint found – using random weights!")
    else:
        model_ckpt = os.path.join(best_dir, "model.ckpt")
        logger.info(f"Loading best checkpoint (loss={best_loss:.4f}): {best_dir}")
        state = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(state)

    model.eval()
    return model


def infer_mixture(model, mix_path, num_spks, max_len, device):
    """Run the model on a single mixture file. Returns (mix, [est_spk1, ...])."""
    mix, sr = torchaudio.load(mix_path)          # [1, L]
    if sr != 16000:
        mix = torchaudio.functional.resample(mix, sr, 16000)

    mix = mix[0, :max_len]                        # [L]
    mix_in = mix.unsqueeze(0).to(device)          # [1, L]

    with torch.no_grad():
        est_sources = model.extract_all(mix_in, num_speakers=num_spks, training=False)
        # est_sources is a list of num_spks tensors, each [1, L]

    return mix.cpu(), [s.squeeze(0).cpu() for s in est_sources]


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_model(hparams, device)

    num_spks   = hparams["num_spks"]
    sample_rate = hparams.get("sample_rate", 16000)
    max_len     = int(hparams.get("max_audio_length", 3.0) * sample_rate)

    out_dir = os.path.join(hparams["output_folder"], "best_checkpoint_audio")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Saving separated audio to: {out_dir}")

    # Read the CSV to get all mixture paths
    csv_path = hparams["train_data"]
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info(f"Running inference on {len(rows)} mixtures …")

    data_root = hparams.get("data_folder", "")

    for idx, row in enumerate(rows):
        mix_path = row["mix_wav"].replace("{data_root}", data_root)

        mix, est_list = infer_mixture(model, mix_path, num_spks, max_len, device)

        prefix = f"mix_{idx:02d}"

        # Save mixture
        torchaudio.save(
            os.path.join(out_dir, f"{prefix}_mix.wav"),
            mix.unsqueeze(0), sample_rate
        )

        # Save ground-truth speakers (if paths exist in CSV)
        for spk_idx in range(1, num_spks + 1):
            key = f"s{spk_idx}_wav"
            if key in row:
                gt_path = row[key].replace("{data_root}", data_root)
                gt, sr = torchaudio.load(gt_path)
                gt = gt[0, :max_len]
                torchaudio.save(
                    os.path.join(out_dir, f"{prefix}_gt_spk{spk_idx}.wav"),
                    gt.unsqueeze(0), sample_rate
                )

        # Save estimated speakers
        for spk_idx, est in enumerate(est_list, start=1):
            torchaudio.save(
                os.path.join(out_dir, f"{prefix}_est_spk{spk_idx}.wav"),
                est.unsqueeze(0), sample_rate
            )

        logger.info(f"  [{idx+1}/{len(rows)}] Saved {prefix}_*.wav")

    logger.info("Done! All audio saved.")


if __name__ == "__main__":
    main()
