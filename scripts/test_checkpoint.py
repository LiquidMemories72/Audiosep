import argparse
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexformer.models.dexformer import DExFormer
from dexformer.models.dexformer import DExFormer
def main():
    parser = argparse.ArgumentParser(description="Test a DExFormer checkpoint on a single mixture audio file.")
    parser.add_argument("config", help="Path to the model config file (e.g., configs/kaggle_libri4mix.yaml)")
    parser.add_argument("checkpoint", help="Path to the model.ckpt file")
    parser.add_argument("mix_wav", help="Path to the input mixture audio file (.wav)")
    parser.add_argument("--output_dir", default="separated_output", help="Directory to save the separated audio files")
    parser.add_argument("--fixed_k", action="store_true", help="If set, runs exact fixed-K extraction (ignores STC stopping criteria)")
    args = parser.parse_args()
    print(f"Loading configuration from {args.config}...")
    with open(args.config, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = hparams["dexformer"].to(device)
    print(f"Loading checkpoint from: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    num_spks = hparams.get("num_spks", 2)
    sample_rate = hparams.get("sample_rate", 16000)
    print(f"Loading mixture audio: {args.mix_wav}")
    mix, sr = torchaudio.load(args.mix_wav)
    if sr != sample_rate:
        print(f"Resampling from {sr} to {sample_rate}...")
        mix = torchaudio.functional.resample(mix, sr, sample_rate)
    if mix.ndim == 2 and mix.shape[0] > 1:
        mix = mix[0:1, :] 
    elif mix.ndim == 1:
        mix = mix.unsqueeze(0)
    mix_in = mix.to(device)
    if args.fixed_k:
        print(f"Running separation in fixed-K mode to extract EXACTLY {num_spks} speakers...")
    else:
        print(f"Running separation to extract up to {num_spks} speakers (using STC)...")
    with torch.no_grad():
        est_sources = model.extract_all(mix_in, num_speakers=num_spks, training=args.fixed_k)
    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.mix_wav))[0]
    print("Saving separated tracks...")
    for spk_idx, est in enumerate(est_sources, start=1):
        out_path = os.path.join(args.output_dir, f"{basename}_est_spk{spk_idx}.wav")
        if est.ndim == 1:
            est = est.unsqueeze(0)
        elif est.ndim == 3:
            est = est.squeeze(0)
        torchaudio.save(out_path, est.cpu(), sample_rate)
        print(f"  -> Saved {out_path}")
    print("Done!")
if __name__ == "__main__":
    main()