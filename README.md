# DExFormer — Deflationary Extraction Transformer

An implementation of **DExFormer** (Deflationary Extraction Transformer) for multi-speaker speech separation, based on Lee, Kim & Jang, *Sensors* 2025.

---

## Architecture Overview

| Component | Implementation Details |
|:----------|:-----------------------|
| **Encoder** | SpeechBrain `dual_path.Encoder` (Conv1d + ReLU) |
| **Decoder** | SpeechBrain `dual_path.Decoder` (ConvTranspose1d) |
| **MaskNet** | Custom `DExFormerMaskNet` — $N=3$ macro-iterations of (IntraConvT, InterConvT), each with $K=8$ inner repeats |
| **Intra/Inter Blocks** | Custom `ConvTBlock` — Multi-Head Attention (MHA) + MobileNetV2 Bottleneck + Squeeze-and-Excitation (SE) |
| **Deflationary Loop** | Waveform-domain subtraction ($y_{new} = y_{res} - \hat{s}$), re-encoded iteratively |
| **STC** | Sequence Termination Criterion (parameter-free power thresholding, inference only) |
| **Loss** | One-and-Rest Permutation Invariant Training (OR-PIT) optimized via plain SNR |

### Scaling to 3+ Speakers

Because DExFormer separates speech iteratively through a **deflationary extraction process**, it scales to 3 or more speakers natively:
1. The core extraction block (`extract_one`) is speaker-agnostic and uses shared weights.
2. In each iteration, the model extracts one target source and subtracts it from the current residual signal in the waveform domain.
3. To extract $C$ speakers, we simply run this loop $C$ times (e.g. 3 times for Libri3Mix). Since the same extraction module is reused, the model parameters do not grow with the number of speakers.
4. During inference, if the number of speakers is unknown, the Sequence Termination Criterion (STC) monitors the residual energy after each step and decides when to stop the loop automatically.

---

## Preliminary Overfit Results (Tiny Libri3Mix)

A tiny verification experiment was conducted by overfitting the network on 10 real-speech mixtures from LibriSpeech (3 speakers, 16 kHz, mix_clean, cropped to 3 seconds):

* **Initial Loss:** 26.6 dB (OR-PIT SNR)
* **Final Loss (Epoch 50):** 20.8 dB (best checkpoint: 20.4 dB)
* **Gradient Norms:** Highly stable, staying strictly between **10 and 80** after adjusting the learning rate.
* **Peak GPU Memory:** 6593.1 MB (comfortably fits on a 6GB VRAM card).
* **Training Speed:** ~18.7 seconds per iteration (batch size of 1).

> [!NOTE]
> During the initial attempt, a learning rate of `1.0e-3` triggered an exploding gradient norm (climbing to 341+ billion). Lowering the learning rate to `1.0e-4` completely stabilized the training loop and allowed the model to successfully memorize the signals.

---

## Getting Started

### 1. Installation

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
```

### 2. Prepare the Tiny Dataset
Generate a mini 10-mixture dataset using LibriSpeech to verify overfitting:

```bash
python scripts/prepare_tiny_libri3mix.py
```

### 3. Run the Tiny Overfit Experiment
Train on the 10 mixtures to verify that the model can learn and memorize the waveforms:

```bash
$env:PYTHONPATH="."
venv\Scripts\python scripts\train.py configs\tiny_overfit.yaml
```

Training checkpoints, logs, and periodic audio outputs will be written to `results/tiny_overfit/1234/`.

### 4. Run Inference (Best Checkpoint)
Generate clean source separation outputs using the best-performing checkpoint:

```bash
$env:PYTHONPATH="."
venv\Scripts\python scripts\infer_best.py configs\tiny_overfit.yaml
```

Outputs will be saved in `results/tiny_overfit/1234/best_checkpoint_audio/`.

---

## Key Implementation Decisions

* **Normalization Safety:** The paper specifies `BatchNorm1d` inside the MaskNet's MobileNetV2 bottlenecks. Because Dual-Path processing slices audio signals into $S$ overlapping chunks, the effective batch size seen by the norm layers is `B * S` (where $S \approx 80+$ for a 3-second signal). This makes `BatchNorm1d` stable and safe even at `batch_size: 1`.
* **SNR vs. SI-SDR:** In accordance with the paper, the loss uses plain Signal-to-Noise Ratio (SNR) instead of Scale-Invariant SDR. This keeps the scale consistent, which is crucial for the STC module's absolute power-thresholding checks.
* **STC Thresholding:** By default, the relative thresholds for source energy (`H_s`) and residual energy (`H_r`) are set to `-30 dB` relative to the initial mixture's power.
