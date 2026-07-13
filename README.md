# DExFormer — Implementation Notes

Practical implementation of **DExFormer** (Deflationary Extraction Transformer) for multi-speaker speech separation from Lee, Kim & Jang, *Sensors* 2025.

---

## Quick Start

```python
from dexformer.models.dexformer import DExFormer

# Paper-exact reproduction (BatchNorm, perfectly safe even at batch_size=1 due to sequence chunking)
model = DExFormer(norm_type="batchnorm")
```

---

## Architecture Overview

| Component | Implementation |
|:----------|:--------------|
| Encoder | SpeechBrain `dual_path.Encoder` (Conv1d + ReLU), reused unchanged |
| Decoder | SpeechBrain `dual_path.Decoder` (ConvTranspose1d), reused unchanged |
| MaskNet | Custom `DExFormerMaskNet` — N=3 macro-iterations of (IntraConvT, InterConvT), each with K=8 inner repeats |
| IntraConvT / InterConvT | Custom `ConvTBlock` — MHA + MobileNetV2 bottleneck + Squeeze-Excitation |
| Deflationary loop | `DExFormer.extract_all()` — waveform-domain subtraction, re-encode residual |
| Loss | OR-PIT with plain SNR (not SI-SDR) |
| STC | Parameter-free power thresholding (inference only) |

---

## Normalization

The DExFormer paper specifies `BatchNorm1d` inside the MobileNetV2 bottleneck. While training on Kaggle requires `batch_size=1`, the Dual-Path architecture segments the sequence into `S` chunks. The effective batch size seen by the `BatchNorm1d` layers inside the MaskNet is `B * S`. For a typical audio chunk, `S` is 80+, making `BatchNorm1d` perfectly safe and stable at `batch_size=1`. The implementation defaults to `BatchNorm1d` matching the source paper exactly.

---

## Other Implementation Notes

| Item | Note |
|:-----|:-----|
| OR-PIT aggregation | Mean over N steps `[HYPOTHESIS — not specified in paper]` |
| STC thresholds | `-30 dB` relative to mixture power `[APPROXIMATION — not specified in paper]` |
| OR-PIT M=1 edge case | Residual scored against silence `[IMPLEMENTATION DECISION]` |
| Subtraction domain | Waveform domain (confirmed correct) |
| Loss metric | Plain SNR — **not** SI-SDR; SI-SDR breaks STC |
