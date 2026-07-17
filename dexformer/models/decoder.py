import torch.nn as nn
from speechbrain.lobes.models.dual_path import Decoder as SBDecoder
class Decoder(nn.Module):
    """
    Thin wrapper around SpeechBrain's Decoder.
    Input: [B, N_filters, T_enc]
    Output: [B, L'] (waveform)
    """
    def __init__(self, kernel_size=16, in_channels=256, out_channels=1):
        super(Decoder, self).__init__()
        self.decoder = SBDecoder(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False
        )
    def forward(self, x):
        return self.decoder(x)