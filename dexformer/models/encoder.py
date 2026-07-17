import torch.nn as nn
from speechbrain.lobes.models.dual_path import Encoder as SBEncoder
class Encoder(nn.Module):
    """
    Thin wrapper around SpeechBrain's Encoder.
    Input: [B, L]
    Output: [B, N_filters, T_enc]
    """
    def __init__(self, kernel_size=16, out_channels=256, in_channels=1):
        super(Encoder, self).__init__()
        self.encoder = SBEncoder(
            kernel_size=kernel_size,
            out_channels=out_channels,
            in_channels=in_channels
        )
    def forward(self, x):
        return self.encoder(x)