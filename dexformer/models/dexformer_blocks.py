import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalization factory
# ---------------------------------------------------------------------------

def build_norm(norm_type: str, num_channels: int) -> nn.Module:
    """
    Factory for the normalization layer inside MobileNetV2Bottleneck.

    Args:
        norm_type: One of:
            "groupnorm"  — GroupNorm(1, C), equivalent to LayerNorm over channels.
                           Works correctly at batch_size=1.
                           [DEFAULT for Kaggle training — DEVIATION from source paper]
            "batchnorm"  — BatchNorm1d(C), as specified in the source paper
                           (Lee, Kim & Jang, Sensors 2025).
                           Requires batch_size >= 2 during training; degenerates at
                           batch_size=1 (variance collapses to 0).
        num_channels: Number of channels to normalize over.

    Returns:
        An nn.Module that normalizes a [B, C, T] tensor.
    """
    if norm_type == "groupnorm":
        # num_groups=1 => normalizes over all C channels per sample: equivalent to
        # InstanceNorm but with learnable affine parameters. Safe at batch_size=1.
        return nn.GroupNorm(1, num_channels)
    elif norm_type == "batchnorm":
        return nn.BatchNorm1d(num_channels)
    else:
        raise ValueError(
            f"Unknown norm_type '{norm_type}'. Choose 'groupnorm' (default) or 'batchnorm'."
        )


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        bottleneck_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, bottleneck_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(bottleneck_channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.size()
        y = x.mean(dim=2)           # Global average pool -> [B, C]
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(B, C, 1)
        return x * y


# ---------------------------------------------------------------------------
# MobileNetV2 Bottleneck
# ---------------------------------------------------------------------------

class MobileNetV2Bottleneck(nn.Module):
    """
    MobileNetV2-style inverted residual bottleneck for 1-D sequences.

    Paper spec (Lee, Kim & Jang, DExFormer, Sensors 2025):
      - Convolution order: 1x1 expand -> 3x3 depthwise -> 1x1 project.
      - BatchNorm is applied to the first two convolutions ONLY (not the final projection).
      - Activation: HardSwish on the first two convolutions; the final projection is linear.

    Normalization deviation (this implementation):
      By default, norm_type="groupnorm" replaces BatchNorm1d with
      GroupNorm(1, C) to support batch_size=1 on Kaggle GPUs (16 GB VRAM).
      Set norm_type="batchnorm" to reproduce the exact paper configuration.
      See README.md Section "Normalization Deviation" for a full discussion.
    """

    def __init__(self, in_channels: int, expansion_factor: int = 2, norm_type: str = "batchnorm"):
        super(MobileNetV2Bottleneck, self).__init__()
        hidden_dim = int(in_channels * expansion_factor)
        self.norm_type = norm_type

        # 1x1 pointwise expand -- norm + HardSwish
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.norm1 = build_norm(norm_type, hidden_dim)
        self.act1  = nn.Hardswish()

        # 3x3 depthwise -- norm + HardSwish
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                               groups=hidden_dim, bias=False)
        self.norm2 = build_norm(norm_type, hidden_dim)
        self.act2  = nn.Hardswish()

        # 1x1 pointwise projection -- NO norm, NO activation (linear, as per paper)
        self.conv3 = nn.Conv1d(hidden_dim, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)))
        out = self.conv3(out)           # linear projection -- no norm, no activation
        return out + residual


# ---------------------------------------------------------------------------
# ConvTLayer  (single Intra or Inter transformer layer)
# ---------------------------------------------------------------------------

class ConvTLayer(nn.Module):
    """
    Single ConvT layer: MHA -> AddNorm -> MobileNetV2Bottleneck -> SE -> AddNorm.

    The norm_type argument is forwarded to MobileNetV2Bottleneck and controls
    the normalization inside the convolutional FFN sub-block only. The two
    LayerNorm layers on the attention residual connections are always LayerNorm
    (they operate on [B, T, C] and are unaffected by batch size).
    """

    def __init__(self, d_model: int, nhead: int, expansion_factor: int = 2,
                 dropout: float = 0.1, norm_type: str = "groupnorm"):
        super(ConvTLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)      # attention residual -- always LN

        self.bottleneck = MobileNetV2Bottleneck(d_model, expansion_factor, norm_type=norm_type)
        self.se         = SqueezeExcitation(d_model)
        self.norm2      = nn.LayerNorm(d_model)  # conv residual -- always LN
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]  (Dual_Computation_Block passes [B*S, K, N] or [B*K, S, N])

        # --- Multi-head attention ---
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # --- Convolutional FFN (expects [B, C, T]) ---
        x_conv   = x.transpose(1, 2)                   # [B, C, T]
        conv_out = self.se(self.bottleneck(x_conv))     # [B, C, T]
        conv_out = conv_out.transpose(1, 2)             # [B, T, C]

        x = self.norm2(x + self.dropout(conv_out))
        return x


# ---------------------------------------------------------------------------
# ConvTBlock  (K-repeated ConvTLayer, used as intra_mdl / inter_mdl)
# ---------------------------------------------------------------------------

class ConvTBlock(nn.Module):
    """
    Stacks K ConvTLayer modules. Used as both IntraConvT and InterConvT.

    Args:
        K:               Number of repeated layers (DExFormer paper: K=8).
        d_model:         Feature dimension.
        nhead:           Number of attention heads.
        expansion_factor: Bottleneck expansion ratio (default 2).
        dropout:         Dropout probability.
        norm_type:       Normalization inside the bottleneck. See build_norm().
                         "groupnorm" (default) for batch_size=1 safety.
                         "batchnorm" to reproduce the paper exactly.
    """

    def __init__(self, K: int, d_model: int, nhead: int,
                 expansion_factor: int = 2, dropout: float = 0.1,
                 norm_type: str = "batchnorm"):
        super(ConvTBlock, self).__init__()
        self.layers = nn.ModuleList([
            ConvTLayer(d_model, nhead, expansion_factor, dropout, norm_type)
            for _ in range(K)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, T, C = 2, 100, 256

    for norm in ("groupnorm", "batchnorm"):
        dummy = torch.randn(B, T, C)
        block = ConvTBlock(K=8, d_model=C, nhead=8, norm_type=norm)
        out   = block(dummy)
        assert out.shape == dummy.shape, f"[{norm}] shape mismatch: {out.shape}"
        print(f"[{norm}] OK -- {dummy.shape} -> {out.shape}")
