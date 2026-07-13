import copy
import torch
import torch.nn as nn
from speechbrain.lobes.models.dual_path import (
    Dual_Computation_Block,
    Dual_Path_Model,
    select_norm,
)
from .dexformer_blocks import ConvTBlock
import torch.utils.checkpoint as cp

class DExFormerMaskNet(Dual_Path_Model):
    """
    Modified Dual_Path_Model for DExFormer.
    - Uses ConvTBlock for intra and inter models.
    - num_spks = 1 (outputs a single mask).
    - N=3 macro-iterations of (Intra, Inter) pairs.
    """
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        N=3,
        K=8,
        nhead=8,
        expansion_factor=2,
        dropout=0.1,
        norm="ln",
        chunk_size=100,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        norm_type="batchnorm",
    ):
        # We don't call super().__init__() directly because we want to precisely control the layers.
        # Instead, we just inherit to reuse _Segmentation and _over_add methods.
        nn.Module.__init__(self)

        self.K_chunk = chunk_size  # avoid naming conflict with K inner-repeat loops
        self.num_spks = 1
        self.num_layers = N
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        # Instantiate Intra and Inter models with the chosen norm_type
        intra_model = ConvTBlock(K, out_channels, nhead, expansion_factor, dropout, norm_type=norm_type)
        inter_model = ConvTBlock(K, out_channels, nhead, expansion_factor, dropout, norm_type=norm_type)
        
        self.dual_mdl = nn.ModuleList([])
        for i in range(N):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )
            
        self.conv2d = nn.Conv2d(
            out_channels, out_channels * self.num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        """
        Arguments:
        x : torch.Tensor of dimension [B, N, L]. (B: Batch, N: Filters, L: Time)
        
        Returns:
        out : torch.Tensor of dimension [B, N, L] (Single mask)
        """
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)

        # [B, N, K, S] (K is chunk_size, S is number of chunks)
        x, gap = self._Segmentation(x, self.K_chunk)

        # [B, N, K, S]
        for i in range(self.num_layers):
            if self.training and x.requires_grad:
                x = cp.checkpoint(self.dual_mdl[i], x, use_reentrant=False)
            else:
                x = self.dual_mdl[i](x)
        x = self.prelu(x)

        # [B, N, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B, N, K, S] since num_spks = 1
        x = x.view(B * self.num_spks, -1, K, S)

        # [B, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B, N, L]
        x = self.end_conv1x1(x)
        
        # In DExFormer, we just return the mask which can be bounded or unbounded. 
        # SepFormer uses ReLU at the end. We'll use ReLU.
        x = self.activation(x)
        
        # The dual_path_model usually returns [spks, B, N, L].
        # Since spks = 1, we return [B, N, L] directly.
        return x
