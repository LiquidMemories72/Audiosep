import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .masknet import DExFormerMaskNet
from ..modules.stc import SequenceTerminationCriterion

class DExFormer(nn.Module):
    """
    Deflationary Extraction Transformer for Speech Separation.
    """
    def __init__(
        self,
        N_filters=256,
        kernel_size=16,
        stride=8,
        N_macro_iterations=3,
        K_inner_repeats=8,
        nhead=8,
        expansion_factor=2,
        dropout=0.1,
        chunk_size=100,
        norm_type="batchnorm",  # "batchnorm" (default, exact paper config) or "groupnorm"
    ):
        super(DExFormer, self).__init__()
        
        self.encoder = Encoder(
            kernel_size=kernel_size,
            out_channels=N_filters,
            in_channels=1
        )
        
        self.masknet = DExFormerMaskNet(
            in_channels=N_filters,
            out_channels=N_filters,
            N=N_macro_iterations,
            K=K_inner_repeats,
            nhead=nhead,
            expansion_factor=expansion_factor,
            dropout=dropout,
            chunk_size=chunk_size,
            norm_type=norm_type,
        )
        
        self.decoder = Decoder(
            kernel_size=kernel_size,
            in_channels=N_filters,
            out_channels=1
        )
        
        self.stc = SequenceTerminationCriterion()

    def extract_one(self, residual_waveform):
        """
        Performs a single deflationary extraction step.
        
        Args:
            residual_waveform: torch.Tensor of shape [B, L]
            
        Returns:
            estimated_source: torch.Tensor of shape [B, L]
            new_residual: torch.Tensor of shape [B, L]
        """
        B, L = residual_waveform.shape
        
        # 1. Encode
        # Input shape: [B, L] -> Encoder expects [B, L] 
        h = self.encoder(residual_waveform) # [B, N_filters, T_enc]
        
        # 2. Mask
        mask = self.masknet(h) # [B, N_filters, T_enc]
        
        # 3. Apply mask
        h_masked = h * mask
        
        # 4. Decode
        # Output shape: [B, L']
        source_estimate = self.decoder(h_masked) 
        
        # Match length (Decoder output might be slightly different length due to conv strides)
        L_out = source_estimate.shape[1]
        if L_out > L:
            source_estimate = source_estimate[:, :L]
        elif L_out < L:
            source_estimate = F.pad(source_estimate, (0, L - L_out))
            
        # 5. Compute new residual
        new_residual = residual_waveform - source_estimate
        
        return source_estimate, new_residual

    def extract_all(self, mixture, num_speakers, training=True, max_inference_steps=10):
        """
        Runs the full deflationary extraction loop.
        
        Args:
            mixture: torch.Tensor [B, L]
            num_speakers: int (known during training)
            training: bool
            
        Returns:
            estimated_sources: list of tensors of shape [B, L]
        """
        residual = mixture
        estimated_sources = []
        
        if training:
            max_steps = num_speakers
        else:
            max_steps = max_inference_steps
            
        if not training:
            # We compute initial mixture power for relative thresholding
            mixture_power = torch.mean(mixture ** 2, dim=-1)
            
        for step in range(max_steps):
            source_estimate, residual = self.extract_one(residual)
            estimated_sources.append(source_estimate)
            
            # STC check (implemented in Milestone 9)
            if not training:
                stop = self.stc(source_estimate, residual, mixture_power)
                # stop is [B]. For simplicity, if all items in batch should stop, we break.
                # In practice, batch size for inference is often 1, or we pad.
                if stop.all():
                    break
                
        return estimated_sources
