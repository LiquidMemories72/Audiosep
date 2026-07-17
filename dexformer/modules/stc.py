import torch
import torch.nn as nn
class SequenceTerminationCriterion(nn.Module):
    """
    Parameter-free power thresholding module for DExFormer inference.
    """
    def __init__(self, H_s=-30.0, H_r=-20.0):
        """
        Args:
            H_s: Power threshold for the estimated source in dB.
                 [APPROXIMATION - numeric values not specified in source paper]
            H_r: Power threshold for the residual in dB.
                 [APPROXIMATION - numeric values not specified in source paper]
        """
        super(SequenceTerminationCriterion, self).__init__()
        self.H_s = H_s
        self.H_r = H_r
    def forward(self, source_estimate, residual, mixture_power=None):
        """
        Evaluates whether extraction should stop.
        Args:
            source_estimate: torch.Tensor [B, L]
            residual: torch.Tensor [B, L]
            mixture_power: torch.Tensor [B] optional power of initial mixture to use as reference.
        Returns:
            stop: torch.Tensor of bools [B], True if extraction should stop.
        """
        power_s = torch.mean(source_estimate ** 2, dim=-1)
        power_r = torch.mean(residual ** 2, dim=-1)
        power_s_db = 10 * torch.log10(power_s + 1e-8)
        power_r_db = 10 * torch.log10(power_r + 1e-8)
        if mixture_power is not None:
            mixture_power_db = 10 * torch.log10(mixture_power + 1e-8)
            thresh_s = mixture_power_db + self.H_s
            thresh_r = mixture_power_db + self.H_r
        else:
            thresh_s = self.H_s
            thresh_r = self.H_r
        stop = (power_s_db < thresh_s) | (power_r_db < thresh_r)
        return stop