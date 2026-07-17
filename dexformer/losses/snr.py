import torch
def snr(estimate, target, eps=1e-8):
    """
    Computes Signal-to-Noise Ratio (SNR).
    SNR = 10 * log10( ||target||^2 / ||target - estimate||^2 )
    Args:
        estimate: torch.Tensor [B, L]
        target: torch.Tensor [B, L]
    Returns:
        torch.Tensor [B]
    """
    target_power = torch.sum(target ** 2, dim=-1)
    noise = target - estimate
    noise_power = torch.sum(noise ** 2, dim=-1)
    return 10 * torch.log10(target_power / (noise_power + eps) + eps)