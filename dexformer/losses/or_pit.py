import torch
from .snr import snr

def compute_or_pit_loss(estimated_sources, ground_truth_sources, initial_mixture):
    """
    Computes the OR-PIT loss across all extraction steps for a batch.
    
    Args:
        estimated_sources: list of N tensors of shape [B, L]
        ground_truth_sources: list of N tensors of shape [B, L]
        initial_mixture: tensor of shape [B, L] (typically sum of ground_truths)
        
    Returns:
        total_loss: scalar tensor (mean across batch and steps)
    """
    N = len(ground_truth_sources)
    B = estimated_sources[0].shape[0]
    device = estimated_sources[0].device
    
    # To compute the loss per batch item properly, we can do it with a loop over B,
    # or use batched operations. Since N is small (e.g., 3), a loop over B is acceptable,
    # but batched operations are preferred.
    
    # Let's track the actual implicit residual during extraction.
    # implicit_residual[b] tracks the current residual for batch item b.
    implicit_residual = initial_mixture.clone()
    
    # Track which speakers have been assigned for each batch item.
    # remaining_targets is a list of lists (or sets), one per batch item.
    remaining_targets = [set(range(N)) for _ in range(B)]
    
    total_batch_loss = 0.0
    
    for step_i in range(N):
        s_hat = estimated_sources[step_i] # [B, L]
        M = N - step_i # Number of remaining speakers
        
        # We will compute the combined loss for each batch item separately
        # because the assignments can differ per batch item.
        step_losses = []
        for b in range(B):
            best_combined_loss = None
            best_j = None
            
            s_hat_b = s_hat[b].unsqueeze(0) # [1, L]
            implicit_resid_b = implicit_residual[b].unsqueeze(0) - s_hat_b # [1, L]
            
            # Find best assignment without gradients
            with torch.no_grad():
                for j in remaining_targets[b]:
                    s_j = ground_truth_sources[j][b].unsqueeze(0)
                    target_loss_j = -snr(s_hat_b, s_j)
                    
                    if M > 1:
                        # sum of remaining speakers excluding j
                        rest_gt_j = sum(ground_truth_sources[k][b].unsqueeze(0) for k in remaining_targets[b] if k != j)
                        residual_loss_j = -snr(implicit_resid_b, rest_gt_j)
                    else:
                        # M=1 edge case: score residual against silence
                        silence = torch.zeros_like(implicit_resid_b)
                        residual_loss_j = -snr(implicit_resid_b, silence)
                        
                    combined_loss_j = target_loss_j + (1.0 / max(M - 1, 1)) * residual_loss_j
                    
                    if best_combined_loss is None or combined_loss_j < best_combined_loss:
                        best_combined_loss = combined_loss_j
                        best_j = j
            
            # Now recompute the chosen assignment WITH gradients
            s_j = ground_truth_sources[best_j][b].unsqueeze(0)
            target_loss_j = -snr(s_hat_b, s_j)
            
            if M > 1:
                rest_gt_j = sum(ground_truth_sources[k][b].unsqueeze(0) for k in remaining_targets[b] if k != best_j)
                residual_loss_j = -snr(implicit_resid_b, rest_gt_j)
            else:
                silence = torch.zeros_like(implicit_resid_b)
                residual_loss_j = -snr(implicit_resid_b, silence)
                
            final_combined_loss_j = target_loss_j + (1.0 / max(M - 1, 1)) * residual_loss_j
            step_losses.append(final_combined_loss_j)
            
            # Remove chosen target
            remaining_targets[b].remove(best_j)
            
        # Update implicit residual for the next step (batched, out-of-place)
        implicit_residual = implicit_residual - s_hat
        
        # [HYPOTHESIS] Average across batch
        step_loss_batch = torch.cat(step_losses).mean()
        total_batch_loss += step_loss_batch
        
    # [HYPOTHESIS - not specified in source paper]: Default to mean-over-steps
    return total_batch_loss / N
