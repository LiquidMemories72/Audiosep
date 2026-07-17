import os
import sys
import torch
import torch.optim as optim
try:
    import torch._dynamo
except ImportError:
    pass
import speechbrain
import speechbrain.utils.importutils
for k in list(sys.modules.keys()):
    if 'k2_fsa' in k or 'k2' == k:
        sys.modules.pop(k, None)
from dexformer.models.dexformer import DExFormer
from dexformer.losses.or_pit import compute_or_pit_loss
def gradient_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DExFormer(
        N_filters=256,
        kernel_size=16,
        stride=8,
        N_macro_iterations=3,
        K_inner_repeats=8,
        nhead=8,
        expansion_factor=2,
        dropout=0.1,
        chunk_size=100,
        norm_type="batchnorm"
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1.5e-4)
    model.train()
    B = 1
    L = 8000
    num_spks = 3
    mixture = torch.randn(B, L, requires_grad=True).to(device)
    targets = [torch.randn(B, L).to(device) for _ in range(num_spks)]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024**3)
        print(f"Memory before forward: {mem_before:.3f} GB")
    est_sources = model.extract_all(
        mixture, 
        num_speakers=num_spks, 
        training=True
    )
    loss = compute_or_pit_loss(est_sources, targets, mixture)
    if torch.isnan(loss):
        print("ERROR: Loss is NaN!")
        sys.exit(1)
    optimizer.zero_grad()
    loss.backward()
    print("\n--- Gradient Audit ---")
    all_ok = True
    zero_grad_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"WARNING: {name} does not require grad.")
            all_ok = False
            continue
        if param.grad is None:
            print(f"ERROR: {name} has None gradient.")
            all_ok = False
            continue
        if not torch.isfinite(param.grad).all():
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            msg = f"ERROR: {name} has non-finite gradients ("
            if has_nan: msg += "NaN "
            if has_inf: msg += "Inf "
            msg += ")."
            print(msg)
            all_ok = False
            continue
        if torch.all(param.grad == 0):
            zero_grad_params.append(name)
    if zero_grad_params:
        print(f"\nWARNING: {len(zero_grad_params)} parameters have exactly zero gradients:")
        for name in zero_grad_params[:10]:
            print(f"  - {name}")
        if len(zero_grad_params) > 10:
            print(f"  - ... and {len(zero_grad_params) - 10} more.")
    if all_ok:
        print("All parameters passed the gradient audit (requires_grad=True, grad is not None, grad is finite).")
    optimizer.step()
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\nPeak GPU Memory: {peak_mem:.3f} GB")
    else:
        print("\nCUDA not available. Could not profile GPU memory.")
if __name__ == "__main__":
    gradient_audit()