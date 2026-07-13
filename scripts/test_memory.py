import os
import sys

# Workaround for SpeechBrain k2 lazy import bug with torch._dynamo:
# Import torch, optim, and dynamo first so it doesn't see SpeechBrain's LazyModules
import torch
import torch.optim as optim
try:
    import torch._dynamo
except ImportError:
    pass

import speechbrain
from dexformer.models.dexformer import DExFormer
from dexformer.losses.or_pit import compute_or_pit_loss

def test_memory():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up model using Kaggle default (groupnorm)
    print("Initializing DExFormer...")
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
        norm_type="groupnorm"
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1.5e-4)

    # Dummy batch
    B = 1
    L = 8000 # 1 second at 8kHz, reduced to avoid immediate OOM on 6GB GPU
    num_spks = 3
    
    print(f"Creating dummy batch: B={B}, L={L}, num_spks={num_spks}")
    mixture = torch.randn(B, L).to(device)
    targets = [torch.randn(B, L).to(device) for _ in range(num_spks)]
    
    # ------------------
    # Forward Pass
    # ------------------
    print("Starting forward pass...")
    model.train()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024**3)
        print(f"Memory before forward: {mem_before:.3f} GB")
        
    est_sources = model.extract_all(
        mixture, 
        num_speakers=num_spks, 
        training=True
    )
    
    # ------------------
    # Objective & Backward Pass
    # ------------------
    print("Computing loss...")
    loss = compute_or_pit_loss(est_sources, targets, mixture)
    print(f"Loss: {loss.item()}")
    
    if torch.isnan(loss):
        print("ERROR: Loss is NaN!")
        sys.exit(1)

    print("Starting backward pass...")
    optimizer.zero_grad()
    loss.backward()
    
    print("Taking optimizer step...")
    optimizer.step()

    print("\n--- Success! Forward, backward, and optimizer step completed without errors. ---")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        current_mem = torch.cuda.memory_allocated() / (1024**3)
        print(f"Peak GPU Memory: {peak_mem:.3f} GB")
        print(f"Current GPU Memory: {current_mem:.3f} GB")
        
        if peak_mem > 15.0:
            print("WARNING: Peak memory is dangerously close to 16GB limit.")
    else:
        print("CUDA not available. Could not profile GPU memory.")

if __name__ == "__main__":
    test_memory()
