import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dynamic import for scripts starting with numbers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
train_mod = importlib.import_module('04_train_diffusion')
DiffusionModel = train_mod.DiffusionModel
LitDiffusion = train_mod.LitDiffusion

def generate_samples(checkpoint_path, num_samples=12):
    """
    Loads the diffusion model and generates synthetic water usage samples.
    """
    model_backbone = DiffusionModel(cond_in_dim=12, cond_out_dim=10)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    lit_model = LitDiffusion.load_from_checkpoint(checkpoint_path, model=model_backbone, weights_only=True)
    lit_model.eval()
    
    # Use MPS if available (Mac M-series), otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    lit_model.to(device)
    
    print(f"Generating {num_samples} samples on {device}...")
    
    c_in = torch.zeros(num_samples, 12, 96).to(device)
    c_out = torch.zeros(num_samples, 10, 96).to(device)
    
    # Load stats for normalization
    import json
    with open("norm_stats.json", "r") as f:
        stats = json.load(f)
    
    # Simulate variation and NORMALIZE
    for i in range(num_samples):
        # Indoor conditions indices 0-5 are calendar (sin/cos already normalized)
        c_in[i, 0, :] = 0.8  # Dummy normalized seasonal factor
        
        # Indoor usage lags (Indices 6-11)
        # Simulate some baseline activity (e.g., 0.1 gallons) and normalize it
        base_lag_usage = 0.1 
        norm_lag = (np.log1p(base_lag_usage) - stats["log_mean"]) / (stats["log_std"] + 1e-6)
        c_in[i, 6:12, :] = torch.tensor(norm_lag)
        
        # Outdoor conditions (e.g., Temp = 35C)
        temp_raw = 35.0 + np.random.normal(0, 2)
        temp_norm = (temp_raw - stats["temp_c_mean"]) / (stats["temp_c_std"] + 1e-6)
        c_out[i, 0, :] = torch.tensor(temp_norm)
        
        # Precipitation (0.0 mm)
        precip_norm = (0.0 - stats["precip_mm_mean"]) / (stats["precip_mm_std"] + 1e-6)
        c_out[i, 1, :] = torch.tensor(precip_norm)
        
        # GDD (High GDD for summer)
        gdd_raw = 150.0 
        gdd_norm = (gdd_raw - stats["gdd_7d_mean"]) / (stats["gdd_7d_std"] + 1e-6)
        c_out[i, 9, :] = torch.tensor(gdd_norm)
        
    with torch.no_grad():
        x = torch.randn(num_samples, 2, 96).to(device)
        
        T = lit_model.T
        alphas = lit_model.alphas
        alphas_hat = lit_model.alphas_hat
        betas = lit_model.betas
        
        for t in reversed(range(T)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise component
            eps_theta = lit_model.model(x, t_batch, c_in, c_out)
            
            # DDPM Update Step
            alpha = alphas[t]
            alpha_hat = alphas_hat[t]
            beta = betas[t]
            
            z = torch.randn_like(x) if t > 0 else 0
            
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * eps_theta
            ) + torch.sqrt(beta) * z
            
    x_np = x.cpu().numpy() # Shape: (B, 2, 96)
    occurrence_mask = x_np[:, 0, :]
    log_magnitude = x_np[:, 1, :]
    
    # Load stats dynamically for unscaling
    import json
    with open("norm_stats.json", "r") as f:
        stats = json.load(f)
        log_mean = stats["log_mean"]
        log_std = stats["log_std"]
    
    # Inverse Transform
    gate = (occurrence_mask > 0.1).astype(float) # Using optimized threshold
    unscaled_log = np.clip((log_magnitude * log_std) + log_mean, a_min=-10.0, a_max=10.0)
    samples = gate * np.maximum(np.expm1(unscaled_log), 0.0)
    
    import pandas as pd
    data_list = []
    for i in range(num_samples):
        for t in range(96):
            data_list.append({
                'sample_id': i,
                'interval': t,
                'usage': samples[i, t]
            })
    df = pd.DataFrame(data_list)
    df.to_csv("scratch/synthetic_samples_raw.csv", index=False)
    print("Successfully exported samples to scratch/synthetic_samples_raw.csv")
    
    return samples

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic samples from a trained diffusion model.")
    parser.add_argument("--ckpt", type=str, default="final_water_diffusion_model.ckpt", help="Path to the checkpoint file.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate.")
    parser.add_argument("--output", type=str, default="images/synthetic_samples_poc.png", help="Path to save the output plot.")
    
    args = parser.parse_args()
    
    # Ensure images directory exists
    os.makedirs("images", exist_ok=True)
    
    print(f"Loading model from {args.ckpt}...")
    try:
        generate_samples(args.ckpt, num_samples=args.num_samples)
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
