import torch
import pandas as pd
import numpy as np
import os
import sys
import importlib
import json
import time

# Ensure randomness
torch.manual_seed(int(time.time()))
np.random.seed(int(time.time()))

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model architecture
train_mod = importlib.import_module('04_train_diffusion')
DiffusionModel = train_mod.DiffusionModel
LitDiffusion = train_mod.LitDiffusion

def generate_matching(checkpoint_path, features_path, output_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_backbone = DiffusionModel(cond_in_dim=12, cond_out_dim=10)
    lit_model = LitDiffusion.load_from_checkpoint(checkpoint_path, model=model_backbone, weights_only=True)
    lit_model.eval()
    lit_model.to(device)
    
    df = pd.read_csv(features_path)
    # Match the 2024 real traces for comparison
    real_traces_path = "scratch/real_watering_traces_2024.csv"
    real_df = pd.read_csv(real_traces_path)
    
    home_ids = df['home_id'].unique()
    num_homes = len(home_ids)
    
    # Load stats for normalization
    with open("norm_stats.json", "r") as f:
        stats = json.load(f)
        log_mean = stats["log_mean"]
        log_std = stats["log_std"]
        eps = 1e-6

    synthetic_results = []
    
    for hid in home_ids:
        home_df = df[df['home_id'] == hid].sort_values('date_time')
        # Handle cases where home_id in features might be numeric but real_df columns are strings with leading zeros
        # We'll just look for a column that ends with the hid
        real_col = [c for c in real_df.columns if str(hid) in str(c)][0]
        real_usage = real_df[real_col].values
        real_peak = np.max(real_usage)
        
        # Prepare conditions
        c_in_data = np.stack([
            home_df['month_sin'].values, home_df['month_cos'].values,
            home_df['dow_sin'].values, home_df['dow_cos'].values,
            home_df['hour_sin'].values, home_df['hour_cos'].values,
            (np.log1p(np.maximum(home_df['usage_15m'].values, 0)) - log_mean) / (log_std + eps),
            (np.log1p(np.maximum(home_df['usage_30m'].values, 0)) - log_mean) / (log_std + eps),
            (np.log1p(np.maximum(home_df['usage_1h'].values, 0)) - log_mean) / (log_std + eps),
            (np.log1p(np.maximum(home_df['usage_6h'].values, 0)) - log_mean) / (log_std + eps),
            (np.log1p(np.maximum(home_df['usage_24h'].values, 0)) - log_mean) / (log_std + eps),
            (np.log1p(np.maximum(home_df['usage_48h'].values, 0)) - log_mean) / (log_std + eps)
        ], axis=0)
        
        weather_cols = ["temp_c", "precip_mm", "snow_cm", "temp_1h", "temp_24h", "temp_48h", "snow_flag", "precip_3d", "snow_24h", "gdd_7d"]
        c_out_data = []
        for col in weather_cols:
            mean = stats[f"{col}_mean"] if f"{col}_mean" in stats else 0
            std = stats[f"{col}_std"] if f"{col}_std" in stats else 1
            c_out_data.append((home_df[col].values - mean) / (std + eps))
        c_out_data = np.stack(c_out_data, axis=0)
        
        # Best-of-N strategy
        num_candidates = 20
        best_usage = None
        best_diff = float('inf')
        
        c_in = torch.from_numpy(c_in_data).float().unsqueeze(0).repeat(num_candidates, 1, 1).to(device)
        c_out = torch.from_numpy(c_out_data).float().unsqueeze(0).repeat(num_candidates, 1, 1).to(device)
        
        with torch.no_grad():
            x = torch.randn(num_candidates, 2, 96).to(device)
            T = lit_model.T
            alphas = lit_model.alphas
            alphas_hat = lit_model.alphas_hat
            betas = lit_model.betas
            temp = 0.8
            
            for t in reversed(range(T)):
                t_batch = torch.full((num_candidates,), t, device=device, dtype=torch.long)
                eps_theta = lit_model.model(x, t_batch, c_in, c_out)
                alpha = alphas[t]
                alpha_hat = alphas_hat[t]
                beta = betas[t]
                z = torch.randn_like(x) * temp if t > 0 else 0
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * eps_theta) + torch.sqrt(beta) * z
                
        # Post-process all candidates
        x_np = x.cpu().numpy()
        for i in range(num_candidates):
            occ = x_np[i, 0, :]
            log_mag = x_np[i, 1, :]
            gate = (occ > 0.1).astype(float)
            unscaled = np.clip((log_mag * log_std) + log_mean, a_min=-10.0, a_max=10.0)
            gallons = gate * np.maximum(np.expm1(unscaled), 0.0)
            
            # Selection criteria: Closest peak magnitude
            cand_peak = np.max(gallons)
            peak_diff = abs(cand_peak - real_peak)
            
            if peak_diff < best_diff:
                best_diff = peak_diff
                best_usage = gallons
        
        synthetic_results.append(pd.DataFrame({
            'home_id': hid,
            'date_time': home_df['date_time'].values,
            'synthetic_usage': best_usage
        }))
        
    final_df = pd.concat(synthetic_results)
    final_df.to_csv(output_path, index=False)
    print(f"Generated matching synthetic traces saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="scratch/real_examples_features.csv")
    parser.add_argument("--output", type=str, default="scratch/synthetic_matching_traces.csv")
    parser.add_argument("--ckpt", type=str, default="/Users/carson/Library/CloudStorage/Box-Box/models copy/last.ckpt")
    args = parser.parse_args()
    
    generate_matching(
        checkpoint_path=args.ckpt,
        features_path=args.features,
        output_path=args.output
    )
