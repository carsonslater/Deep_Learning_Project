#!/usr/bin/env python3
"""Generate seasonal synthetic traces for Winter, Spring, and Fall."""
import sys, os, json, time
import torch
import numpy as np
import pandas as pd
import importlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(int(time.time()))
np.random.seed(int(time.time()))

train_mod = importlib.import_module('04_train_diffusion')
DiffusionModel = train_mod.DiffusionModel
LitDiffusion   = train_mod.LitDiffusion

CKPT = "/Users/carson/Library/CloudStorage/Box-Box/models copy/water_diffusion_20260504_213400-step=187500.ckpt"

SEASONS = {
    "summer": {"features": "scratch/real_examples_features_2024.csv",
               "real":     "scratch/real_traces_summer_2024.csv",
               "output":   "scratch/synthetic_traces_summer_2024.csv"},
    "winter": {"features": "scratch/real_examples_features_winter_2024.csv",
               "real":     "scratch/real_traces_winter_2024.csv",
               "output":   "scratch/synthetic_traces_winter_2024.csv"},
    "spring": {"features": "scratch/real_examples_features_spring_2024.csv",
               "real":     "scratch/real_traces_spring_2024.csv",
               "output":   "scratch/synthetic_traces_spring_2024.csv"},
    "fall":   {"features": "scratch/real_examples_features_fall_2024.csv",
               "real":     "scratch/real_traces_fall_2024.csv",
               "output":   "scratch/synthetic_traces_fall_2024.csv"},
}

def run_denoising(lit_model, c_in, c_out, n_candidates, device, temp=0.8):
    x = torch.randn(n_candidates, 2, 96).to(device)
    T = lit_model.T
    for t in reversed(range(T)):
        t_batch = torch.full((n_candidates,), t, device=device, dtype=torch.long)
        eps = lit_model.model(x, t_batch, c_in, c_out)
        alpha     = lit_model.alphas[t]
        alpha_hat = lit_model.alphas_hat[t]
        beta      = lit_model.betas[t]
        z = torch.randn_like(x) * temp if t > 0 else 0
        x = (1/torch.sqrt(alpha)) * (x - ((1-alpha)/torch.sqrt(1-alpha_hat))*eps) + torch.sqrt(beta)*z
    return x.cpu().numpy()

def unscale(x_np_i, log_mean, log_std, thresh=0.1):
    gate   = (x_np_i[0] > thresh).astype(float)
    unscal = np.clip(x_np_i[1]*log_std + log_mean, -10, 10)
    return gate * np.maximum(np.expm1(unscal), 0.0)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone = DiffusionModel(cond_in_dim=12, cond_out_dim=10)
    lit = LitDiffusion.load_from_checkpoint(CKPT, model=backbone, weights_only=True)
    lit.eval().to(device)
    print(f"Loaded model from {CKPT}")

    with open("norm_stats.json") as f:
        stats = json.load(f)
    log_mean = stats["log_mean"]
    log_std  = stats["log_std"]
    eps_n    = 1e-6

    for season, paths in SEASONS.items():
        print(f"\nGenerating for {season}...")
        df      = pd.read_csv(paths["features"])
        real_df = pd.read_csv(paths["real"])

        results = []
        for hid in df['home_id'].unique():
            home_df = df[df['home_id'] == hid].sort_values('date_time')

            # Real peak for best-of-N selection
            real_col  = [c for c in real_df.columns if str(int(hid)).lstrip('0') in str(c).lstrip('0')]
            if real_col:
                real_peak = real_df[real_col[0]].max()
            else:
                real_peak = 10.0  # fallback

            # Build conditioning tensors
            c_in_data = np.stack([
                home_df['month_sin'].values, home_df['month_cos'].values,
                home_df['dow_sin'].values,   home_df['dow_cos'].values,
                home_df['hour_sin'].values,  home_df['hour_cos'].values,
                (np.log1p(np.maximum(home_df['usage_15m'].values, 0)) - log_mean) / (log_std+eps_n),
                (np.log1p(np.maximum(home_df['usage_30m'].values, 0)) - log_mean) / (log_std+eps_n),
                (np.log1p(np.maximum(home_df['usage_1h'].values,  0)) - log_mean) / (log_std+eps_n),
                (np.log1p(np.maximum(home_df['usage_6h'].values,  0)) - log_mean) / (log_std+eps_n),
                (np.log1p(np.maximum(home_df['usage_24h'].values, 0)) - log_mean) / (log_std+eps_n),
                (np.log1p(np.maximum(home_df['usage_48h'].values, 0)) - log_mean) / (log_std+eps_n),
            ], axis=0)

            weather_cols = ["temp_c","precip_mm","snow_cm","temp_1h","temp_24h","temp_48h",
                            "snow_flag","precip_3d","snow_24h","gdd_7d"]
            c_out_data = np.stack([
                (home_df[col].values - stats.get(f"{col}_mean", 0)) / (stats.get(f"{col}_std", 1)+eps_n)
                for col in weather_cols
            ], axis=0)

            N = 20
            c_in  = torch.from_numpy(c_in_data).float().unsqueeze(0).repeat(N,1,1).to(device)
            c_out = torch.from_numpy(c_out_data).float().unsqueeze(0).repeat(N,1,1).to(device)

            with torch.no_grad():
                x_np = run_denoising(lit, c_in, c_out, N, device)

            best, best_diff = None, float('inf')
            for i in range(N):
                gallons = unscale(x_np[i], log_mean, log_std)
                diff = abs(gallons.max() - real_peak)
                if diff < best_diff:
                    best_diff = diff
                    best = gallons

            results.append(pd.DataFrame({
                'home_id': hid,
                'date_time': home_df['date_time'].values,
                'synthetic_usage': best
            }))

        pd.concat(results).to_csv(paths["output"], index=False)
        print(f"  Saved to {paths['output']}")

if __name__ == "__main__":
    main()
