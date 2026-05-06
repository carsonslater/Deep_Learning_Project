import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import importlib
from scipy.stats import wasserstein_distance

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model architecture and dataloader
train_mod = importlib.import_module('04_train_diffusion')
dataset_mod = importlib.import_module('03_export_to_parquet')
DiffusionModel = train_mod.DiffusionModel
LitDiffusion = train_mod.LitDiffusion
get_dataloader = dataset_mod.get_dataloader

def evaluate_final(checkpoint_path, num_batches=5, temp=0.8, thresh=0.1):
    """
    Generates a final comparison using optimized parameters.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_backbone = DiffusionModel(cond_in_dim=12, cond_out_dim=10)
    lit_model = LitDiffusion.load_from_checkpoint(checkpoint_path, model=model_backbone, weights_only=True)
    lit_model.eval()
    lit_model.to(device)
    
    loader = get_dataloader(batch_size=32, num_workers=0)
    real_x_list, real_c_in_list, real_c_out_list = [], [], []
    for i, (x, c_in, c_out) in enumerate(loader):
        real_x_list.append(x)
        real_c_in_list.append(c_in)
        real_c_out_list.append(c_out)
        if i >= num_batches - 1: break
            
    real_x = torch.cat(real_x_list, dim=0)
    real_c_in = torch.cat(real_c_in_list, dim=0).to(device)
    real_c_out = torch.cat(real_c_out_list, dim=0).to(device)
    num_samples = real_x.shape[0]

    with open("norm_stats.json", "r") as f:
        stats = json.load(f)
        log_mean, log_std = stats["log_mean"], stats["log_std"]

    with torch.no_grad():
        x_gen = torch.randn(num_samples, 2, 96).to(device)
        T, alphas, alphas_hat, betas = lit_model.T, lit_model.alphas, lit_model.alphas_hat, lit_model.betas
        for t in reversed(range(T)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            eps_theta = lit_model.model(x_gen, t_batch, real_c_in, real_c_out)
            alpha, alpha_hat, beta = alphas[t], alphas_hat[t], betas[t]
            z = torch.randn_like(x_gen) * temp if t > 0 else 0
            x_gen = (1 / torch.sqrt(alpha)) * (x_gen - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * eps_theta) + torch.sqrt(beta) * z

    def process(x_tensor, threshold):
        x_np = x_tensor.cpu().numpy()
        occurrence_mask, log_magnitude = x_np[:, 0, :], x_np[:, 1, :]
        gate = (occurrence_mask > threshold).astype(float)
        unscaled_log = np.clip((log_magnitude * log_std) + log_mean, -10, 10)
        return gate * np.maximum(np.expm1(unscaled_log), 0.0)

    samples_real = process(real_x, 0.5)
    samples_gen = process(x_gen, thresh)
    
    real_nonzero = samples_real[samples_real > 0]
    gen_nonzero = samples_gen[samples_gen > 0]
    w_dist = wasserstein_distance(real_nonzero, gen_nonzero)
    
    import pandas as pd
    os.makedirs("scratch", exist_ok=True)
    real_df = pd.DataFrame({'usage': real_nonzero, 'source': 'Real'})
    gen_df = pd.DataFrame({'usage': gen_nonzero, 'source': 'Synthetic'})
    dist_df = pd.concat([real_df, gen_df])
    dist_df.to_csv("scratch/distribution_data.csv", index=False)
    
    # Also export means for profile plot
    real_mean = samples_real.mean(axis=0)
    gen_mean = samples_gen.mean(axis=0)
    profile_df = pd.DataFrame({
        'interval': np.arange(96),
        'real_mean': real_mean,
        'synthetic_mean': gen_mean
    })
    profile_df.to_csv("scratch/profile_data.csv", index=False)
    
    print("Exported distribution and profile data to scratch/")

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batches", type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs("images", exist_ok=True)
    evaluate_final(args.ckpt, num_batches=args.batches)
