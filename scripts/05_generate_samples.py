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
    # 1. Load Model
    # Dimensions match the training script (9 indoor, 10 outdoor features)
    model_backbone = DiffusionModel(cond_in_dim=9, cond_out_dim=10)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    lit_model = LitDiffusion.load_from_checkpoint(checkpoint_path, model=model_backbone)
    lit_model.eval()
    
    # Use MPS if available (Mac M-series), otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    lit_model.to(device)
    
    print(f"✨ Generating {num_samples} samples on {device}...")
    
    # 2. Create Dummy Conditions
    # Simulating a hot summer weekday: 
    # High temp, low humidity, weekday (dummy encoding)
    c_in = torch.zeros(num_samples, 9, 96).to(device)
    c_out = torch.zeros(num_samples, 10, 96).to(device)
    
    # Simulate some variation in conditions across samples
    for i in range(num_samples):
        # Indoor conditions (e.g., house size, family size - static for the day)
        c_in[i, 0, :] = 0.8  # Normalized scale factor
        c_in[i, 1, :] = np.sin(np.pi * (i / num_samples)) # Seasonal variation
        
        # Outdoor conditions (e.g., Temp = 35C + noise)
        c_out[i, 0, :] = 35.0 + np.random.normal(0, 2)
    
    # 3. Reverse Diffusion Loop (DDPM Sampling)
    with torch.no_grad():
        # Start with pure Gaussian noise
        x = torch.randn(num_samples, 1, 96).to(device)
        
        T = lit_model.T
        alphas = lit_model.alphas.to(device)
        alphas_hat = lit_model.alphas_hat.to(device)
        betas = lit_model.betas.to(device)
        
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
            
    # 4. Post-process and Plot
    samples = x.cpu().numpy().squeeze()
    
    # Normalize or clip for visualization if needed
    # (Assuming the model was trained on log-transformed or normalized data)
    samples = np.maximum(samples, 0) # Water usage can't be negative
    
    # Plotting with Premium Aesthetics
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    
    colors = plt.cm.viridis(np.linspace(0.3, 1.0, num_samples))
    
    for i in range(num_samples):
        ax.plot(samples[i], color=colors[i], alpha=0.6, linewidth=1.5, 
                label=f"Sample {i+1}" if i < 5 else "")
    
    # Average profile
    ax.plot(samples.mean(axis=0), color='white', linewidth=3, linestyle='--', 
            label="Mean Profile", zorder=10)
    
    ax.set_title("Synthetic Water Usage Generation (Diffusion Model POC)", 
                 fontsize=18, fontweight='bold', pad=20, color='white')
    ax.set_xlabel("Time of Day (15-min Intervals)", fontsize=12, color='#cccccc')
    ax.set_ylabel("Normalized Usage Intensity", fontsize=12, color='#cccccc')
    
    # Formatting grid and spines
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    
    # Add a custom legend
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("synthetic_samples_poc.png", facecolor='#0a0a0a')
    print("✅ Successfully generated samples and saved to synthetic_samples_poc.png")
    
    return samples

if __name__ == "__main__":
    # Check if we are in the scripts directory or root
    ckpt_path = os.path.abspath("final_water_diffusion_model.ckpt")
        
    try:
        generate_samples(ckpt_path)
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
