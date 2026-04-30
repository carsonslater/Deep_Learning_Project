import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Import dataloader from step 03
import sys
import os
import importlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
dataset_mod = importlib.import_module('03_export_to_parquet')
get_dataloader = dataset_mod.get_dataloader

class ConditioningEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, c):
        return self.net(c)

class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, hidden_dim)
        self.to_beta  = nn.Linear(cond_dim, hidden_dim)

    def forward(self, h, c):
        gamma = self.to_gamma(c).unsqueeze(-1)
        beta  = self.to_beta(c).unsqueeze(-1)
        return gamma * h + beta

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.film  = FiLM(cond_dim, out_ch)
        self.act   = nn.SiLU()

    def forward(self, x, c):
        h = self.act(self.conv1(x))
        h = self.film(h, c)
        h = self.act(self.conv2(h))
        # Handle skip connection channel mismatch
        if x.shape[1] != h.shape[1]:
            # This handles in_ch -> out_ch dimension change
            # Alternatively use a 1x1 conv, but for simplicity here we assume
            # pad or just simple 1x1 projection
            pass # We'll just add a 1x1 conv if needed. Wait, in standard unet:
        # We need a 1x1 conv if channels don't match
        if not hasattr(self, 'skip_conv'):
            if in_ch != out_ch:
                self.skip_conv = nn.Conv1d(in_ch, out_ch, 1).to(x.device)
            else:
                self.skip_conv = nn.Identity().to(x.device)
        
        return h + self.skip_conv(x)

class UNet1D(nn.Module):
    def __init__(self, cond_dim=64):
        super().__init__()

        self.down1 = ResBlock(1, 32, cond_dim)
        self.down2 = ResBlock(32, 64, cond_dim)

        self.pool = nn.AvgPool1d(2)

        self.up1 = ResBlock(64, 32, cond_dim)
        self.up2 = ResBlock(32, 32, cond_dim)

        self.out = nn.Conv1d(32, 1, 1)

    def forward(self, x, c):
        d1 = self.down1(x, c)
        d2 = self.down2(self.pool(d1), c)

        u1 = F.interpolate(d2, scale_factor=2)
        u1 = self.up1(u1, c)

        u2 = self.up2(u1 + d1, c)

        return self.out(u2)

class DiffusionModel(nn.Module):
    def __init__(self, cond_in_dim, cond_out_dim):
        super().__init__()
        # Dual-stream conditioning model
        self.encoder_in = ConditioningEncoder(cond_in_dim, 64)
        self.encoder_out = ConditioningEncoder(cond_out_dim, 64)
        
        # We'll use two UNets for the two streams (indoor + outdoor)
        # as specified in "epsilon_theta = epsilon^(in) + epsilon^(out)"
        self.unet_in = UNet1D(cond_dim=64)
        self.unet_out = UNet1D(cond_dim=64)

    def forward(self, x_t, t, c_in, c_out):
        # We also need to encode t and add it to c, or just use it.
        # The formulation in HOW_TO_IMPLEMENT doesn't explicitly show t injection into UNet,
        # but typically t is embedded and added to c.
        # For simplicity based strictly on the provided snippet, t is passed but not used directly
        # in the unet since c is already encoded. We'll encode t and add to c.
        
        # Simple sinusoidal positional embedding for t
        t_embed = self._get_time_embed(t, 64, x_t.device)
        
        c_in_embed = self.encoder_in(c_in) + t_embed
        c_out_embed = self.encoder_out(c_out) + t_embed
        
        pred_in = self.unet_in(x_t, c_in_embed)
        pred_out = self.unet_out(x_t, c_out_embed)
        
        return pred_in, pred_out

    def _get_time_embed(self, t, dim, device):
        half_dim = dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float().unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class LitDiffusion(pl.LightningModule):
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model
        self.T = T
        
        # Precompute noise schedule
        self.register_buffer('beta', torch.linspace(1e-4, 0.02, T))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

    def q_sample(self, x0, t, noise):
        sqrt_ab = self.alpha_bar[t].sqrt().view(-1, 1, 1)
        sqrt_one_minus = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_one_minus * noise

    def training_step(self, batch, batch_idx):
        x, c_in, c_out = batch

        noise = torch.randn_like(x)
        t = torch.randint(0, self.T, (x.size(0),), device=self.device)

        x_t = self.q_sample(x, t, noise)

        pred_in, pred_out = self.model(x_t, t, c_in, c_out)
        noise_pred = pred_in + pred_out

        loss = ((noise - noise_pred) ** 2).mean()
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__ == "__main__":
    print("Setting up training...")
    
    # Check for MPS
    if torch.backends.mps.is_available():
        accelerator = "mps"
        print("Using MPS backend.")
    else:
        accelerator = "cpu"
        print("MPS not found, using CPU.")
    
    # Feature dimensions from script 02
    d_in = 9  # indoor features
    d_out = 10 # outdoor features
    
    # Instantiate model
    model = DiffusionModel(cond_in_dim=d_in, cond_out_dim=d_out)
    lit_model = LitDiffusion(model, T=1000)
    
    # Dataloader
    try:
        train_dataloader = get_dataloader(batch_size=64, num_workers=4)
        
        # Setup PyTorch Lightning Trainer
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            max_epochs=2,
            precision=32
        )
        
        print("Starting training...")
        trainer.fit(lit_model, train_dataloader)
        
    except Exception as e:
        print(f"Failed to start training. Are the parquet files generated? Error: {e}")
