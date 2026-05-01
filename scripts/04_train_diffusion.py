import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import sys
import os
import importlib
import subprocess
import argparse

# Ensure we can import the dataloader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
dataset_mod = importlib.import_module('03_export_to_parquet')
get_dataloader = dataset_mod.get_dataloader

def notify_me(subject, body=""):
    """Helper to call the R notification function from Python"""
    cmd = [
        "Rscript", "-e",
        f"source('scripts/utils.R'); notify_me_done(subject=\"{subject}\", body=\"{body}\")"
    ]
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"Failed to send notification: {e}")

class FiLM(nn.Module):
    def __init__(self, cond_dim, out_ch):
        super().__init__()
        self.fc = nn.Linear(cond_dim, out_ch * 2)
    def forward(self, x, c):
        # c: [B, cond_dim]
        gamma_beta = self.fc(c).unsqueeze(-1) # [B, out_ch*2, 1]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return x * gamma + beta

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.gn1   = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.gn2   = nn.GroupNorm(8, out_ch)
        self.film  = FiLM(cond_dim, out_ch)
        self.act   = nn.SiLU()
        
        # Handle skip connection channel mismatch
        if in_ch != out_ch:
            self.skip_conv = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x, c):
        h = self.act(self.gn1(self.conv1(x)))
        h = self.film(h, c)
        h = self.act(self.gn2(self.conv2(h)))
        return h + self.skip_conv(x)

class UNet1D(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.down1 = ResBlock(2, 64, cond_dim)
        self.down2 = ResBlock(64, 128, cond_dim)
        self.mid   = ResBlock(128, 128, cond_dim)
        self.up2   = ResBlock(256, 64, cond_dim) # Concat from down2
        self.up1   = ResBlock(128, 64, cond_dim) # Concat from down1
        self.final = nn.Conv1d(64, 2, 1)
        
        # Initialize final layer to be nearly zero to stabilize initial training
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        
        self.pool  = nn.AvgPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

    def forward(self, x, t_emb, c):
        # Merge time and features
        cond = torch.cat([t_emb, c], dim=1)
        
        d1 = self.down1(x, cond)
        d2 = self.down2(self.pool(d1), cond)
        
        m = self.mid(self.pool(d2), cond)
        
        u2 = self.up2(torch.cat([self.upsample(m), d2], dim=1), cond)
        u1 = self.up1(torch.cat([self.upsample(u2), d1], dim=1), cond)
        
        return self.final(u1)

class DiffusionModel(nn.Module):
    def __init__(self, cond_in_dim, cond_out_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        # Dual-stream: specialized branches that both see the full context
        self.unet_in = UNet1D(cond_dim=64+128) # time + full_cond
        self.unet_out = UNet1D(cond_dim=64+128) 
        
        self.c_in_enc = nn.Linear(cond_in_dim, 64)
        self.c_out_enc = nn.Linear(cond_out_dim, 64)

    def forward(self, x, t, c_in, c_out):
        # Ensure batch dimension exists
        if x.ndim == 1: x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2: x = x.unsqueeze(0)
        
        if c_in.ndim == 1: c_in = c_in.unsqueeze(0).unsqueeze(0)
        elif c_in.ndim == 2: c_in = c_in.unsqueeze(0)
        
        if c_out.ndim == 1: c_out = c_out.unsqueeze(0).unsqueeze(0)
        elif c_out.ndim == 2: c_out = c_out.unsqueeze(0)
        
        if t.ndim == 0: t = t.unsqueeze(0)

        # Normalize time to [0, 1] to prevent gradient explosion from raw integers
        t_norm = t.float().view(-1, 1) / 1000.0
        t_emb = self.time_mlp(t_norm) # [B, 64]
        
        # Global pooling of conditions to get a sequence-level embedding
        c_in_emb = self.c_in_enc(c_in.mean(dim=-1))
        c_out_emb = self.c_out_enc(c_out.mean(dim=-1))
        
        # Concatenate so BOTH streams know the calendar/time AND the weather
        full_cond = torch.cat([c_in_emb, c_out_emb], dim=1) # [B, 128]
        
        eps_in = self.unet_in(x, t_emb, full_cond)
        eps_out = self.unet_out(x, t_emb, full_cond)
        
        return eps_in + eps_out

class LitDiffusion(pl.LightningModule):
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model
        self.T = T
        # Simple linear schedule
        self.betas = torch.linspace(1e-4, 0.02, T)
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

    def forward(self, x, t, c_in, c_out):
        return self.model(x, t, c_in, c_out)

    def training_step(self, batch, batch_idx):
        x_0, c_in, c_out = batch
        # x_0: [B, 1, 96]
        
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=self.device)
        noise = torch.randn_like(x_0)
        
        alpha_hat = self.alphas_hat.to(self.device)[t].view(-1, 1, 1)
        x_t = torch.sqrt(alpha_hat) * x_0 + torch.sqrt(1 - alpha_hat) * noise
        
        eps_theta = self.model(x_t, t, c_in, c_out)
        loss = nn.MSELoss()(eps_theta, noise)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get("train_loss")
        if avg_loss is not None:
            notify_me(
                subject=f"[PROGRESS] Training Progress: Epoch {self.current_epoch}",
                body=f"Average Train Loss: {avg_loss:.6f}"
            )

    def on_train_end(self):
        notify_me(
            subject="[DONE] Training Complete!", 
            body="The diffusion model training has finished and the final model is saved as final_water_diffusion_model.ckpt."
        )

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
    d_in = 12  # indoor features (6 calendar + 6 usage lags)
    d_out = 10 # outdoor features (weather)
    
    # Instantiate model
    model = DiffusionModel(cond_in_dim=d_in, cond_out_dim=d_out)
    lit_model = LitDiffusion(model, T=1000)
    
    # Dataloader
    try:
        train_dataloader = get_dataloader(batch_size=128, num_workers=0)
        
        # Versioned Filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"water_diffusion_{timestamp}"
        
        # Setup PyTorch Lightning Trainer
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.getcwd(),
            filename=model_name,
            save_top_k=1,
            monitor="train_loss",
            mode="min"
        )
        
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="train_loss",
            patience=10,
            min_delta=1e-4, # Low tolerance for convergence
            verbose=True,
            mode="min"
        )
        
        # Handle command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
        args = parser.parse_args()

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            max_epochs=args.epochs,
            precision=32,
            gradient_clip_val=1.0,
            callbacks=[checkpoint_callback, early_stop_callback]
        )
        
        print(f"Starting training (Checkpoint: {model_name}.ckpt)...")
        trainer.fit(lit_model, train_dataloader)
        
        # Save explicit final copy with timestamp
        final_path = f"{model_name}_final.ckpt"
        trainer.save_checkpoint(final_path)
        
        # Create a copy as 'final_water_diffusion_model.ckpt' for the pipeline scripts
        import shutil
        shutil.copyfile(final_path, "final_water_diffusion_model.ckpt")
        
        print(f"Training finished. Model saved as {final_path} and synced to final_water_diffusion_model.ckpt")
        
    except Exception as e:
        print(f"Failed to start training. Error: {e}")
