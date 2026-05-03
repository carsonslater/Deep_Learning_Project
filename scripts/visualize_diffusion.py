import torch
import sys
import os
from modelviz import visualize_threejs, visualize
try:
    from torchview import draw_graph
    HAS_TORCHVIEW = True
except ImportError:
    HAS_TORCHVIEW = False

import importlib.util

# Add the scripts directory to path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(scripts_dir)

# Import the model architecture from 04_train_diffusion.py using importlib
module_name = "train_diffusion"
file_path = os.path.join(scripts_dir, "04_train_diffusion.py")
spec = importlib.util.spec_from_file_location(module_name, file_path)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
DiffusionModel = train_mod.DiffusionModel

def main():
    print("Initializing UNet1D for visualization...")
    
    # Feature dimensions
    cond_dim = 64 + 128 # t_emb (64) + full_cond (128)
    
    # Instantiate one of the UNet streams
    model = train_mod.UNet1D(cond_dim=cond_dim)
    
    # UNet1D.forward(self, x, t_emb, c)
    # x: [B, 2, 96]
    # t_emb: [B, 64]
    # c: [B, 128]
    # Wait, looking at UNet1D.forward:
    # cond = torch.cat([t_emb, c], dim=1)
    
    # Let's wrap it to take a single input for the visualization tool
    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
        def forward(self, x_and_cond):
            # Split into x [B, 2, 96] and cond [B, 192]
            # print(f"Input shape: {x_and_cond.shape}")
            if x_and_cond.ndim == 3:
                x = x_and_cond[:, :2, :]
            else:
                x = x_and_cond[:2, :]
                x = x.unsqueeze(0) # Add batch dim for UNet
            batch_size = x.shape[0]
            t_emb = torch.zeros(batch_size, 64).to(x.device)
            c = torch.zeros(batch_size, 128).to(x.device)
            return self.unet(x, t_emb, c)

    wrapper = UNetWrapper(model)
    
    save_path_3d = "images/diffusion_unet_3d.html"
    save_path_2d = "images/diffusion_unet_2d.png"
    
    print(f"Generating 3D visualization to {save_path_3d}...")
    try:
        visualize_threejs(
            wrapper, 
            input_shape=(2, 96), 
            save_path=save_path_3d
        )
        print(f"Success! 3D visualization generated: {save_path_3d}")
    except Exception as e:
        print(f"3D Visualization failed: {e}")

    print(f"Generating 2D diagram to {save_path_2d}...")
    try:
        visualize(
            wrapper, 
            input_shape=(2, 96), 
            save_path=save_path_2d
        )
        print(f"Success! 2D diagram generated: {save_path_2d}")
    except Exception as e:
        print(f"2D Visualization failed: {e}")

    if HAS_TORCHVIEW:
        save_path_torchview = "images/diffusion_unet_u_shape"
        print(f"Generating high-fidelity U-shape diagram using torchview...")
        try:
            # We need to provide dummy inputs for the trace
            batch_size = 1
            x = torch.randn(batch_size, 2, 96)
            t_emb = torch.randn(batch_size, 64)
            c = torch.randn(batch_size, 128)
            
            # draw_graph will trace the functional graph and show the skip connections
            model_graph = draw_graph(
                model, 
                input_data=[x, t_emb, c],
                expand_nested=True,
                graph_name="UNet1D",
                depth=2
            )
            # Set high DPI for PNG resolution
            model_graph.visual_graph.graph_attr['dpi'] = '300'
            
            model_graph.visual_graph.render(save_path_torchview, format="png", cleanup=True)
            # Also save as SVG for perfect scalability
            model_graph.visual_graph.render(save_path_torchview, format="svg", cleanup=True)
            print(f"Success! High-res diagrams generated: {save_path_torchview}.png and .svg")
        except Exception as e:
            import traceback
            print(f"torchview Visualization failed: {e}")
            traceback.print_exc()
    else:
        print("torchview not found. Skipping high-fidelity diagram.")

if __name__ == "__main__":
    main()
