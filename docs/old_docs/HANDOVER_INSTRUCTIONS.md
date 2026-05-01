# Handover Instructions: Synthetic Water Diffusion Project

## Context
We are building a 1D Conditional Diffusion model to generate synthetic 24-hour water usage profiles. 

## Data State
- **Raw Data**: Located in `data/parquet/`.
- **Features**: Generated via `scripts/01_build_features.R`.
- **Windows**: Generated via `scripts/02_make_windows.R`. 
- **Dataloader**: `scripts/03_export_to_parquet.py` uses an `IterableDataset` with multi-worker sharding.

## Current Problem
The model training is failing with an `IndexError: Dimension out of range` or `linear shape mismatch`. This is happening because the tensors coming from the dataloader in `LitDiffusion.training_step` are not always preserving the batch dimension in the expected way when passed to the model's `forward` pass.

Specifically, in `scripts/04_train_diffusion.py`:
- `x_t` is expected as `[B, 1, 96]`
- `c_in` as `[B, 9, 96]`
- `c_out` as `[B, 10, 96]`
- `t` as `[B]`

## Objective for Next Chat
1.  **Fix Tensor Shapes**: Ensure that the model's `forward` pass robustly handles both batched (training) and single-sample (inference) inputs.
2.  **Run Sanity Check**: Train for 2 epochs on 10 homes (`head(10)` in R scripts).
3.  **Generate Proof of Concept**: Use `scripts/05_generate_samples.py` to produce `synthetic_samples.png`.
4.  **Production Run**: Once verified, scale to 100 epochs on the full population.

## Key Files
- `scripts/03_export_to_parquet.py`: Data loading logic.
- `scripts/04_train_diffusion.py`: Model architecture and training loop.
- `scripts/05_generate_samples.py`: Sampling/Inference logic.

## Recommended Fix
In `DiffusionModel.forward`, add explicit `unsqueeze(0)` if `ndim` is 1 or 2 to guarantee a batch dimension before any `torch.cat` or `mean(dim=-1)` operations.
