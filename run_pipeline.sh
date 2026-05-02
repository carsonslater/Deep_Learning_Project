#!/bin/bash

# Master Pipeline Script for Water Diffusion Project
# This script is "Smart": it automatically skips stages that have already been computed.

set -e

# Default Settings (Sanity Check)
MODE="sanity"
LIMIT=10
EPOCHS=2

# Check for force flag (must come before stage checks)
if [[ $* == *"--force"* ]]; then
    echo "[RESTART] FORCING RESTART: Clearing existing features, windows, models, and virtual environment..."
    rm -f data/features/*.parquet
    rm -f data/windows/*.parquet
    rm -f .stage2_done .stage3_done
    rm -f final_water_diffusion_model.ckpt
    rm -rf .venv
fi

# Check for production flag
if [[ $* == *"--full"* ]]; then
    MODE="full"
    EPOCHS=20
    echo "[MODE] FULL PRODUCTION RUN (All meters, $EPOCHS epochs)"
else
    echo "[MODE] SANITY CHECK ($LIMIT meters, $EPOCHS epochs)"
fi

# 1. Environment Setup
echo "--- Stage 1: Environment Setup ---"
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    chmod +x setup_env.sh
    source ./setup_env.sh
else
    echo "[DONE] Environment already exists."
fi

# 2. Data Processing (R - Features)
echo "--- Stage 2: Feature Engineering ---"
if [ -f ".stage2_done" ]; then
    echo "[DONE] Stage 2 skipped: .stage2_done marker found (stage completed successfully previously)."
else
    echo "[CLEAN] Cleaning up any partial files from Stage 2..."
    rm -f data/features/*.parquet
    
    if [[ $MODE == "full" ]]; then
        Rscript scripts/01_build_features.R
    else
        Rscript scripts/01_build_features.R --limit $LIMIT
    fi
    touch .stage2_done
fi

# 3. Data Processing (R - Windows)
echo "--- Stage 3: Window Extraction ---"
if [ -f ".stage3_done" ]; then
    echo "[DONE] Stage 3 skipped: .stage3_done marker found (stage completed successfully previously)."
else
    echo "[CLEAN] Cleaning up any partial files from Stage 3..."
    rm -f data/windows/*.parquet
    
    if [[ $MODE == "full" ]]; then
        Rscript scripts/02_make_windows.R
    else
        Rscript scripts/02_make_windows.R --limit $LIMIT
    fi
    touch .stage3_done
fi

# 4. Training (Python)
echo "--- Stage 4: Training Diffusion Model ---"
if [ -f "final_water_diffusion_model.ckpt" ]; then
    echo "[DONE] Stage 4 skipped: Model checkpoint 'final_water_diffusion_model.ckpt' already exists."
else
    source .venv/bin/activate
    # Prevent MPS from keeping an unbounded memory cache
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    python3 scripts/04_train_diffusion.py --epochs $EPOCHS
fi

# 5. Inference (Python)
echo "--- Stage 5: Generating Synthetic Samples ---"
source .venv/bin/activate
python3 scripts/05_generate_samples.py

echo "[DONE] Pipeline Complete! Check 'synthetic_samples_poc.png' for the latest results."
