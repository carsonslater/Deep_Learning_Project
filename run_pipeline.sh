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
    echo "🗑️  FORCING RESTART: Clearing existing features, windows, and models..."
    rm -f data/features/*.parquet
    rm -f data/windows/*.parquet
    rm -f final_water_diffusion_model.ckpt
fi

# Check for production flag
if [[ $* == *"--full"* ]]; then
    MODE="full"
    EPOCHS=100
    echo "🌕 MODE: FULL PRODUCTION RUN (All meters, $EPOCHS epochs)"
else
    echo "🕒 MODE: SANITY CHECK ($LIMIT meters, $EPOCHS epochs)"
fi

# 1. Environment Setup
echo "--- Stage 1: Environment Setup ---"
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    chmod +x setup_env.sh
    source ./setup_env.sh
else
    echo "✅ Environment already exists."
fi

# 2. Data Processing (R - Features)
echo "--- Stage 2: Feature Engineering ---"
if [ "$(ls -A data/features/*.parquet 2>/dev/null)" ]; then
    echo "✅ Stage 2 skipped: data/features/ already contains computed files."
else
    if [[ $MODE == "full" ]]; then
        Rscript scripts/01_build_features.R
    else
        Rscript scripts/01_build_features.R --limit $LIMIT
    fi
fi

# 3. Data Processing (R - Windows)
echo "--- Stage 3: Window Extraction ---"
if [ "$(ls -A data/windows/*.parquet 2>/dev/null)" ]; then
    echo "✅ Stage 3 skipped: data/windows/ already contains computed files."
else
    if [[ $MODE == "full" ]]; then
        Rscript scripts/02_make_windows.R
    else
        Rscript scripts/02_make_windows.R --limit $LIMIT
    fi
fi

# 4. Training (Python)
echo "--- Stage 4: Training Diffusion Model ---"
if [ -f "final_water_diffusion_model.ckpt" ]; then
    echo "✅ Stage 4 skipped: Model checkpoint 'final_water_diffusion_model.ckpt' already exists."
else
    source .venv/bin/activate
    python3 scripts/04_train_diffusion.py --epochs $EPOCHS
fi

# 5. Inference (Python)
echo "--- Stage 5: Generating Synthetic Samples ---"
source .venv/bin/activate
python3 scripts/05_generate_samples.py

echo "✅ Pipeline Complete! Check 'synthetic_samples_poc.png' for the latest results."
