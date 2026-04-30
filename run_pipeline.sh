#!/bin/bash

# Master Pipeline Script for Water Diffusion Project
# Usage:
#   ./run_pipeline.sh          # Runs sanity check (10 meters, 2 epochs)
#   ./run_pipeline.sh --full   # Runs full production (all meters, 100 epochs)

set -e

# Default Settings (Sanity Check)
MODE="sanity"
LIMIT=10
EPOCHS=2

# Check for production flag
if [[ $1 == "--full" ]]; then
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
    echo "Environment already exists."
fi

# 2. Data Processing (R)
echo "--- Stage 2: Feature Engineering (R) ---"
if [[ $MODE == "full" ]]; then
    Rscript scripts/01_build_features.R
    echo "--- Stage 3: Window Extraction (R) ---"
    Rscript scripts/02_make_windows.R
else
    Rscript scripts/01_build_features.R --limit $LIMIT
    echo "--- Stage 3: Window Extraction (R) ---"
    Rscript scripts/02_make_windows.R --limit $LIMIT
fi

# 3. Training & Inference (Python)
echo "--- Stage 4: Training Diffusion Model (MPS) ---"
source .venv/bin/activate
python3 scripts/04_train_diffusion.py --epochs $EPOCHS

echo "--- Stage 5: Generating Synthetic Samples ---"
python3 scripts/05_generate_samples.py

echo "✅ Pipeline Complete! Check 'synthetic_samples_poc.png' for the latest results."
