#!/bin/bash

# Setup script for Water Usage Diffusion Pipeline
# This script handles Python virtual environment, R dependencies, and directory setup.

set -e # Exit on error

echo "🚀 Starting environment setup..."

# 1. Directory Setup
echo "📁 Creating data directories..."
mkdir -p data/features data/windows data/parquet

# 2. Python Setup (using uv)
echo "🐍 Setting up Python virtual environment..."
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create venv and install dependencies
# Note: We specify python 3.10 or higher for compatibility
uv venv --python 3.10
source .venv/bin/activate

echo "📦 Installing Python packages..."
uv pip install \
    torch \
    pytorch-lightning \
    pyarrow \
    pandas \
    numpy \
    torchmetrics \
    matplotlib \
    "fsspec>=2023.6.0"

# 3. R Setup
echo "📊 Checking R dependencies..."
Rscript -e '
required_packages <- c("dplyr", "future", "future.apply", "arrow", "lubridate", "tidyr", "purrr", "zoo", "fs", "blastula", "stringr", "DBI", "duckdb")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if(length(missing_packages) > 0) {
    cat("Installing missing R packages:", paste(missing_packages, collapse=", "), "\n")
    install.packages(missing_packages, repos="https://cloud.r-project.org")
} else {
    cat("All R dependencies are already installed.\n")
}
'

echo "✅ Setup Complete!"
echo ""
echo "To start the pipeline, run:"
echo "source .venv/bin/activate"
echo "Rscript scripts/01_build_features.R && Rscript scripts/02_make_windows.R && python scripts/04_train_diffusion.py"
