#!/bin/bash
# Setup script for TNShap conda environment

set -e

echo "🚀 Setting up TNShap conda environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate environment
echo "🔄 Activating tnshap environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tnshap

# Install package in development mode
echo "🔧 Installing TNShap in development mode..."
pip install -e .

# Verify installation
echo "✅ Verifying installation..."
python -c "
import torch
import numpy as np
from src import BinaryTensorTree, FeatureMappedTN
print('✅ All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "🎉 Setup complete! To activate the environment in the future, run:"
echo "   conda activate tnshap"
echo ""
echo "📚 Next steps:"
echo "   1. Read the Quick Start Guide: docs/quickstart.md"
echo "   2. Run experiments: cd experiments/UCI && python scripts/eval_local_student_k123.py --help"
echo "   3. Check out the notebooks: jupyter notebook notebooks/"
