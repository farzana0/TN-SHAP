#!/bin/bash
# Quick activation script for TNShap environment

echo "🔄 Activating TNShap environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    # Initialize conda for this shell
    source $(conda info --base)/etc/profile.d/conda.sh
    
    # Activate the environment
    conda activate tnshap
    
    if [ $? -eq 0 ]; then
        echo "✅ TNShap environment activated successfully!"
        echo ""
        echo "📚 Quick commands:"
        echo "  python test_imports.py          # Test installation"
        echo "  cd experiments/01_feature_maps  # Run experiments"
        echo "  jupyter notebook notebooks/     # Open notebooks"
        echo ""
        echo "🔧 To deactivate: conda deactivate"
    else
        echo "❌ Failed to activate tnshap environment."
        echo "   Try running: conda env create -f environment.yml"
        exit 1
    fi
else
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    echo "   Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
