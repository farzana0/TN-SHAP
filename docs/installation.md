# Installation Guide

This guide provides detailed instructions for installing TNShap and its dependencies.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **GPU**: Optional but recommended for large-scale experiments

### Hardware Recommendations
- **GPU**: 8GB+ VRAM (GTX 1080, RTX 3080, V100, A100)
- **Memory**: 32GB+ RAM
- **Storage**: 50GB+ free space

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/tnshap.git
cd tnshap

# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Method 2: conda (Recommended)

#### Quick Setup
```bash
# Clone the repository
git clone https://github.com/your-org/tnshap.git
cd tnshap

# Run automated setup script
./setup_conda.sh
```

#### Manual Setup
```bash
# Clone the repository
git clone https://github.com/your-org/tnshap.git
cd tnshap

# Create conda environment
conda env create -f environment.yml
conda activate tnshap

# Install in development mode
pip install -e .
```

#### Verify Installation
```bash
# Activate environment
conda activate tnshap

# Test imports
python -c "
import torch
from src import BinaryTensorTree, FeatureMappedTN
print('✅ Installation successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Method 3: Docker

```bash
# Build Docker image
docker build -t tnshap .

# Run container
docker run -it --gpus all tnshap
```

## GPU Support

### CUDA Installation

For GPU acceleration, you'll need CUDA-compatible PyTorch:

```bash
# For CUDA 11.3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU Installation

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Dependencies

### Core Dependencies
- **torch**: Deep learning framework
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities
- **scipy**: Scientific computing

### Shapley Value Computation
- **shapiq**: Exact Shapley value computation
- **shap**: SHAP library for baselines

### Visualization
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **jupyter**: Interactive notebooks

### Development
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in experiments
export CUDA_VISIBLE_DEVICES=0
python scripts/experiment.py --batch_size 32
```

#### 2. Import Errors
```bash
# Ensure you're in the correct environment
conda activate tnshap
# or
source venv/bin/activate

# Reinstall if needed
pip install -e . --force-reinstall
```

#### 3. Missing Dependencies
```bash
# Update pip
pip install --upgrade pip

# Install missing packages
pip install -r requirements.txt --upgrade
```

#### 4. Version Conflicts
```bash
# Create fresh environment
conda create -n tnshap-fresh python=3.10
conda activate tnshap-fresh
pip install -r requirements.txt
```

### Platform-Specific Issues

#### Windows
- Use Anaconda instead of pip for better compatibility
- Install Visual Studio Build Tools for C++ extensions

#### macOS
- Use Homebrew for system dependencies
- May need to install Xcode command line tools

#### Linux
- Install system dependencies: `sudo apt-get install build-essential`
- For CUDA: follow NVIDIA's installation guide

## Development Installation

For contributing to TNShap:

```bash
# Clone repository
git clone https://github.com/your-org/tnshap.git
cd tnshap

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ experiments/

# Lint code
flake8 src/ experiments/
```

## Verification

Test your installation:

```python
# Test basic imports
import torch
import numpy as np
from src import BinaryTensorTree, FeatureMappedTN

# Test tensor network creation
tn = BinaryTensorTree(
    leaf_phys_dims=[2, 2, 2],
    ranks=3,
    seed=42
)

# Test forward pass
X = torch.randn(10, 3)
output = tn(X)
print(f"Output shape: {output.shape}")

# Test GPU if available
if torch.cuda.is_available():
    tn_gpu = tn.cuda()
    X_gpu = X.cuda()
    output_gpu = tn_gpu(X_gpu)
    print(f"GPU output shape: {output_gpu.shape}")
```

## Getting Help

If you encounter issues:

1. Check the [FAQ](faq.md)
2. Search [GitHub Issues](https://github.com/your-org/tnshap/issues)
3. Create a new issue with:
   - Operating system and version
   - Python version
   - Error message and traceback
   - Steps to reproduce

## Next Steps

After installation:

1. Read the [Quick Start Guide](quickstart.md)
2. Explore the [experiments](experiments/)
3. Check out the [API documentation](api.md)
4. Run the [tutorial notebooks](notebooks/)
