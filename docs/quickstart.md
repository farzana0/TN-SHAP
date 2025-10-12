# Quick Start Guide

This guide will get you up and running with TNShap in minutes.

## Basic Usage

### 1. Create a Simple Tensor Network

```python
import torch
from src import BinaryTensorTree

# Create a tensor network for 5 features
tn = BinaryTensorTree(
    leaf_phys_dims=[2] * 5,  # 2D input per feature (value + bias)
    ranks=4,                  # Tensor network rank
    seed=42
)

# Generate test data
X = torch.randn(100, 5)  # 100 samples, 5 features

# Forward pass
predictions = tn(X)
print(f"Predictions shape: {predictions.shape}")  # [100, 1]
```

### 2. Feature-Mapped Tensor Network

```python
from src import make_feature_mapped_tn

# Create feature-mapped tensor network
model = make_feature_mapped_tn(
    d_in=10,           # Number of features
    fmap_out_dim=4,    # Feature map output channels
    ranks=6,           # Tensor network rank
    seed=42
)

# Test data
X = torch.randn(50, 10)

# Forward pass
output = model(X)
print(f"Output shape: {output.shape}")  # [50, 1]
```

### 3. Training a Surrogate Model

```python
import torch.nn as nn
import torch.optim as optim

# Create model
model = make_feature_mapped_tn(d_in=8, fmap_out_dim=2, ranks=4)

# Create some training data (simulate a teacher model)
X_train = torch.randn(1000, 8)
y_train = torch.randn(1000, 1)  # Teacher predictions

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test the trained model
model.eval()
with torch.no_grad():
    X_test = torch.randn(100, 8)
    test_predictions = model(X_test)
    print(f"Test predictions shape: {test_predictions.shape}")
```

## Running Experiments

### Activate Environment First

```bash
# Activate the conda environment
conda activate tnshap

# Or if using pip/venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 1. UCI Dataset Experiment

```bash
cd experiments/UCI

# Train a surrogate model
python scripts/build_local_student_singlegrid.py \
    --dataset diabetes \
    --seed 2711

# Evaluate TNShap vs baselines
python scripts/eval_local_student_k123.py \
    --dataset diabetes \
    --orders 1 2 3 \
    --with-baselines
```

### 2. Synthetic Validation

```bash
cd experiments/03_synthetic_experiments

# Run rank sweep experiment
python scripts/teacher_student_rank_sweep.py \
    --ranks 2 4 6 8 \
    --seed 42

# Multi-seed robustness study
python scripts/teacher_student_rank_sweep_multi_seed.py \
    --n_seeds 10
```

### 3. Scaling Experiment

```bash
cd experiments/04_scaling

# Run dimension sweep
python scripts/run_d_sweep_gpu.py \
    --dimensions 10 20 30 40

# Simple example
python scripts/simple_tn_shapley_example.py
```

## Understanding the Results

### 1. Performance Metrics

TNShap experiments typically report several metrics:

- **Training R²**: How well the student model fits the teacher
- **TNShap R² Order 1**: Accuracy of Shapley value estimation
- **TNShap R² Order 2**: Accuracy of pairwise interactions
- **TNShap R² Order 3**: Accuracy of three-way interactions

### 2. Example Output

```
Dataset: diabetes, Seed: 2711
Training R²: 0.987 ± 0.003
TNShap R² Order 1: 0.923 ± 0.012
TNShap R² Order 2: 0.856 ± 0.018
TNShap R² Order 3: 0.789 ± 0.025
Runtime: 0.15s per evaluation
```

### 3. Interpreting Results

- **R² > 0.9**: Excellent accuracy
- **R² > 0.8**: Good accuracy
- **R² > 0.7**: Acceptable accuracy
- **R² < 0.7**: May need parameter tuning

## Customizing Experiments

### 1. Modify Hyperparameters

```python
# In experiment scripts, you can modify:
model = make_feature_mapped_tn(
    d_in=10,
    fmap_out_dim=4,    # Try 1, 2, 4, 8
    ranks=6,           # Try 2, 4, 6, 8, 10
    fmap_hidden=32,    # Try 16, 32, 64
    fmap_act="relu"    # Try "tanh", "relu", "gelu"
)
```

### 2. Custom Datasets

```python
# Load your own data
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("your_data.csv")
X = data.drop("target", axis=1).values
y = data["target"].values

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to tensors
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

# Train model
model = make_feature_mapped_tn(
    d_in=X.shape[1],
    fmap_out_dim=4,
    ranks=6
)
```

### 3. Custom Architectures

```python
from src import BinaryTensorTree, FeatureMappedTN

# Create custom tensor network
tn = BinaryTensorTree(
    leaf_phys_dims=[3] * 10,  # 3D input per feature
    ranks={0: (4, 6), 1: (6, 8)},  # Per-node ranks
    out_dim=1
)

# Wrap with feature mapping
model = FeatureMappedTN(
    tn=tn,
    d_in=10,
    fmap_out_dim=2,
    fmap_hidden=64,
    fmap_act="gelu"
)
```

## GPU Acceleration

### 1. Enable GPU

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Move model to GPU
model = model.to(device)
X = X.to(device)
```

### 2. Batch Processing

```python
# Process data in batches for memory efficiency
def evaluate_batch(model, X, batch_size=1000):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch_pred = model(batch)
            predictions.append(batch_pred)
    
    return torch.cat(predictions, dim=0)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or model rank
2. **Slow Training**: Use GPU acceleration
3. **Poor Accuracy**: Increase model rank or training epochs
4. **Import Errors**: Check installation and environment

### Getting Help

- Check the [FAQ](faq.md)
- Search [GitHub Issues](https://github.com/your-org/tnshap/issues)
- Read the [API documentation](api.md)

## Next Steps

1. **Explore Experiments**: Run the full experiment suite
2. **Read Methodology**: Understand the theory behind TNShap
3. **Customize**: Adapt TNShap for your specific use case
4. **Contribute**: Help improve TNShap by contributing code or documentation

## Example Notebooks

Check out the Jupyter notebooks in the `notebooks/` directory for interactive examples:

- `basic_usage.ipynb`: Introduction to TNShap
- `experiment_analysis.ipynb`: Analyzing experiment results
- `custom_datasets.ipynb`: Working with your own data
- `performance_benchmarks.ipynb`: Comparing different configurations
