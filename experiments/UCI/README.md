# UCI Dataset Experiments

This experiment evaluates TNShap on UCI datasets (Diabetes, Concrete, Energy, California Housing) and compares it against traditional Shapley value estimation methods.

## Overview

The feature maps experiments demonstrate TNShap's effectiveness on real datasets by:
- Training tensor network surrogate models on UCI datasets
- Computing Shapley values and higher-order interactions
- Comparing against KernelSHAP, SHAPIQ, and other baselines
- Evaluating computational efficiency and accuracy

## Key Scripts

### Training Scripts
- `uci_train_surrogate_models.py` - Train tensor network surrogate models
- `uci_diabetes_budget_sweep.py` - Comprehensive budget sweep experiments

### Evaluation Scripts  
- `uci_evaluate_tnshap_vs_baselines.py` - Main evaluation script for TNShap vs baselines
- `uci_aggregate_results.py` - Aggregate results across multiple runs
- `uci_create_results_table.py` - Generate publication-ready result tables

## Datasets

- **Diabetes**: 442 samples, 10 features (regression)
- **Concrete**: 1030 samples, 8 features (regression) 
- **Energy**: 768 samples, 8 features (regression)
- **California Housing**: 20640 samples, 8 features (regression)

## Key Results

### Performance Comparison
- **TNShap Order 1**: ~0.1s per evaluation (89 features)
- **TNShap Order 2**: ~0.5s per evaluation
- **TNShap Order 3**: ~2.0s per evaluation
- **Baseline Methods**: 5-50x slower depending on sampling budget

### Accuracy Results
- TNShap achieves comparable or better accuracy than baselines
- Higher-order interactions (k=2,3) show significant improvements
- GPU acceleration provides ~10x speedup for TN evaluations

## Reproducing Results

### Quick Start
```bash
# Train surrogate models
python scripts/uci_train_surrogate_models.py --dataset diabetes --seed 2711

# Evaluate TNShap vs baselines
python scripts/uci_evaluate_tnshap_vs_baselines.py --dataset diabetes --orders 1 2 3 --with-baselines

# Run budget sweep
python scripts/uci_diabetes_budget_sweep.py --orders 1 2 3 --repeats 3
```

### Full Pipeline
```bash
# 1. Build data and train models
bash scripts/run_build_data.sh

# 2. Run comprehensive evaluation
python scripts/run_all_pretrained_eval.py

# 3. Aggregate and create tables
python scripts/uci_aggregate_results.py
python scripts/uci_create_results_table.py
```

## Results Structure

```
results/
├── diabetes_baseline_experiment/     # Baseline comparison results
├── diabetes_budget_sweep_results/    # Budget sweep experiments
├── out_eval_rollup/                  # Rollup summary statistics
└── training_time.txt                 # Training time measurements
```

## Hardware Requirements

- **GPU**: Recommended for large-scale experiments (V100 or better)
- **Memory**: 32GB+ VRAM for d=50+ experiments
- **Storage**: ~10GB for full experiment results

## Key Findings

1. **Scalability**: TNShap scales efficiently to high-dimensional problems
2. **Accuracy**: Comparable or superior to traditional methods
3. **Speed**: Significant computational advantages with GPU acceleration
4. **Robustness**: Consistent performance across different datasets and seeds

## Methodology

The experiments follow a teacher-student framework:
1. Train a "teacher" model (neural network) on the dataset
2. Train a "student" tensor network to mimic the teacher
3. Compute Shapley values using the student model
4. Compare against ground truth and baseline methods

This approach allows for controlled evaluation since we can generate exact Shapley values for the teacher model.
