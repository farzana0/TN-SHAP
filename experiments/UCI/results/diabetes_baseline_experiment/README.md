# Diabetes Baseline Sweep Experiment

This directory contains the complete diabetes dataset baseline comparison experiment that evaluates TNShap against traditional baseline methods.

## Files

- `diabetes_baseline_sweep.py` - Main experiment script
- `diabetes_baseline_results.tex` - LaTeX formatted results table
- `results_diabetes_baseline_sweep/` - Complete experimental results
  - `detailed_diabetes_baseline_sweep.csv` - Detailed per-test-point results
  - `summary_diabetes_baseline_sweep.csv` - Aggregated summary statistics

## Experiment Overview

### Dataset
- **Source**: sklearn diabetes dataset
- **Size**: 442 samples, 10 features
- **Seed**: 2711
- **Pre-trained models**: Located in `out_local_student_singlegrid/diabetes_seed2711_K89_m10/`

### Models
- **Teacher**: MLPRegressor (pre-trained)
- **Student**: FeatureMappedTN with rank 32 (R² = 0.9888)

### Methods Compared
1. **TNShap** (reference baseline)
2. **KernelSHAP**
3. **SHAPIQ variants**:
   - regression_SII
   - regression_FSII  
   - permutation_SII
   - montecarlo_SII

### Experimental Parameters
- **Orders**: 1 (Shapley values), 2 (pairwise), 3 (3rd-order)
- **Budgets**: 50, 100, 200, 500, 1000, 1500, 2000, 10000
- **Test points**: 10 randomly selected from diabetes dataset
- **Metrics**: Runtime, MSE vs TNShap, Cosine similarity vs TNShap

## Key Results

### Order 1 (Shapley Values)
- **Winner**: KernelSHAP
- **Runtime**: 0.006-0.020 seconds
- **Accuracy**: Cosine similarity > 0.997
- **TNShap**: Competitive at 0.020s, cosine similarity 0.991

### Higher-Order Interactions (Orders 2-3)
- **TNShap**: Only method providing reliable higher-order calculations
- **TNShap Runtime**: 0.053s (order 2), 0.278s (order 3)
- **SHAPIQ methods**: 6-12x slower with poor accuracy for higher orders

### Budget Efficiency
- KernelSHAP optimal at low budgets (50-100)
- SHAPIQ methods don't improve significantly with larger budgets
- TNShap provides consistent performance independent of budget concept

## Usage

```bash
# Ensure you're in the concepts conda environment
conda activate concepts

# Run the experiment
python diabetes_baseline_sweep.py
```

## Dependencies
- torch
- numpy
- pandas
- sklearn
- shap
- shapiq
- Custom modules: tntree_model.py, feature_mapped_tn.py

## Results Location
Results are automatically saved to `results_diabetes_baseline_sweep/` with timestamp-based naming.

## Citation
This experiment demonstrates TNShap's unique capability for efficient higher-order Shapley interaction analysis compared to traditional baselines that excel only for first-order Shapley values.
