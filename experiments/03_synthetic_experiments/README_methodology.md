# Teacher-Student Multi-Seed Experiment Design

## Overview

This experiment addresses a critical question in tensor network (TN) training: **How much does random initialization and SGD stochasticity affect student model performance and interpretability?**

## Motivation

When training student TN models to mimic teacher models, we observe variation in performance across different runs. This variation could be due to:

1. **Random Initialization**: Different starting parameters
2. **SGD Stochasticity**: Different optimization trajectories  
3. **Early Stopping**: Variable convergence points
4. **Numerical Precision**: Small differences in floating point operations

Understanding this variance is crucial for:
- **Reproducibility**: Ensuring consistent results
- **Model Selection**: Choosing appropriate hyperparameters
- **Interpretability**: Validating TNShap consistency
- **Benchmarking**: Fair comparison with baselines

## Experimental Design

### Fixed Components (Constant Across All Runs)
- **Teacher Models**: Same architecture and parameters
- **Test Points**: Same 15 test points for evaluation
- **Training Data Generation**: Same teacher evaluation points
- **Hyperparameters**: Learning rate, epochs, patience, etc.

### Variable Components (Changes Across Runs)
- **Student Initialization**: Different random seeds
- **SGD Trajectory**: Different optimization paths due to randomness

### Methodology
```
For each Teacher Type:
    For each Student Rank:
        For each of 10 Seeds:
            1. Initialize student TN with seed
            2. Train student to mimic teacher
            3. Evaluate TNShap consistency on test points
            4. Record training R² and TNShap R² (orders 1,2,3)
        Compute mean ± std across 10 runs
```

## Metrics

### Training Performance
- **Training R²**: How well student fits teacher on training data
- **Mean/Std**: Consistency of training across seeds

### Interpretability Preservation
- **TNShap R² Order 1**: Shapley value preservation (feature importance)
- **TNShap R² Order 2**: Pairwise interaction preservation
- **TNShap R² Order 3**: Three-way interaction preservation

### Statistical Analysis
- **Mean**: Expected performance
- **Standard Deviation**: Consistency measure
- **Coefficient of Variation**: Relative variability

## Expected Outcomes

### Hypothesis 1: Training Consistency
**Expectation**: Higher-rank students should have more consistent training due to increased model capacity.

**Rationale**: More parameters provide multiple paths to good solutions.

### Hypothesis 2: Interpretability Stability
**Expectation**: TNShap preservation should be most stable for Order 1 (Shapley values) and least stable for Order 3.

**Rationale**: Higher-order interactions are more sensitive to small parameter changes.

### Hypothesis 3: Teacher Type Effects
**Expectation**: TensorTree teachers should show more consistent student training than GenericMultilinear teachers.

**Rationale**: Same architecture between teacher and student provides better learning bias.

## Practical Implications

### Low Variance (Std < 0.01)
- **Interpretation**: Robust training process
- **Action**: Can use single-run results confidently
- **Benchmarking**: Direct comparison valid

### Medium Variance (0.01 ≤ Std < 0.05)
- **Interpretation**: Some sensitivity to initialization
- **Action**: Report mean ± std, consider multiple runs
- **Benchmarking**: Use statistical significance tests

### High Variance (Std ≥ 0.05)
- **Interpretation**: Unstable training process
- **Action**: Investigate hyperparameters, increase epochs, or use ensembles
- **Benchmarking**: Requires careful experimental design

## Reproducibility Protocol

### Seed Management
1. **Teacher Seeds**: Fixed (42, 43, 44) for reproducible teachers
2. **Student Seeds**: Generated with fixed seed (12345) for reproducible randomness
3. **Test Data**: Fixed seed (42) for consistent evaluation points
4. **Documentation**: All seeds recorded in results

### Hardware Considerations
- **GPU**: Results may vary slightly across different GPU architectures
- **CUDA**: Different CUDA versions may introduce minor numerical differences
- **System**: Hardware info automatically documented

### Verification Steps
1. Run experiment multiple times and verify seed consistency
2. Check that teacher outputs remain identical across runs
3. Validate that student performance variance is within expected ranges

## File Organization

```
results_teacher_student_multi_seed/
├── summary_YYYYMMDD_HHMMSS.csv      # Mean ± std statistics
├── detailed_YYYYMMDD_HHMMSS.csv     # Individual run results  
├── README_experiment.md             # Auto-generated run documentation
└── README_methodology.md            # This methodology file
```

## Comparison with Single-Seed Experiments

### Original Approach
- Single run per configuration
- No variance assessment
- Potential cherry-picking of "good" runs
- Unclear reproducibility

### Multi-Seed Approach  
- 10 runs per configuration
- Quantified uncertainty
- Statistically valid conclusions
- Full reproducibility documentation

## Statistical Interpretation

### Confidence Intervals
With 10 runs, we can construct approximate 95% confidence intervals:
```
CI = mean ± (1.96 * std / sqrt(10))
CI = mean ± (0.62 * std)
```

### Practical Significance
- **Training R²**: Differences > 0.02 likely practically significant
- **TNShap R²**: Differences > 0.05 likely practically significant for interpretability

### Reporting Standards
Always report:
1. Number of runs (n=10)
2. Mean ± standard deviation
3. Seeds used for full reproducibility
4. Hardware/software environment

---

**Document Version**: 1.0  
**Created**: October 7, 2025  
**Author**: Automated experimental framework
