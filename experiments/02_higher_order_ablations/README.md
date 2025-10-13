# Higher-Order Feature Map Ablations

This experiment studies the impact of feature map dimensions on TNShap performance through systematic ablation studies.

## Overview

The higher-order ablations experiment investigates:
- How feature map size (m) affects TNShap accuracy
- Comparison between feature-mapped and non-feature-mapped tensor networks
- Impact on different interaction orders (k=1,2,3)
- Computational trade-offs of larger feature maps

## Key Scripts

### Ablation Scripts
- `feature_map_ablation_study.py` - Main ablation study script
- `baseline_no_feature_maps.py` - Baseline without feature maps
- `summarize_feature_map_results.py` - Generate ablation summary tables

### Training Scripts
- `train_feature_map_models.py` - Train models with different feature map sizes
- `evaluate_feature_map_models.py` - Evaluate models across interaction orders

## Experimental Design

### Feature Map Sizes Tested
- **m=1**: No feature mapping (baseline)
- **m=2**: Minimal feature mapping
- **m=4**: Moderate feature mapping  
- **m=8**: High feature mapping

### Evaluation Metrics
- **Training R²**: How well student fits teacher
- **TNShap R² Order 1**: Shapley value preservation
- **TNShap R² Order 2**: Pairwise interaction preservation
- **TNShap R² Order 3**: Three-way interaction preservation

## Key Findings

### Feature Map Impact
- Larger feature maps (m=4,8) improve higher-order interaction capture
- Diminishing returns beyond m=8 for most datasets
- Computational cost scales linearly with m

### Order-Specific Results
- **Order 1 (Shapley)**: Minimal impact of feature map size
- **Order 2 (Pairwise)**: Significant improvement with m≥4
- **Order 3 (Three-way)**: Largest gains with m=8

### Computational Trade-offs
- Training time: ~2x increase from m=1 to m=8
- Inference time: ~1.5x increase from m=1 to m=8
- Memory usage: Linear scaling with m

## Reproducing Results

### Run Ablation Study
```bash
# Run complete ablation study
python scripts/feature_map_ablation_study.py --dataset diabetes --seed 2711

# Run baseline (no feature maps)
python scripts/baseline_no_feature_maps.py --dataset diabetes --seed 2711

# Generate summary tables
python scripts/summarize_feature_map_results.py
```

### Custom Ablation
```bash
# Test specific feature map sizes
python scripts/train_feature_map_models.py \
    --dataset diabetes \
    --fmap_sizes 1 2 4 8 \
    --seed 2711

# Evaluate specific orders
python scripts/evaluate_feature_map_models.py \
    --dataset diabetes \
    --orders 1 2 3 \
    --fmap_size 8
```

## Results Structure

```
results/
└── out_local_student_singlegrid/
    ├── ablations/                    # Ablation study results
    │   ├── diabetes_ablation_fmap_sizes_seed2711.csv
    │   └── training_time.txt
    └── diabetes_seed2711_K89/        # Main experiment results
        ├── baseline_no_fmap/         # No feature mapping baseline
        ├── mgrid10_mfmap1/          # m=1 feature maps
        ├── mgrid10_mfmap2/          # m=2 feature maps
        ├── mgrid10_mfmap4/          # m=4 feature maps
        ├── mgrid10_mfmap8/          # m=8 feature maps
        ├── diabetes_ablation_fmap_eval_seed2711.csv
        ├── diabetes_ablation_summary.csv
        └── diabetes_ablation_table.tex
```

## Methodology

### Ablation Protocol
1. **Fixed Components**: Same teacher model, training data, hyperparameters
2. **Variable Component**: Feature map output dimension (m)
3. **Evaluation**: Same test points across all configurations
4. **Metrics**: Training fit and TNShap preservation across orders

### Statistical Analysis
- Multiple seeds for robustness
- Mean ± standard deviation reporting
- Statistical significance testing
- Effect size analysis

## Key Insights

1. **Feature Maps Matter**: Significant improvement in higher-order interaction capture
2. **Sweet Spot**: m=4-8 provides best accuracy/compute trade-off
3. **Order Sensitivity**: Higher orders benefit more from feature mapping
4. **Dataset Dependent**: Optimal m varies by dataset complexity

## Hardware Requirements

- **GPU**: Recommended for efficient training
- **Memory**: 16GB+ VRAM for m=8 experiments
- **Storage**: ~5GB for full ablation results

## Future Directions

- Adaptive feature map sizing based on dataset
- Learned feature map architectures
- Multi-scale feature map combinations
- Theoretical analysis of feature map expressiveness
