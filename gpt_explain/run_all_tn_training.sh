#!/usr/bin/env bash
set -euo pipefail

# Batch script to train TN-tree models on all GPT datasets and compute Shapley values
# Usage: ./run_all_tn_training.sh

echo "=============================================="
echo "TN-TREE BATCH TRAINING ON ALL GPT DATASETS"
echo "=============================================="

#!/bin/bash

# Configuration
RANK=4          # Reduced rank for GPU memory
SEED=42
MAX_EPOCHS=20   # Reduced epochs for faster execution
OUTPUT_DIR="./tn_results"
BASELINE="zero"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# List of datasets
DATASETS=(
    "sentence_1_dataset.json"  # The food was cheap, fresh, and tasty
    "sentence_2_dataset.json"  # The test was easy and simple
    "sentence_3_dataset.json"  # The product is not very reliable
    "sentence_4_dataset.json"  # Great, just what I needed
)

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "================================================"
    echo "Processing: $dataset"
    echo "================================================"
    
    if [ ! -f "$dataset" ]; then
        echo "ERROR: Dataset file $dataset not found!"
        continue
    fi
    
    # Run training and Shapley computation
    python train_tn_shapley.py \
        --dataset "$dataset" \
        --rank "$RANK" \
        --seed "$SEED" \
        --max-epochs "$MAX_EPOCHS" \
        --output-dir "$OUTPUT_DIR" \
        --baseline "$BASELINE"
    
    echo "Completed: $dataset"
    echo ""
done

echo "=============================================="
echo "BATCH PROCESSING COMPLETED"
echo "=============================================="

# Create summary
echo "Creating summary of results..."

SUMMARY_FILE="$OUTPUT_DIR/batch_summary.txt"
echo "TN-Tree Training Summary" > "$SUMMARY_FILE"
echo "========================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Rank: $RANK" >> "$SUMMARY_FILE"
echo "Seed: $SEED" >> "$SUMMARY_FILE"
echo "Max Epochs: $MAX_EPOCHS" >> "$SUMMARY_FILE"
echo "Baseline: $BASELINE" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Files created:" >> "$SUMMARY_FILE"
for result_file in "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.pt; do
    if [ -f "$result_file" ]; then
        file_size=$(du -h "$result_file" | cut -f1)
        echo "  $(basename "$result_file") - $file_size" >> "$SUMMARY_FILE"
    fi
done

echo "" >> "$SUMMARY_FILE"
echo "Total files created: $(find "$OUTPUT_DIR" -name "*.json" -o -name "*.pt" | wc -l)" >> "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "To analyze results:"
echo "  python analyze_tn_results.py --results-dir $OUTPUT_DIR"
