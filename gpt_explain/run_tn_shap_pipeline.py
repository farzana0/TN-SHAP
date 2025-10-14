#!/usr/bin/env python3
"""
Complete TN-SHAP pipeline:
1. Generate/use existing dataset
2. Train TN-tree with heavy mask injection
3. Compute TN-SHAP values and interactions (no baselines, no KernelSHAP)
4. Create heatmaps and bar plots
"""

import os
import sys
import argparse
import subprocess
import time

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time:.1f} seconds")
    
    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
    else:
        print("❌ FAILED")
        if result.stderr:
            print("Error:")
            print(result.stderr)
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Complete TN-SHAP pipeline')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset JSON')
    parser.add_argument('--rank', type=int, default=8, help='TN-tree rank')
    parser.add_argument('--mask-prob', type=float, default=0.8, help='Mask injection probability')
    parser.add_argument('--pairwise-multiplier', type=int, default=5, help='Pairwise mask injection multiplier')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--output-base', type=str, default='./tn_shap_pipeline', help='Base output directory')
    
    args = parser.parse_args()
    
    # Create output directories
    train_output = os.path.join(args.output_base, 'training_results')
    shap_output = os.path.join(args.output_base, 'shap_results')
    plot_output = os.path.join(args.output_base, 'visualizations')
    
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(shap_output, exist_ok=True)
    os.makedirs(plot_output, exist_ok=True)
    
    print("="*80)
    print("TN-SHAP COMPLETE PIPELINE")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"TN-tree rank: {args.rank}")
    print(f"Mask probability: {args.mask_prob}")
    print(f"Pairwise multiplier: {args.pairwise_multiplier}x")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Output base: {args.output_base}")
    
    # Step 1: Train TN-tree with mask injection
    train_cmd = f"""python train_tn_with_mask_injection.py \\
        --dataset "{args.dataset}" \\
        --output-dir "{train_output}" \\
        --rank {args.rank} \\
        --mask-probability {args.mask_prob} \\
        --pairwise-multiplier {args.pairwise_multiplier} \\
        --max-epochs {args.max_epochs} \\
        --patience 20"""
    
    if not run_command(train_cmd, "Training TN-tree with mask injection"):
        print("Training failed. Exiting.")
        return False
    
    # Find the training result file
    import glob
    result_files = glob.glob(os.path.join(train_output, '*_tn_masked_results.json'))
    
    if not result_files:
        print("❌ No training result files found!")
        return False
    
    training_result = result_files[0]  # Use the first (should be only) result file
    print(f"Found training result: {training_result}")
    
    # Step 2: Compute TN-SHAP values and interactions
    shap_cmd = f"""python compute_tn_shap_only.py \\
        --result "{training_result}" \\
        --output-dir "{shap_output}" \\
        --max-pairs 20"""
    
    if not run_command(shap_cmd, "Computing TN-SHAP values and interactions"):
        print("TN-SHAP computation failed. Exiting.")
        return False
    
    # Find the TN-SHAP result file
    shap_files = glob.glob(os.path.join(shap_output, '*_tn_shap_results.json'))
    
    if not shap_files:
        print("❌ No TN-SHAP result files found!")
        return False
    
    shap_result = shap_files[0]
    print(f"Found TN-SHAP result: {shap_result}")
    
    # Step 3: Create visualizations
    viz_cmd = f"""python visualize_tn_shap.py \\
        --result "{shap_result}" \\
        --output-dir "{plot_output}" \\
        --plot-type all"""
    
    if not run_command(viz_cmd, "Creating TN-SHAP visualizations"):
        print("Visualization failed. Exiting.")
        return False
    
    # Success summary
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Load and display key results
    try:
        import json
        
        with open(training_result, 'r') as f:
            train_results = json.load(f)
        
        with open(shap_result, 'r') as f:
            shap_results = json.load(f)
        
        print(f"\nSentence: {train_results['sentence_text']}")
        print(f"Tokens: {train_results['tokens']}")
        print(f"Training epochs: {train_results['final_metrics']['epochs_trained']}")
        print(f"Final validation R²: {train_results['final_metrics']['val_r2']:.4f}")
        print(f"TN-student general R²: {shap_results['tn_student_performance']['general_r2']:.4f}")
        
        # Shapley value summary
        shap_stats = shap_results['shapley_values']['statistics']
        print(f"\nTN-SHAP values:")
        print(f"  Range: [{shap_stats['min']:.4f}, {shap_stats['max']:.4f}]")
        print(f"  Mean: {shap_stats['mean']:.4f}")
        
        # Interaction summary
        n_interactions = len(shap_results['interaction_values'])
        print(f"\nTN-SHAP interactions: {n_interactions} token pairs computed")
        
        print(f"\nFiles created:")
        print(f"  Training results: {train_output}")
        print(f"  TN-SHAP results: {shap_output}")
        print(f"  Visualizations: {plot_output}")
        
        # List visualization files
        viz_files = glob.glob(os.path.join(plot_output, '*.png'))
        if viz_files:
            print(f"  Generated plots:")
            for viz_file in sorted(viz_files):
                print(f"    {os.path.basename(viz_file)}")
        
    except Exception as e:
        print(f"Could not load results for summary: {e}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
