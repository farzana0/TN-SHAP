#!/usr/bin/env python3
"""
Create comprehensive visualizations for TN-tree Shapley values and interactions.

This script generates:
1. Heatmaps for pairwise interactions (order-2 Shapley values)  
2. Bar plots for single-token Shapley values (order-1)
3. Feature importance heatmaps (tokens × embedding dimensions)
4. Comparison plots across different masking patterns

Compatible with the eval_tn_shapley_orders.py output format.

Usage:
    python create_shapley_visualizations.py --result tn_shapley_eval/sentence_shapley_eval.json
    python create_shapley_visualizations.py --result-dir tn_shapley_eval/ --batch-mode
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from matplotlib.patches import Patch
from itertools import combinations

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

def ensure_dir(directory: str):
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(directory, exist_ok=True)

def load_shapley_results(results_path: str) -> Dict:
    """Load Shapley evaluation results from JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)

def create_pairwise_interaction_heatmap(results: Dict, output_dir: str, 
                                      mean_center: bool = True, 
                                      show_values: bool = True):
    """
    Create heatmap for pairwise Shapley interactions (order-2).
    
    Args:
        results: Shapley evaluation results
        output_dir: Output directory for plots
        mean_center: Whether to subtract mean from values  
        show_values: Whether to show values in heatmap cells
    """
    ensure_dir(output_dir)
    
    # Check if order-2 data exists
    if '2' not in results.get('shapley_values', {}):
        print("No order-2 Shapley values found for pairwise heatmap")
        return
    
    order_2_data = results['shapley_values']['2']
    pairwise_values = np.array(order_2_data['values'])
    tokens = results['tokens']
    sentence = results.get('sentence_text', 'Unknown sentence')
    
    print(f"Creating pairwise interaction heatmap for: {sentence}")
    print(f"  Found {len(pairwise_values)} pairwise values")
    
    # Create matrix for heatmap
    n_tokens = len(tokens)
    heatmap_matrix = np.zeros((n_tokens, n_tokens))
    
    # Fill matrix with pairwise values
    # Assuming pairwise values are ordered as combinations(range(n_tokens), 2)
    pair_idx = 0
    for i, j in combinations(range(n_tokens), 2):
        if pair_idx < len(pairwise_values):
            value = pairwise_values[pair_idx]
            heatmap_matrix[i, j] = value
            heatmap_matrix[j, i] = value  # Make symmetric
            pair_idx += 1
    
    # Mean centering
    original_values = pairwise_values.copy()
    if mean_center and len(pairwise_values) > 0:
        mean_val = np.mean(pairwise_values)
        heatmap_matrix = heatmap_matrix - mean_val
        heatmap_matrix[heatmap_matrix == -mean_val] = 0  # Re-zero empty cells
        print(f"  Mean-centered values (removed mean: {mean_val:.6f})")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Original heatmap
    mask1 = heatmap_matrix == 0
    np.fill_diagonal(mask1, True)  # Mask diagonal
    
    # Get color range
    non_zero = heatmap_matrix[~mask1]
    if len(non_zero) > 0:
        vmax = max(abs(non_zero.min()), abs(non_zero.max()))
        vmin = -vmax
    else:
        vmin, vmax = -1, 1
    
    # Heatmap 1: Full matrix
    sns.heatmap(heatmap_matrix, 
                annot=show_values, 
                fmt='.3f' if show_values else '',
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='RdBu_r',  # Red-blue: red=positive, blue=negative
                center=0,
                vmin=vmin,
                vmax=vmax,
                mask=mask1,
                square=True,
                ax=ax1,
                cbar_kws={'label': 'Pairwise Shapley Value'},
                linewidths=0.5)
    
    title_suffix = " (Mean-Centered)" if mean_center else ""
    ax1.set_title(f'Pairwise Token Interactions{title_suffix}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Token Index', fontsize=12)
    ax1.set_ylabel('Token Index', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Bar plot of all pairwise values  
    pair_labels = [f"{tokens[i]}-{tokens[j]}" for i, j in combinations(range(n_tokens), 2)]
    
    # Use mean-centered values if requested
    plot_values = heatmap_matrix[np.triu_indices(n_tokens, k=1)] if mean_center else original_values
    
    # Color by value sign
    colors = ['red' if val > 0 else 'blue' for val in plot_values]
    
    bars = ax2.bar(range(len(plot_values)), plot_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_title(f'Pairwise Interactions - Bar View{title_suffix}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Token Pairs', fontsize=12)
    ax2.set_ylabel('Shapley Value', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Rotate x-labels
    ax2.set_xticks(range(len(pair_labels)))
    ax2.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=10)
    
    # Add statistics text
    if len(plot_values) > 0:
        stats_text = f"Values: {len(plot_values)}\n"
        stats_text += f"Range: [{plot_values.min():.4f}, {plot_values.max():.4f}]\n"
        stats_text += f"Mean: {plot_values.mean():.4f}\n"
        stats_text += f"Std: {plot_values.std():.4f}"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=10)
    
    # Overall title
    safe_sentence = sentence.replace('"', '').replace("'", "")
    fig.suptitle(f'Pairwise Shapley Interactions\n"{safe_sentence}"', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save plot  
    safe_filename = safe_sentence.replace(' ', '_').replace(',', '').replace('.', '')[:50]
    suffix = "_mean_centered" if mean_center else ""
    output_path = os.path.join(output_dir, f'{safe_filename}_pairwise_interactions{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved pairwise interaction heatmap: {output_path}")
    
    # Print statistics
    if len(plot_values) > 0:
        print(f"  Pairwise values range: [{plot_values.min():.6f}, {plot_values.max():.6f}]")
        print(f"  Mean: {plot_values.mean():.6f}, Std: {plot_values.std():.6f}")
        print(f"  Mean absolute value: {np.abs(plot_values).mean():.6f}")

def create_single_token_barplot(results: Dict, output_dir: str):
    """Create bar plot for single-token Shapley values (order-1)"""
    ensure_dir(output_dir)
    
    # Check if order-1 data exists
    if '1' not in results.get('shapley_values', {}):
        print("No order-1 Shapley values found for single-token barplot")
        return
    
    order_1_data = results['shapley_values']['1']
    single_values = np.array(order_1_data['values'])
    tokens = results['tokens']
    sentence = results.get('sentence_text', 'Unknown sentence')
    
    print(f"Creating single-token barplot for: {sentence}")
    print(f"  Found {len(single_values)} single-token values")
    
    # If we have token-level Shapley values, use those instead
    if 'token_shapley' in order_1_data:
        token_shapley = order_1_data['token_shapley']
        # Use sum of features per token
        token_values = [token_shapley[token]['sum'] for token in tokens if token in token_shapley]
        if len(token_values) == len(tokens):
            single_values = np.array(token_values)
            print(f"  Using token-level aggregated values")
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Color by value sign and magnitude
    colors = []
    for val in single_values:
        if val > 0:
            colors.append('red')
        elif val < 0:
            colors.append('blue') 
        else:
            colors.append('gray')
    
    # Create bar plot
    bars = plt.bar(range(len(tokens)), single_values, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, single_values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 * np.sign(height) if height != 0 else 0.01),
                 f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                 fontsize=10, fontweight='bold')
    
    # Customize plot
    safe_sentence = sentence.replace('"', '').replace("'", "")
    plt.title(f'Single Token Shapley Values\n"{safe_sentence}"', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tokens', fontsize=12, fontweight='bold')
    plt.ylabel('Shapley Value', fontsize=12, fontweight='bold')
    
    # Set x-axis
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right', fontsize=11)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Positive'),
        Patch(facecolor='blue', alpha=0.7, label='Negative'),
        Patch(facecolor='gray', alpha=0.7, label='Zero')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"Values: {len(single_values)}\n"
    stats_text += f"Range: [{single_values.min():.4f}, {single_values.max():.4f}]\n"  
    stats_text += f"Mean: {single_values.mean():.4f}\n"
    stats_text += f"Sum: {single_values.sum():.4f}\n"
    stats_text += f"Positive: {np.sum(single_values > 0)}\n"
    stats_text += f"Negative: {np.sum(single_values < 0)}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    safe_filename = safe_sentence.replace(' ', '_').replace(',', '').replace('.', '')[:50]
    output_path = os.path.join(output_dir, f'{safe_filename}_single_token_barplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved single-token barplot: {output_path}")
    
    # Print statistics
    print(f"  Single-token values range: [{single_values.min():.6f}, {single_values.max():.6f}]")
    print(f"  Sum: {single_values.sum():.6f}, Mean: {single_values.mean():.6f}")
    print(f"  Positive values: {np.sum(single_values > 0)}, Negative: {np.sum(single_values < 0)}")

def create_masking_r2_comparison(results: Dict, output_dir: str):
    """Create visualization comparing R² scores across different masking patterns"""
    ensure_dir(output_dir)
    
    masking_eval = results.get('masking_evaluation', {})
    if not masking_eval:
        print("No masking evaluation data found")
        return
    
    sentence = results.get('sentence_text', 'Unknown sentence')
    print(f"Creating masking R² comparison for: {sentence}")
    
    # Extract R² values
    r2_data = {}
    
    # Full data
    if 'full_data' in masking_eval:
        r2_data['Full Data'] = masking_eval['full_data']['r2']
    
    # Single token masked
    if 'single_token_masked' in masking_eval:
        r2_data['Single Token\nMasked'] = masking_eval['single_token_masked']['r2_mean']
    
    # Pair tokens masked  
    if 'pair_tokens_masked' in masking_eval:
        r2_data['Pair Tokens\nMasked'] = masking_eval['pair_tokens_masked']['r2_mean']
    
    # Random masked
    if 'random_masked' in masking_eval:
        r2_data['Random\nMasked'] = masking_eval['random_masked']['r2_mean']
    
    # All masked (zero baseline)
    if 'all_masked' in masking_eval:
        r2_data['All Masked\n(Zero Baseline)'] = masking_eval['all_masked']['r2']
    
    if not r2_data:
        print("No R² data to plot")
        return
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    categories = list(r2_data.keys())
    values = list(r2_data.values())
    
    # Color based on performance (green=good, red=bad)
    colors = []
    for val in values:
        if val > 0.5:
            colors.append('green')
        elif val > 0:
            colors.append('orange')
        else:
            colors.append('red')
    
    bars = plt.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.05),
                 f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                 fontsize=12, fontweight='bold')
    
    # Customize plot
    safe_sentence = sentence.replace('"', '').replace("'", "")
    plt.title(f'R² Performance Across Masking Patterns\n"{safe_sentence}"', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('R² Score', fontsize=12, fontweight='bold')
    plt.xlabel('Masking Pattern', fontsize=12, fontweight='bold')
    
    # Add horizontal lines for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good Performance')
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Excellent Performance')
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Legend for performance levels
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='R² > 0.5 (Good)'),
        Patch(facecolor='orange', alpha=0.7, label='0 < R² < 0.5 (Fair)'),
        Patch(facecolor='red', alpha=0.7, label='R² < 0 (Poor)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    safe_filename = safe_sentence.replace(' ', '_').replace(',', '').replace('.', '')[:50]
    output_path = os.path.join(output_dir, f'{safe_filename}_masking_r2_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved masking R² comparison: {output_path}")
    print(f"  R² values: {r2_data}")

def create_comprehensive_shapley_summary(results: Dict, output_dir: str):
    """Create comprehensive summary dashboard with all key metrics"""
    ensure_dir(output_dir)
    
    sentence = results.get('sentence_text', 'Unknown sentence')
    tokens = results['tokens']
    print(f"Creating comprehensive summary for: {sentence}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Order-1 Shapley values (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    if '1' in results.get('shapley_values', {}):
        order_1_data = results['shapley_values']['1']
        values = np.array(order_1_data['values'])
        
        # Use token-aggregated values if available
        if 'token_shapley' in order_1_data and len(order_1_data['token_shapley']) == len(tokens):
            token_shapley = order_1_data['token_shapley']
            values = np.array([token_shapley[token]['sum'] for token in tokens])
        
        colors = ['red' if v > 0 else 'blue' for v in values]
        bars = ax1.bar(range(len(values)), values, color=colors, alpha=0.7)
        ax1.set_title('Order-1 Shapley Values', fontweight='bold')
        ax1.set_xticks(range(len(tokens)))
        ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        ax1.axhline(y=0, color='black', alpha=0.3)
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Order-2 heatmap (top-middle and top-right)
    ax2 = fig.add_subplot(gs[0, 1:])
    if '2' in results.get('shapley_values', {}):
        order_2_data = results['shapley_values']['2']
        pairwise_values = np.array(order_2_data['values'])
        
        # Create matrix
        n_tokens = len(tokens)
        heatmap_matrix = np.zeros((n_tokens, n_tokens))
        
        pair_idx = 0
        for i, j in combinations(range(n_tokens), 2):
            if pair_idx < len(pairwise_values):
                value = pairwise_values[pair_idx]
                heatmap_matrix[i, j] = value
                heatmap_matrix[j, i] = value
                pair_idx += 1
        
        mask = heatmap_matrix == 0
        np.fill_diagonal(mask, True)
        
        sns.heatmap(heatmap_matrix, annot=True, fmt='.3f',
                    xticklabels=tokens, yticklabels=tokens,
                    cmap='RdBu_r', center=0, mask=mask, square=True,
                    ax=ax2, cbar_kws={'label': 'Order-2 Shapley'})
        ax2.set_title('Order-2 Pairwise Interactions', fontweight='bold')
    
    # 3. R² comparison (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    masking_eval = results.get('masking_evaluation', {})
    if masking_eval:
        r2_data = {}
        if 'full_data' in masking_eval:
            r2_data['Full'] = masking_eval['full_data']['r2']
        if 'single_token_masked' in masking_eval:
            r2_data['Single\nMasked'] = masking_eval['single_token_masked']['r2_mean']
        if 'all_masked' in masking_eval:
            r2_data['Zero\nBaseline'] = masking_eval['all_masked']['r2']
        
        if r2_data:
            categories = list(r2_data.keys())
            values = list(r2_data.values())
            colors = ['green' if v > 0 else 'red' for v in values]
            
            bars = ax3.bar(categories, values, color=colors, alpha=0.7)
            ax3.set_title('R² Comparison', fontweight='bold')
            ax3.axhline(y=0, color='black', alpha=0.3)
            ax3.set_ylabel('R² Score')
            
            # Add values on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                         f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                         fontweight='bold')
    
    # 4. Statistics table (middle-center and middle-right)
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('off')
    
    # Collect statistics
    stats_text = f"SHAPLEY VALUE STATISTICS\n{'='*40}\n\n"
    
    # Order-1 stats
    if '1' in results.get('shapley_values', {}):
        order_1_stats = results['shapley_values']['1']['statistics']
        stats_text += f"ORDER-1 SHAPLEY VALUES:\n"
        stats_text += f"  Count: {results['shapley_values']['1']['n_values']}\n"
        stats_text += f"  Range: [{order_1_stats['min']:.6f}, {order_1_stats['max']:.6f}]\n"
        stats_text += f"  Mean: {order_1_stats['mean']:.6f}\n"
        stats_text += f"  Std: {order_1_stats['std']:.6f}\n\n"
    
    # Order-2 stats
    if '2' in results.get('shapley_values', {}):
        order_2_stats = results['shapley_values']['2']['statistics']
        stats_text += f"ORDER-2 SHAPLEY VALUES:\n"
        stats_text += f"  Count: {results['shapley_values']['2']['n_values']}\n"
        stats_text += f"  Range: [{order_2_stats['min']:.6f}, {order_2_stats['max']:.6f}]\n"
        stats_text += f"  Mean: {order_2_stats['mean']:.6f}\n"
        stats_text += f"  Std: {order_2_stats['std']:.6f}\n\n"
    
    # Training metrics
    training_metrics = results.get('training_metrics', {})
    if training_metrics:
        stats_text += f"TRAINING METRICS:\n"
        for key, value in training_metrics.items():
            if isinstance(value, float):
                stats_text += f"  {key}: {value:.4f}\n"
            else:
                stats_text += f"  {key}: {value}\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 5. Computation info (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Computation details
    comp_text = f"COMPUTATION DETAILS\n{'='*60}\n\n"
    comp_text += f"Sentence: \"{sentence}\"\n"
    comp_text += f"Tokens: {tokens}\n"
    comp_text += f"Number of tokens: {len(tokens)}\n\n"
    
    # Add computation times if available
    for order in ['1', '2']:
        if order in results.get('shapley_values', {}):
            comp_time = results['shapley_values'][order].get('computation_time', 0)
            comp_text += f"Order-{order} computation time: {comp_time:.2f} seconds\n"
    
    # Add evaluation config
    eval_config = results.get('evaluation_config', {})
    if eval_config:
        comp_text += f"\nEvaluation configuration:\n"
        for key, value in eval_config.items():
            comp_text += f"  {key}: {value}\n"
    
    ax5.text(0.05, 0.95, comp_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Overall title
    safe_sentence = sentence.replace('"', '').replace("'", "")
    fig.suptitle(f'TN-Tree Shapley Analysis Dashboard\n"{safe_sentence}"', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save plot
    safe_filename = safe_sentence.replace(' ', '_').replace(',', '').replace('.', '')[:50]
    output_path = os.path.join(output_dir, f'{safe_filename}_comprehensive_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive summary dashboard: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create Shapley value visualizations')
    parser.add_argument('--result', type=str, help='Path to single Shapley evaluation result JSON')
    parser.add_argument('--result-dir', type=str, help='Directory containing multiple result files')
    parser.add_argument('--output-dir', type=str, default='./shapley_plots', 
                        help='Output directory for plots')
    parser.add_argument('--batch-mode', action='store_true', 
                        help='Process all JSON files in result-dir')
    parser.add_argument('--mean-center', action='store_true', default=True,
                        help='Mean-center pairwise interactions')
    parser.add_argument('--show-values', action='store_true', default=True,
                        help='Show values in heatmap cells')
    
    args = parser.parse_args()
    
    ensure_dir(args.output_dir)
    
    # Collect result files
    result_files = []
    
    if args.batch_mode and args.result_dir:
        # Process all JSON files in directory
        for filename in os.listdir(args.result_dir):
            if filename.endswith('_shapley_eval.json'):
                result_files.append(os.path.join(args.result_dir, filename))
        print(f"Found {len(result_files)} result files in batch mode")
    
    elif args.result:
        # Process single file
        result_files.append(args.result)
    
    else:
        print("ERROR: Please specify either --result for single file or --result-dir with --batch-mode")
        return 1
    
    if not result_files:
        print("No result files found to process")
        return 1
    
    # Process each result file
    for result_file in result_files:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {result_file}")
        print(f"{'='*80}")
        
        try:
            # Load results
            results = load_shapley_results(result_file)
            
            # Create all visualizations
            print("\n1. Creating pairwise interaction heatmap...")
            create_pairwise_interaction_heatmap(results, args.output_dir, 
                                              mean_center=args.mean_center,
                                              show_values=args.show_values)
            
            print("\n2. Creating single-token barplot...")
            create_single_token_barplot(results, args.output_dir)
            
            print("\n3. Creating masking R² comparison...")
            create_masking_r2_comparison(results, args.output_dir)
            
            print("\n4. Creating comprehensive summary dashboard...")
            create_comprehensive_shapley_summary(results, args.output_dir)
            
            print(f"\n✅ Completed visualizations for: {result_file}")
            
        except Exception as e:
            print(f"❌ Error processing {result_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processed {len(result_files)} result file(s)")
    
    return 0

if __name__ == "__main__":
    exit(main())
