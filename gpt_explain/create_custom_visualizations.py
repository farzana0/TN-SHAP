#!/usr/bin/env python3
"""
Create custom heatmap and bar plots for TN-SHAP results.

Generates:
1. Heatmap plots for pairwise interactions (one per sentence)
2. Bar plots for single-token Shapley values (one per sentence)

Created: September 25, 2025
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from matplotlib.patches import Patch

def ensure_dir(directory: str):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_tn_results(results_dir: str) -> Dict[str, Dict]:
    """Load all TN-SHAP results from the results directory"""
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith('_tn_shapley_zero.json'):
            filepath = os.path.join(results_dir, filename)
            
            # Extract sentence name from filename
            sentence_name = filename.replace('_tn_shapley_zero.json', '')
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                results[sentence_name] = data
                
    return results

def create_pairwise_heatmap(sentence_name: str, shapley_data: Dict, output_dir: str):
    """Create heatmap for pairwise Shapley interactions with mean removed"""
    ensure_dir(output_dir)
    
    if 'order_2' not in shapley_data:
        print(f"No pairwise data found for {sentence_name}")
        return
    
    pairwise_values = shapley_data['order_2']['values']
    tokens = shapley_data['metadata']['tokens']
    n_tokens = len(tokens)
    
    # Create symmetric matrix for heatmap
    heatmap_matrix = np.zeros((n_tokens, n_tokens))
    
    # Fill in pairwise values
    pairwise_array = []
    for pair_key, value in pairwise_values.items():
        i, j = map(int, pair_key.split('_'))
        heatmap_matrix[i, j] = value
        heatmap_matrix[j, i] = value  # Make symmetric
        pairwise_array.append(value)
    
    # Remove mean from pairwise values
    if len(pairwise_array) > 0:
        mean_pairwise = np.mean(pairwise_array)
        print(f"  Removing pairwise mean: {mean_pairwise:.4f}")
        heatmap_matrix = heatmap_matrix - mean_pairwise
        # Re-zero the diagonal and empty cells
        heatmap_matrix[heatmap_matrix == -mean_pairwise] = 0
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Use red-white colormap as requested, with range matched to data
    mask = heatmap_matrix == 0
    mask[np.diag_indices_from(mask)] = True  # Mask diagonal
    
    # Get the data range for color normalization
    non_zero_values = heatmap_matrix[~mask]
    if len(non_zero_values) > 0:
        vmin, vmax = non_zero_values.min(), non_zero_values.max()
        # Make symmetric around zero if needed
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -1, 1
    
    # Create heatmap with custom colormap
    sns.heatmap(heatmap_matrix, 
                annot=True, 
                fmt='.2f',
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='RdBu_r',  # Red-blue colormap (red=positive, blue=negative)
                center=0,
                vmin=vmin,
                vmax=vmax,
                mask=mask,
                square=True,
                cbar_kws={'label': 'Pairwise Shapley Value (mean-removed)'},
                linewidths=0.5)
    
    plt.title(f'Pairwise Shapley Interactions (Mean Removed)\n{sentence_name.replace("_", " ")}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Token Index', fontsize=12, fontweight='bold')
    plt.ylabel('Token Index', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{sentence_name}_pairwise_heatmap_mean_removed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved pairwise heatmap: {output_path}")
    
    # Print statistics
    if len(non_zero_values) > 0:
        print(f"  Pairwise values range (mean-removed): [{non_zero_values.min():.4f}, {non_zero_values.max():.4f}]")
        print(f"  Mean absolute value: {np.abs(non_zero_values).mean():.4f}")

def create_single_token_barplot(sentence_name: str, shapley_data: Dict, output_dir: str):
    """Create bar plot for single-token Shapley values"""
    
    if 'order_1' not in shapley_data:
        print(f"No single-token data found for {sentence_name}")
        return
    
    single_values = np.array(shapley_data['order_1']['values'])
    tokens = shapley_data['metadata']['tokens']
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Create color array - positive values in red, negative in blue
    colors = ['red' if val > 0 else 'blue' for val in single_values]
    
    # Create bar plot
    bars = plt.bar(range(len(tokens)), single_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Customize plot
    plt.title(f'Single Token Shapley Values\n{sentence_name.replace("_", " ")}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Token', fontsize=12, fontweight='bold')
    plt.ylabel('Shapley Value', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, single_values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 * np.sign(height) if height != 0 else 0.01),
                f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
    
    # Add grid for better readability
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add legend
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Positive contribution'),
                      Patch(facecolor='blue', alpha=0.7, label='Negative contribution')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{sentence_name}_single_token_barplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved single-token bar plot: {output_path}")
    
    # Print statistics
    print(f"  Single-token values range: [{single_values.min():.4f}, {single_values.max():.4f}]")
    print(f"  Mean: {single_values.mean():.4f}, Std: {single_values.std():.4f}")
    print(f"  Positive contributions: {np.sum(single_values > 0)}/{len(single_values)}")

def create_combined_comparison_plot(results: Dict[str, Dict], output_dir: str):
    """Create comparison plot showing all sentences together"""
    
    # Collect all single-token values
    all_data = []
    
    for sentence_name, shapley_data in results.items():
        if 'order_1' in shapley_data:
            single_values = np.array(shapley_data['order_1']['values'])
            tokens = shapley_data['metadata']['tokens']
            
            for i, (token, value) in enumerate(zip(tokens, single_values)):
                all_data.append({
                    'sentence': sentence_name.replace('_', ' '),
                    'token': token,
                    'token_idx': i,
                    'shapley_value': value
                })
    
    if not all_data:
        print("No data found for comparison plot")
        return
    
    df = pd.DataFrame(all_data)
    
    # Create grouped bar plot
    plt.figure(figsize=(18, 10))
    
    # Create subplot for each sentence
    sentences = df['sentence'].unique()
    n_sentences = len(sentences)
    
    fig, axes = plt.subplots(n_sentences, 1, figsize=(16, 6 * n_sentences), sharex=False)
    if n_sentences == 1:
        axes = [axes]
    
    for i, sentence in enumerate(sentences):
        sentence_data = df[df['sentence'] == sentence]
        
        # Create colors based on values
        colors = ['red' if val > 0 else 'blue' for val in sentence_data['shapley_value']]
        
        bars = axes[i].bar(sentence_data['token'], sentence_data['shapley_value'], 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        axes[i].set_title(f'{sentence}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Shapley Value', fontsize=12, fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        axes[i].grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sentence_data['shapley_value']):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., 
                        height + (0.01 * np.sign(height) if height != 0 else 0.01),
                        f'{value:.2f}', ha='center', 
                        va='bottom' if height >= 0 else 'top', 
                        fontweight='bold', fontsize=9)
    
    plt.suptitle('Single Token Shapley Values Comparison Across All Sentences', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save plot
    output_path = os.path.join(output_dir, 'all_sentences_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create custom heatmap and bar plots for TN-SHAP results')
    parser.add_argument('--results-dir', type=str, default='./tn_results',
                       help='Directory containing TN-SHAP results')
    parser.add_argument('--output-dir', type=str, default='./custom_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CUSTOM TN-SHAP VISUALIZATIONS")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_tn_results(args.results_dir)
    
    if not results:
        print("No TN-SHAP results found!")
        return
    
    print(f"Found {len(results)} result files:")
    for sentence_name in results.keys():
        print(f"  - {sentence_name}")
    
    # Create plots for each sentence
    print("\n" + "="*50)
    print("CREATING INDIVIDUAL PLOTS")
    print("="*50)
    
    for sentence_name, shapley_data in results.items():
        print(f"\nProcessing {sentence_name}...")
        
        # Create pairwise heatmap
        create_pairwise_heatmap(sentence_name, shapley_data, args.output_dir)
        
        # Create single-token bar plot
        create_single_token_barplot(sentence_name, shapley_data, args.output_dir)
    
    # Create comparison plot
    print("\n" + "="*50)
    print("CREATING COMPARISON PLOT")
    print("="*50)
    create_combined_comparison_plot(results, args.output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"All plots saved to: {args.output_dir}")
    
    # List generated files
    plot_files = [f for f in os.listdir(args.output_dir) if f.endswith('.png')]
    print(f"Generated {len(plot_files)} plot files:")
    for filename in sorted(plot_files):
        print(f"  - {filename}")

if __name__ == "__main__":
    main()
