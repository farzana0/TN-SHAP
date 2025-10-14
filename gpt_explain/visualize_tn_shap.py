#!/usr/bin/env python3
"""
Visualization script for TN-SHAP values and interactions.
Creates separate heatmaps and bar plots for Shapley values and interactions.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Configure matplotlib for better plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

def load_tn_shap_results(result_path):
    """Load TN-SHAP results from JSON file"""
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    return results

def create_shapley_bar_plot(results, output_dir):
    """Create bar plot of TN-SHAP values by token"""
    
    tokens = results['tokens']
    token_shapley = results['shapley_values']['by_token']
    tn_r2 = results['tn_student_performance']['general_r2']
    
    # Extract token-level Shapley values
    token_names = []
    token_values = []
    token_abs_values = []
    
    for token in tokens:
        if token in token_shapley:
            token_names.append(token)
            token_values.append(token_shapley[token]['sum'])  # Sum of features per token
            token_abs_values.append(token_shapley[token]['abs_mean'])  # Mean absolute importance
    
    if not token_values:
        print("Warning: No token Shapley values found for bar plot")
        return None
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Raw Shapley values
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in token_values]
    bars1 = ax1.bar(range(len(token_names)), token_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('')
    ax1.set_ylabel('TN-SHAP Value', fontweight='bold')
    ax1.set_title(f'TN-SHAP Values by Token (TN-Student R² = {tn_r2:.3f})', fontweight='bold', pad=15)
    ax1.set_xticks(range(len(token_names)))
    ax1.set_xticklabels(token_names, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, token_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Plot 2: Absolute importance
    bars2 = ax2.bar(range(len(token_names)), token_abs_values, color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Tokens', fontweight='bold')
    ax2.set_ylabel('Absolute TN-SHAP Importance', fontweight='bold')
    ax2.set_title('Absolute Token Importance', fontweight='bold', pad=15)
    ax2.set_xticks(range(len(token_names)))
    ax2.set_xticklabels(token_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, token_abs_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    safe_filename = results['sentence'].replace(' ', '_').replace(',', '_').replace('.', '').replace("'", "").replace('"', '')
    plot_path = os.path.join(output_dir, f'{safe_filename}_tn_shap_barplot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"TN-SHAP bar plot saved to: {plot_path}")
    return plot_path

def create_interaction_heatmap(results, output_dir):
    """Create heatmap of TN-SHAP interaction values"""
    
    tokens = results['tokens']
    interactions = results['interaction_values']
    tn_r2 = results['tn_student_performance']['general_r2']
    
    if not interactions:
        print("Warning: No interaction values found for heatmap")
        return None
    
    n_tokens = len(tokens)
    
    # Create interaction matrix
    interaction_matrix = np.zeros((n_tokens, n_tokens))
    
    for interaction_key, interaction_data in interactions.items():
        i = interaction_data['token_i']
        j = interaction_data['token_j']
        value = interaction_data['value']
        
        # Symmetric matrix
        interaction_matrix[i, j] = value
        interaction_matrix[j, i] = value
    
    # Create custom colormap (red-white-green)
    colors = ['#d62728', '#ffffff', '#2ca02c']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('interaction', colors, N=n_bins)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Determine color scale
    abs_max = max(abs(interaction_matrix.min()), abs(interaction_matrix.max()))
    if abs_max == 0:
        abs_max = 1.0
    
    # Create heatmap
    im = ax.imshow(interaction_matrix, cmap=cmap, aspect='equal', 
                   vmin=-abs_max, vmax=abs_max, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('TN-SHAP Interaction Strength', fontweight='bold', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(n_tokens))
    ax.set_yticks(range(n_tokens))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, n_tokens, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_tokens, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', size=0)
    
    # Add value annotations
    for i in range(n_tokens):
        for j in range(n_tokens):
            if i != j and interaction_matrix[i, j] != 0:  # Only show non-diagonal, non-zero values
                text_color = 'white' if abs(interaction_matrix[i, j]) > 0.5 * abs_max else 'black'
                ax.text(j, i, f'{interaction_matrix[i, j]:.3f}', 
                       ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')
    
    ax.set_title(f'TN-SHAP Token Interaction Heatmap (TN-Student R² = {tn_r2:.3f})', 
                fontweight='bold', pad=20)
    ax.set_xlabel('Tokens', fontweight='bold')
    ax.set_ylabel('Tokens', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    safe_filename = results['sentence'].replace(' ', '_').replace(',', '_').replace('.', '').replace("'", "").replace('"', '')
    plot_path = os.path.join(output_dir, f'{safe_filename}_tn_shap_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"TN-SHAP interaction heatmap saved to: {plot_path}")
    return plot_path

def create_combined_visualization(results, output_dir):
    """Create a combined visualization with both bar plot and heatmap"""
    
    tokens = results['tokens']
    token_shapley = results['shapley_values']['by_token']
    interactions = results['interaction_values']
    tn_r2 = results['tn_student_performance']['general_r2']
    sentence = results['sentence']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Subplot 1: Shapley values bar plot
    ax1 = plt.subplot(1, 2, 1)
    
    token_names = []
    token_values = []
    
    for token in tokens:
        if token in token_shapley:
            token_names.append(token)
            token_values.append(token_shapley[token]['sum'])
    
    if token_values:
        colors = ['#d62728' if v < 0 else '#2ca02c' for v in token_values]
        bars = ax1.bar(range(len(token_names)), token_values, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=0.5)
        
        ax1.set_ylabel('TN-SHAP Value', fontweight='bold')
        ax1.set_title('Token Shapley Values', fontweight='bold')
        ax1.set_xticks(range(len(token_names)))
        ax1.set_xticklabels(token_names, rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, val in zip(bars, token_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                    f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    # Subplot 2: Interaction heatmap
    ax2 = plt.subplot(1, 2, 2)
    
    if interactions:
        n_tokens = len(tokens)
        interaction_matrix = np.zeros((n_tokens, n_tokens))
        
        for interaction_key, interaction_data in interactions.items():
            i = interaction_data['token_i']
            j = interaction_data['token_j']
            value = interaction_data['value']
            
            interaction_matrix[i, j] = value
            interaction_matrix[j, i] = value
        
        # Custom colormap
        colors = ['#d62728', '#ffffff', '#2ca02c']
        cmap = LinearSegmentedColormap.from_list('interaction', colors, N=256)
        
        abs_max = max(abs(interaction_matrix.min()), abs(interaction_matrix.max()))
        if abs_max == 0:
            abs_max = 1.0
        
        im = ax2.imshow(interaction_matrix, cmap=cmap, aspect='equal', 
                       vmin=-abs_max, vmax=abs_max, interpolation='nearest')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Interaction', fontweight='bold', rotation=270, labelpad=15)
        
        ax2.set_xticks(range(n_tokens))
        ax2.set_yticks(range(n_tokens))
        ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        ax2.set_yticklabels(tokens, fontsize=9)
        
        # Add grid
        ax2.set_xticks(np.arange(-0.5, n_tokens, 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, n_tokens, 1), minor=True)
        ax2.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.tick_params(which='minor', size=0)
        
        ax2.set_title('Token Interactions', fontweight='bold')
    
    # Main title
    fig.suptitle(f'TN-SHAP Analysis: "{sentence}" (R² = {tn_r2:.3f})', 
                fontweight='bold', fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    safe_filename = sentence.replace(' ', '_').replace(',', '_').replace('.', '').replace("'", "").replace('"', '')
    plot_path = os.path.join(output_dir, f'{safe_filename}_tn_shap_combined.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Combined TN-SHAP visualization saved to: {plot_path}")
    return plot_path

def main():
    parser = argparse.ArgumentParser(description='Create TN-SHAP visualizations')
    parser.add_argument('--result', type=str, required=True, help='Path to TN-SHAP result JSON')
    parser.add_argument('--output-dir', type=str, default='./tn_shap_plots', help='Output directory')
    parser.add_argument('--plot-type', type=str, choices=['bar', 'heatmap', 'combined', 'all'], 
                       default='all', help='Type of plot to create')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("TN-SHAP VISUALIZATION")
    print("=" * 80)
    print(f"Results file: {args.result}")
    print(f"Output directory: {args.output_dir}")
    print(f"Plot type: {args.plot_type}")
    
    try:
        # Load results
        results = load_tn_shap_results(args.result)
        
        print(f"\nSentence: {results['sentence']}")
        print(f"TN-Student R²: {results['tn_student_performance']['general_r2']:.4f}")
        print(f"Tokens: {results['tokens']}")
        
        created_plots = []
        
        # Create requested plots
        if args.plot_type in ['bar', 'all']:
            plot_path = create_shapley_bar_plot(results, args.output_dir)
            if plot_path:
                created_plots.append(plot_path)
        
        if args.plot_type in ['heatmap', 'all']:
            plot_path = create_interaction_heatmap(results, args.output_dir)
            if plot_path:
                created_plots.append(plot_path)
        
        if args.plot_type in ['combined', 'all']:
            plot_path = create_combined_visualization(results, args.output_dir)
            if plot_path:
                created_plots.append(plot_path)
        
        print(f"\nVisualization complete! Created {len(created_plots)} plots:")
        for plot_path in created_plots:
            print(f"  {plot_path}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
