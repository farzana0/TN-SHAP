#!/usr/bin/env python3
"""
Analyze TN-SHAP results from trained models.
Visualize and compare Shapley values across different sentences.

Created: September 25, 2025
"""

import os
import json
import argparse
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_tn_results(results_dir: str) -> Dict:
    """Load all TN-SHAP results from directory"""
    results = {}
    
    # Find all JSON results files
    json_files = list(Path(results_dir).glob("*_tn_shapley_*.json"))
    
    for json_file in json_files:
        print(f"Loading: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract sentence identifier
        sentence_key = json_file.stem.replace("_tn_shapley_zero", "").replace("_tn_shapley_mean", "")
        baseline = "zero" if "zero" in json_file.stem else "mean"
        
        if sentence_key not in results:
            results[sentence_key] = {}
        
        results[sentence_key][baseline] = data
    
    return results

def create_shapley_analysis(results: Dict, output_dir: str):
    """Create comprehensive analysis of Shapley values"""
    
    print("Creating Shapley value analysis...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Analyze single token Shapley values (order 1)
    print("Analyzing order 1 (single token) Shapley values...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    sentence_names = list(results.keys())
    
    for idx, sentence_key in enumerate(sentence_names):
        ax = axes[idx]
        
        if 'zero' in results[sentence_key]:
            data = results[sentence_key]['zero']
            
            if 'order_1' in data:
                values = np.array(data['order_1']['values'])
                tokens = data['metadata']['tokens']
                
                # Create bar plot
                bars = ax.bar(range(len(values)), values, alpha=0.7)
                ax.set_xlabel('Token Position', fontweight='bold')
                ax.set_ylabel('Shapley Value', fontweight='bold')
                ax.set_title(f'{sentence_key.replace("_", " ").title()}\\nSingle Token Shapley Values', 
                           fontweight='bold')
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels([token.replace('Ġ', '') for token in tokens], 
                                 rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # Color bars by magnitude
                for bar, val in zip(bars, values):
                    if val > 0:
                        bar.set_color('green')
                        bar.set_alpha(0.7)
                    else:
                        bar.set_color('red')
                        bar.set_alpha(0.7)
                
                # Add value annotations
                for i, (bar, val) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                              xytext=(0, 3 if height >= 0 else -15),
                              textcoords="offset points", ha='center', va='bottom' if height >= 0 else 'top',
                              fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'single_token_shapley_values.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze pairwise Shapley values (order 2)
    print("Analyzing order 2 (pairwise) Shapley values...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, sentence_key in enumerate(sentence_names):
        ax = axes[idx]
        
        if 'zero' in results[sentence_key]:
            data = results[sentence_key]['zero']
            
            if 'order_2' in data:
                pair_values = data['order_2']['values']
                tokens = data['metadata']['tokens']
                num_tokens = len(tokens)
                
                # Create pairwise interaction matrix
                interaction_matrix = np.zeros((num_tokens, num_tokens))
                
                for pair_key, value in pair_values.items():
                    i, j = map(int, pair_key.split('_'))
                    interaction_matrix[i, j] = value
                    interaction_matrix[j, i] = value  # Symmetric
                
                # Create heatmap
                im = ax.imshow(interaction_matrix, cmap='RdBu_r', aspect='auto')
                ax.set_xlabel('Token Position', fontweight='bold')
                ax.set_ylabel('Token Position', fontweight='bold')
                ax.set_title(f'{sentence_key.replace("_", " ").title()}\\nPairwise Interaction Matrix', 
                           fontweight='bold')
                
                # Set ticks and labels
                ax.set_xticks(range(num_tokens))
                ax.set_yticks(range(num_tokens))
                ax.set_xticklabels([token.replace('Ġ', '') for token in tokens], 
                                 rotation=45, ha='right')
                ax.set_yticklabels([token.replace('Ġ', '') for token in tokens])
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
                
                # Add text annotations for non-zero values
                for i in range(num_tokens):
                    for j in range(i+1, num_tokens):
                        if abs(interaction_matrix[i, j]) > 1e-6:
                            ax.text(j, i, f'{interaction_matrix[i, j]:.3f}', 
                                  ha='center', va='center', fontsize=8, 
                                  color='white' if abs(interaction_matrix[i, j]) > np.max(np.abs(interaction_matrix))/2 else 'black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairwise_shapley_interactions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics table
    print("Creating summary statistics...")
    
    summary_data = []
    for sentence_key in sentence_names:
        if 'zero' in results[sentence_key]:
            data = results[sentence_key]['zero']
            
            row = {
                'Sentence': sentence_key.replace('_', ' ').title(),
                'Original': data['metadata']['sentence'],
                'Num_Tokens': data['metadata']['num_tokens'],
                'Center_Prediction': data['metadata']['center_prediction']
            }
            
            if 'order_1' in data:
                values = np.array(data['order_1']['values'])
                row.update({
                    'Single_Mean': values.mean(),
                    'Single_Std': values.std(),
                    'Single_Min': values.min(),
                    'Single_Max': values.max()
                })
            
            if 'order_2' in data:
                pair_values = list(data['order_2']['values'].values())
                if pair_values:
                    pair_array = np.array(pair_values)
                    row.update({
                        'Pairs_Count': len(pair_values),
                        'Pairs_Mean': pair_array.mean(),
                        'Pairs_Std': pair_array.std(),
                        'Pairs_Min': pair_array.min(),
                        'Pairs_Max': pair_array.max()
                    })
            
            if 'timing' in data:
                row.update({
                    'Training_Time': data['timing']['training_time'],
                    'Shapley_Time': data['timing']['shapley_time'],
                    'Total_Time': data['timing']['total_time']
                })
            
            summary_data.append(row)
    
    # Save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'tn_shapley_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved: {summary_path}")
    
    # Create comparison plot
    print("Creating comparison plots...")
    
    # Compare single token importance across sentences
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    all_single_values = []
    all_labels = []
    
    for sentence_key in sentence_names:
        if 'zero' in results[sentence_key] and 'order_1' in results[sentence_key]['zero']:
            values = np.array(results[sentence_key]['zero']['order_1']['values'])
            tokens = results[sentence_key]['zero']['metadata']['tokens']
            
            for i, (val, token) in enumerate(zip(values, tokens)):
                all_single_values.append(val)
                clean_token = token.replace('Ġ', '')
                sentence_short = sentence_key.split('_')[1] if '_' in sentence_key else sentence_key[:10]
                all_labels.append(f'{sentence_short}\\n{clean_token}')
    
    # Create violin plot
    sentence_groups = []
    sentence_values = []
    for sentence_key in sentence_names:
        if 'zero' in results[sentence_key] and 'order_1' in results[sentence_key]['zero']:
            values = np.array(results[sentence_key]['zero']['order_1']['values'])
            sentence_groups.extend([sentence_key.replace('_', ' ').title()[:20]] * len(values))
            sentence_values.extend(values)
    
    if sentence_groups:
        df_plot = pd.DataFrame({'Sentence': sentence_groups, 'Shapley_Value': sentence_values})
        sns.violinplot(data=df_plot, x='Sentence', y='Shapley_Value', ax=ax)
        ax.set_xlabel('Sentence', fontweight='bold')
        ax.set_ylabel('Single Token Shapley Value', fontweight='bold')
        ax.set_title('Distribution of Single Token Shapley Values Across Sentences', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shapley_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis complete! Results saved to: {output_dir}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Analyze TN-SHAP results')
    parser.add_argument('--results-dir', type=str, default='./tn_results',
                       help='Directory containing TN-SHAP results')
    parser.add_argument('--output-dir', type=str, default='./tn_analysis',
                       help='Output directory for analysis plots')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TN-SHAPLEY RESULTS ANALYSIS")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load results
    results = load_tn_results(args.results_dir)
    
    if not results:
        print("No TN-SHAP results found!")
        return
    
    print(f"Found results for {len(results)} sentences:")
    for sentence_key in results.keys():
        baselines = list(results[sentence_key].keys())
        print(f"  {sentence_key}: {baselines}")
    
    # Create analysis
    summary_df = create_shapley_analysis(results, args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    print("\\nSingle Token Shapley Statistics:")
    if 'Single_Mean' in summary_df.columns:
        print(f"  Overall mean: {summary_df['Single_Mean'].mean():.6f}")
        print(f"  Overall std: {summary_df['Single_Mean'].std():.6f}")
        print(f"  Range: [{summary_df['Single_Min'].min():.6f}, {summary_df['Single_Max'].max():.6f}]")
    
    if 'Pairs_Mean' in summary_df.columns:
        print("\\nPairwise Interaction Statistics:")
        print(f"  Overall mean: {summary_df['Pairs_Mean'].mean():.6f}")
        print(f"  Overall std: {summary_df['Pairs_Mean'].std():.6f}")
        print(f"  Range: [{summary_df['Pairs_Min'].min():.6f}, {summary_df['Pairs_Max'].max():.6f}]")
    
    if 'Total_Time' in summary_df.columns:
        print("\\nTiming Summary:")
        print(f"  Average training time: {summary_df['Training_Time'].mean():.2f}s")
        print(f"  Average Shapley time: {summary_df['Shapley_Time'].mean():.2f}s")
        print(f"  Average total time: {summary_df['Total_Time'].mean():.2f}s")
    
    print("\\nGenerated files:")
    for file_path in Path(args.output_dir).glob("*"):
        if file_path.is_file():
            print(f"  {file_path.name}")

if __name__ == "__main__":
    main()
