#!/usr/bin/env python3
"""
Quick verification and demo script for the GPT synthetic datasets.
Shows how to load and analyze the created datasets.
"""

import json
import torch
import numpy as np
from pathlib import Path

def load_dataset(dataset_path):
    """Load a dataset JSON file"""
    with open(dataset_path, 'r') as f:
        return json.load(f)

def analyze_dataset(data):
    """Analyze a loaded dataset"""
    print(f"Sentence: '{data['sentence']}'")
    print(f"Target token: '{data['target_token']}'")
    print(f"Number of tokens: {len(data['token_info']['token_strings'])}")
    print(f"Tokens: {data['token_info']['token_strings']}")
    print(f"Embedding dimension: {len(data['data'][0]['center_embedding'])}")
    
    # Analyze by order
    orders = {}
    for item in data['data']:
        order = len(item['masked_indices'])
        if order not in orders:
            orders[order] = []
        orders[order].append(item)
    
    print(f"\nData breakdown by order:")
    for order in sorted(orders.keys()):
        items = orders[order]
        probs = [item['target_prob'] for item in items]
        print(f"  Order k={order}: {len(items)} samples")
        print(f"    Target prob: {np.mean(probs):.6f} ± {np.std(probs):.6f}")
        print(f"    Range: [{np.min(probs):.6f}, {np.max(probs):.6f}]")
        
        # Show a few example masked indices
        examples = [item['masked_indices'] for item in items[:3]]
        print(f"    Example masks: {examples}")

def main():
    print("="*60)
    print("GPT SYNTHETIC DATASET VERIFICATION")
    print("="*60)
    
    dataset_dir = Path(".")
    dataset_files = list(dataset_dir.glob("sentence_*_dataset.json"))
    
    if not dataset_files:
        print("No dataset files found! Please run create_gpt_synthetic_dataset.py first.")
        return
    
    print(f"Found {len(dataset_files)} dataset files:")
    for file in sorted(dataset_files):
        print(f"  - {file.name}")
    
    # Load and analyze statistics
    stats_file = dataset_dir / "dataset_statistics.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"\n" + "="*60)
        print("OVERALL STATISTICS")
        print("="*60)
        
        for sentence_key, sentence_data in stats.items():
            print(f"\n{sentence_key.upper()}:")
            print(f"  Sentence: '{sentence_data['sentence']}'")
            print(f"  Tokens: {sentence_data['n_tokens']}")
            
            total_samples = 0
            for order, order_data in sentence_data['orders'].items():
                n_samples = order_data['total_samples']
                total_samples += n_samples
                prob_mean = order_data['target_prob_mean']
                prob_std = order_data['target_prob_std']
                print(f"  Order k={order}: {n_samples} samples, prob={prob_mean:.6f}±{prob_std:.6f}")
            
            print(f"  Total samples: {total_samples}")
    
    # Detailed analysis of first dataset
    if dataset_files:
        print(f"\n" + "="*60)
        print("DETAILED ANALYSIS - FIRST DATASET")
        print("="*60)
        
        first_dataset = sorted(dataset_files)[0]
        data = load_dataset(first_dataset)
        analyze_dataset(data)
        
        # Show embedding shapes
        print(f"\nEmbedding analysis:")
        center_embed = np.array(data['data'][0]['center_embedding'])
        print(f"  Center embedding shape: {center_embed.shape}")
        print(f"  Center embedding stats: mean={center_embed.mean():.4f}, std={center_embed.std():.4f}")
        
        # Show noise examples
        if 'neighborhood_embeddings' in data['data'][0]:
            neighbor_embeds = np.array(data['data'][0]['neighborhood_embeddings'])
            print(f"  Neighborhood embeddings shape: {neighbor_embeds.shape}")
            print(f"  Noise level (std of differences): {np.std(neighbor_embeds - center_embed):.4f}")

if __name__ == "__main__":
    main()
