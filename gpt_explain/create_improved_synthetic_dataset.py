#!/usr/bin/env python3
"""
Improved synthetic dataset generator for TN-tree training.

This script creates synthetic datasets with:
1. Better signal-to-noise ratio
2. More structured embeddings 
3. Separate evaluation on masked vs unmasked data
4. Controllable complexity for different orders of interactions

Key improvements:
- Uses structured synthetic embeddings instead of random noise
- Creates meaningful interaction patterns
- Generates both masked and unmasked samples for R² evaluation
- Controllable variance and complexity

Created: September 26, 2025
"""

import numpy as np
import json
import argparse
import os
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F

def create_structured_embeddings(n_tokens: int, embed_dim: int, 
                                base_seed: int = 42) -> np.ndarray:
    """Create structured embeddings that have meaningful patterns"""
    np.random.seed(base_seed)
    
    # Create base patterns for each token position
    embeddings = np.zeros((n_tokens, embed_dim))
    
    for i in range(n_tokens):
        # Each token has a unique signature + some shared patterns
        # Unique component (50% of dimensions)
        unique_dims = embed_dim // 2
        embeddings[i, :unique_dims] = np.random.randn(unique_dims) * 0.5
        embeddings[i, i % unique_dims] += 2.0  # Strong unique signal
        
        # Shared patterns (remaining dimensions)
        shared_start = unique_dims
        # Add some global patterns
        embeddings[i, shared_start:shared_start+2] = [0.3, -0.2]  # Global bias
        # Add position-dependent patterns
        if i < n_tokens // 2:
            embeddings[i, shared_start+2:shared_start+4] = [0.8, 0.1]
        else:
            embeddings[i, shared_start+2:shared_start+4] = [-0.1, 0.8]
    
    return embeddings

def generate_target_function(embeddings: np.ndarray, mask: np.ndarray, 
                           interaction_order: int = 2) -> float:
    """
    Generate target values based on embeddings and mask.
    
    Args:
        embeddings: [n_tokens, embed_dim] embeddings
        mask: [n_tokens] binary mask (1 = keep, 0 = mask out)
        interaction_order: Maximum order of interactions to include
    """
    n_tokens, embed_dim = embeddings.shape
    
    # Apply mask
    masked_embeddings = embeddings * mask[:, np.newaxis]
    
    target = 0.0
    
    # Order 1: Linear terms (main effects)
    for i in range(n_tokens):
        if mask[i] == 1:  # Only if token is not masked
            # Use first few dimensions for main effect
            main_effect = np.sum(masked_embeddings[i, :3]) * 0.5
            target += main_effect
    
    # Order 2: Pairwise interactions
    if interaction_order >= 2:
        for i in range(n_tokens):
            for j in range(i+1, n_tokens):
                if mask[i] == 1 and mask[j] == 1:  # Both tokens must be present
                    # Interaction based on specific embedding dimensions
                    interaction = (masked_embeddings[i, 0] * masked_embeddings[j, 1] + 
                                 masked_embeddings[i, 1] * masked_embeddings[j, 0]) * 0.3
                    target += interaction
    
    # Order 3: Three-way interactions (if requested)
    if interaction_order >= 3 and n_tokens >= 3:
        for i in range(n_tokens):
            for j in range(i+1, n_tokens):
                for k in range(j+1, n_tokens):
                    if mask[i] == 1 and mask[j] == 1 and mask[k] == 1:
                        # Three-way interaction (smaller coefficient)
                        interaction = (masked_embeddings[i, 0] * 
                                     masked_embeddings[j, 1] * 
                                     masked_embeddings[k, 2]) * 0.1
                        target += interaction
    
    # Add small amount of noise for realism
    target += np.random.randn() * 0.05
    
    return float(target)

def generate_mask_patterns(n_tokens: int, n_samples: int, 
                          mask_prob: float = 0.3) -> List[np.ndarray]:
    """Generate different masking patterns for evaluation"""
    masks = []
    
    # 1. No masking (full data) - 20% of samples
    n_full = max(1, n_samples // 5)
    for _ in range(n_full):
        masks.append(np.ones(n_tokens))
    
    # 2. Single token masking - 40% of samples
    n_single = max(1, int(n_samples * 0.4))
    for _ in range(n_single):
        mask = np.ones(n_tokens)
        mask_idx = np.random.randint(n_tokens)
        mask[mask_idx] = 0
        masks.append(mask)
    
    # 3. Random masking - remaining samples
    n_remaining = n_samples - len(masks)
    for _ in range(n_remaining):
        mask = (np.random.rand(n_tokens) > mask_prob).astype(float)
        # Ensure at least one token is present
        if np.sum(mask) == 0:
            mask[np.random.randint(n_tokens)] = 1
        masks.append(mask)
    
    return masks

def create_improved_dataset(sentence: str, tokens: List[str], 
                          n_samples_per_order: int = 500,
                          embed_dim: int = 16,
                          max_order: int = 2,
                          noise_variance: float = 0.1) -> Dict:
    """
    Create improved synthetic dataset with better structure.
    
    Args:
        sentence: The sentence text
        tokens: List of token strings
        n_samples_per_order: Number of samples to generate per interaction order
        embed_dim: Embedding dimension
        max_order: Maximum interaction order to generate
        noise_variance: Amount of noise to add to embeddings
    """
    
    n_tokens = len(tokens)
    print(f"Creating dataset for: {sentence}")
    print(f"Tokens: {tokens}")
    print(f"Generating {n_samples_per_order} samples per order (1 to {max_order})")
    print(f"Embedding dim: {embed_dim}, Noise variance: {noise_variance}")
    
    # Create base structured embeddings
    base_embeddings = create_structured_embeddings(n_tokens, embed_dim)
    
    dataset = {
        'sentence': sentence,
        'token_info': {
            'token_strings': tokens,
            'n_tokens': n_tokens
        },
        'generation_config': {
            'n_samples_per_order': n_samples_per_order,
            'embed_dim': embed_dim,
            'max_order': max_order,
            'noise_variance': noise_variance
        },
        'data': {}
    }
    
    # Generate samples for each order
    for order in range(1, max_order + 1):
        print(f"\nGenerating order {order} samples...")
        
        samples = []
        
        for sample_idx in range(n_samples_per_order):
            # Add noise to base embeddings
            noisy_embeddings = (base_embeddings + 
                              np.random.randn(n_tokens, embed_dim) * noise_variance)
            
            # Generate mask pattern
            masks = generate_mask_patterns(n_tokens, 1, mask_prob=0.2)[0]
            
            # Generate target based on current order
            target_logit = generate_target_function(noisy_embeddings, masks, 
                                                   interaction_order=order)
            
            # Store sample
            sample = {
                'embeddings': noisy_embeddings.tolist(),  # [n_tokens, embed_dim]
                'mask': masks.tolist(),  # [n_tokens]
                'target_logit': target_logit,
                'interaction_order': order,
                'sample_idx': sample_idx
            }
            
            samples.append(sample)
        
        dataset['data'][str(order)] = {
            'samples': samples,
            'n_samples': len(samples)
        }
        
        # Print statistics
        targets = [s['target_logit'] for s in samples]
        print(f"  Target range: [{min(targets):.4f}, {max(targets):.4f}]")
        print(f"  Target mean: {np.mean(targets):.4f}, std: {np.std(targets):.4f}")
    
    return dataset

def create_masked_evaluation_samples(base_embeddings: np.ndarray, 
                                   n_samples: int = 200) -> List[Dict]:
    """Create specific samples for masked data evaluation"""
    n_tokens, embed_dim = base_embeddings.shape
    samples = []
    
    print(f"\nCreating {n_samples} masked evaluation samples...")
    
    # Different types of masks for evaluation
    mask_types = [
        ('no_mask', lambda: np.ones(n_tokens)),
        ('single_mask', lambda: create_single_token_mask(n_tokens)),
        ('random_mask', lambda: create_random_mask(n_tokens, 0.3)),
        ('half_mask', lambda: create_half_mask(n_tokens))
    ]
    
    samples_per_type = n_samples // len(mask_types)
    
    for mask_type, mask_func in mask_types:
        for i in range(samples_per_type):
            # Add noise to embeddings
            noisy_embeddings = (base_embeddings + 
                              np.random.randn(n_tokens, embed_dim) * 0.1)
            
            # Create mask
            mask = mask_func()
            
            # Generate target (using order 2 interactions)
            target = generate_target_function(noisy_embeddings, mask, 
                                            interaction_order=2)
            
            samples.append({
                'embeddings': noisy_embeddings.tolist(),
                'mask': mask.tolist(),
                'target_logit': target,
                'mask_type': mask_type,
                'n_masked': int(n_tokens - np.sum(mask))
            })
    
    return samples

def create_single_token_mask(n_tokens: int) -> np.ndarray:
    """Mask out a single random token"""
    mask = np.ones(n_tokens)
    mask[np.random.randint(n_tokens)] = 0
    return mask

def create_random_mask(n_tokens: int, mask_prob: float) -> np.ndarray:
    """Random masking with given probability"""
    mask = (np.random.rand(n_tokens) > mask_prob).astype(float)
    # Ensure at least one token remains
    if np.sum(mask) == 0:
        mask[np.random.randint(n_tokens)] = 1
    return mask

def create_half_mask(n_tokens: int) -> np.ndarray:
    """Mask out approximately half the tokens"""
    mask = np.ones(n_tokens)
    n_to_mask = max(1, n_tokens // 2)
    mask_indices = np.random.choice(n_tokens, n_to_mask, replace=False)
    mask[mask_indices] = 0
    return mask

def main():
    parser = argparse.ArgumentParser(description='Create improved synthetic dataset')
    parser.add_argument('--sentence', type=str, 
                       default="The food was delicious and fresh",
                       help='Sentence to create dataset for')
    parser.add_argument('--output-dir', type=str, default='./improved_datasets',
                       help='Output directory')
    parser.add_argument('--n-samples', type=int, default=800,
                       help='Number of samples per interaction order')
    parser.add_argument('--embed-dim', type=int, default=16,
                       help='Embedding dimension')
    parser.add_argument('--max-order', type=int, default=2,
                       help='Maximum interaction order')
    parser.add_argument('--noise-variance', type=float, default=0.1,
                       help='Noise variance for embeddings')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Simple tokenization
    tokens = args.sentence.lower().replace(',', '').replace('.', '').split()
    
    print("=" * 80)
    print("IMPROVED SYNTHETIC DATASET GENERATION")
    print("=" * 80)
    print(f"Sentence: {args.sentence}")
    print(f"Tokens: {tokens}")
    print(f"Output directory: {args.output_dir}")
    
    # Create main dataset
    dataset = create_improved_dataset(
        sentence=args.sentence,
        tokens=tokens,
        n_samples_per_order=args.n_samples,
        embed_dim=args.embed_dim,
        max_order=args.max_order,
        noise_variance=args.noise_variance
    )
    
    # Create base embeddings for masked evaluation
    base_embeddings = create_structured_embeddings(len(tokens), args.embed_dim)
    
    # Create masked evaluation samples
    masked_eval_samples = create_masked_evaluation_samples(
        base_embeddings, n_samples=400
    )
    
    dataset['masked_evaluation'] = {
        'samples': masked_eval_samples,
        'description': 'Samples specifically for evaluating R² on masked data'
    }
    
    # Save dataset
    safe_filename = args.sentence.replace(' ', '_').replace(',', '').replace('.', '')
    output_path = os.path.join(args.output_dir, f'{safe_filename}_improved.json')
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print("DATASET CREATION COMPLETE")
    print("=" * 80)
    print(f"Dataset saved to: {output_path}")
    print(f"Total samples: {sum(len(order_data['samples']) for order_data in dataset['data'].values())}")
    print(f"Masked evaluation samples: {len(masked_eval_samples)}")
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    for order, order_data in dataset['data'].items():
        samples = order_data['samples']
        targets = [s['target_logit'] for s in samples]
        print(f"  Order {order}: {len(samples)} samples")
        print(f"    Target range: [{min(targets):.4f}, {max(targets):.4f}]")
        print(f"    Target mean: {np.mean(targets):.4f} ± {np.std(targets):.4f}")
    
    print(f"\nMasked evaluation statistics:")
    masked_targets = [s['target_logit'] for s in masked_eval_samples]
    print(f"  Samples: {len(masked_eval_samples)}")
    print(f"  Target range: [{min(masked_targets):.4f}, {max(masked_targets):.4f}]")
    print(f"  Target mean: {np.mean(masked_targets):.4f} ± {np.std(masked_targets):.4f}")
    
    # Print mask type distribution
    mask_types = {}
    for sample in masked_eval_samples:
        mask_type = sample['mask_type']
        mask_types[mask_type] = mask_types.get(mask_type, 0) + 1
    
    print(f"\nMask type distribution:")
    for mask_type, count in mask_types.items():
        print(f"  {mask_type}: {count} samples")

if __name__ == "__main__":
    main()
