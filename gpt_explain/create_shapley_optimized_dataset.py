#!/usr/bin/env python3
"""
Optimized synthetic dataset generator focused on Shapley-relevant masking patterns.

This version creates data specifically designed for good Shapley value computation:
1. Focuses on single-token and pairwise masking patterns
2. Creates clearer interaction structures
3. Ensures good R² performance on the masking patterns used by Shapley
"""

import numpy as np
import json
import argparse
import os

def create_token_embeddings(tokens: list, embed_dim: int) -> np.ndarray:
    """Create meaningful embeddings for each token"""
    n_tokens = len(tokens)
    embeddings = np.zeros((n_tokens, embed_dim))
    
    # Each token gets:
    # 1. A unique signature (one-hot-like)
    # 2. Some semantic content (based on token meaning)
    # 3. Positional information
    
    for i, token in enumerate(tokens):
        # Unique signature
        embeddings[i, i % embed_dim] = 2.0
        
        # Semantic content (simple patterns)
        if token in ['the', 'a', 'an']:  # Articles
            embeddings[i, -2] = 1.0
        elif token in ['was', 'is', 'are']:  # Verbs
            embeddings[i, -1] = 1.0
        elif token in ['delicious', 'tasty', 'good', 'bad', 'fresh']:  # Adjectives
            embeddings[i, -3] = 1.0
        else:  # Nouns
            embeddings[i, -4] = 1.0
        
        # Positional encoding
        pos_encoding = np.sin(np.linspace(0, 2*np.pi, embed_dim-4)) * 0.3
        embeddings[i, :embed_dim-4] += pos_encoding * (i + 1) / n_tokens
    
    return embeddings

def generate_shapley_target(embeddings: np.ndarray, mask: np.ndarray) -> float:
    """
    Generate targets with clear Shapley-interpretable structure:
    1. Each token contributes individually (main effects)
    2. Some pairs of tokens have interactions
    3. Clear masking behavior for Shapley computation
    """
    n_tokens = embeddings.shape[0]
    
    # Apply mask
    masked_embeddings = embeddings * mask[:, np.newaxis]
    
    target = 0.0
    
    # Main effects: each token contributes based on its unique signature
    main_effect_weights = [0.5, 0.8, 0.6, 0.7]  # Different importance per position
    for i in range(n_tokens):
        if mask[i] == 1 and i < len(main_effect_weights):
            # Use the unique signature dimension
            token_contribution = masked_embeddings[i, i % embeddings.shape[1]]
            target += token_contribution * main_effect_weights[i]
    
    # Pairwise interactions (only specific pairs)
    interaction_pairs = [(0, 1), (1, 3), (2, 3)]  # Specific meaningful pairs
    interaction_weights = [0.4, 0.3, 0.2]
    
    for (i, j), weight in zip(interaction_pairs, interaction_weights):
        if i < n_tokens and j < n_tokens and mask[i] == 1 and mask[j] == 1:
            # Interaction based on semantic dimensions
            interaction = (masked_embeddings[i, -2] * masked_embeddings[j, -1] + 
                          masked_embeddings[i, -1] * masked_embeddings[j, -2])
            target += interaction * weight
    
    # Small noise for realism
    target += np.random.randn() * 0.01
    
    return float(target)

def create_shapley_focused_samples(base_embeddings: np.ndarray, 
                                  n_samples: int) -> list:
    """Create samples focused on masking patterns used by Shapley computation"""
    n_tokens = base_embeddings.shape[0]
    samples = []
    
    # Distribution of samples:
    # 40% - Single token masking (key for order 1 Shapley)
    # 30% - Pairwise masking (key for order 2 Shapley)  
    # 20% - No masking (baseline)
    # 10% - Random masking (robustness)
    
    # Single token masking - 40%
    n_single = int(n_samples * 0.4)
    for i in range(n_single):
        embeddings = base_embeddings + np.random.randn(*base_embeddings.shape) * 0.05
        mask = np.ones(n_tokens)
        mask[i % n_tokens] = 0  # Cycle through tokens
        
        target = generate_shapley_target(embeddings, mask)
        
        samples.append({
            'embeddings': embeddings.tolist(),
            'mask': mask.tolist(),
            'target_logit': target,
            'mask_type': 'single_token_masked',
            'n_masked': 1
        })
    
    # Pairwise masking - 30%
    n_pairs = int(n_samples * 0.3)
    pair_combinations = [(i, j) for i in range(n_tokens) for j in range(i+1, n_tokens)]
    
    for i in range(n_pairs):
        embeddings = base_embeddings + np.random.randn(*base_embeddings.shape) * 0.05
        mask = np.ones(n_tokens)
        
        # Choose a pair to mask
        pair_idx = i % len(pair_combinations)
        mask_i, mask_j = pair_combinations[pair_idx]
        mask[mask_i] = 0
        mask[mask_j] = 0
        
        target = generate_shapley_target(embeddings, mask)
        
        samples.append({
            'embeddings': embeddings.tolist(),
            'mask': mask.tolist(),
            'target_logit': target,
            'mask_type': 'pair_masked',
            'n_masked': 2
        })
    
    # No masking - 20%
    n_full = int(n_samples * 0.2)
    for i in range(n_full):
        embeddings = base_embeddings + np.random.randn(*base_embeddings.shape) * 0.05
        mask = np.ones(n_tokens)
        
        target = generate_shapley_target(embeddings, mask)
        
        samples.append({
            'embeddings': embeddings.tolist(),
            'mask': mask.tolist(),
            'target_logit': target,
            'mask_type': 'no_mask',
            'n_masked': 0
        })
    
    # Random masking - 10%
    n_random = n_samples - len(samples)
    for i in range(n_random):
        embeddings = base_embeddings + np.random.randn(*base_embeddings.shape) * 0.05
        mask = (np.random.rand(n_tokens) > 0.3).astype(float)
        
        # Ensure at least one token remains
        if np.sum(mask) == 0:
            mask[np.random.randint(n_tokens)] = 1
        
        target = generate_shapley_target(embeddings, mask)
        
        samples.append({
            'embeddings': embeddings.tolist(),
            'mask': mask.tolist(),
            'target_logit': target,
            'mask_type': 'random_mask',
            'n_masked': int(n_tokens - np.sum(mask))
        })
    
    return samples

def create_shapley_dataset(sentence: str, tokens: list,
                          n_samples: int = 1000,
                          embed_dim: int = 12) -> dict:
    """Create dataset optimized for Shapley value computation"""
    
    print(f"Creating Shapley-optimized dataset for: {sentence}")
    print(f"Tokens: {tokens}")
    print(f"Samples: {n_samples}, Embed dim: {embed_dim}")
    
    n_tokens = len(tokens)
    
    # Create meaningful base embeddings
    base_embeddings = create_token_embeddings(tokens, embed_dim)
    
    print(f"Base embedding structure:")
    for i, token in enumerate(tokens):
        unique_dim = i % embed_dim
        print(f"  {token}: unique signal in dim {unique_dim}")
    
    # Generate training samples
    train_samples = create_shapley_focused_samples(base_embeddings, n_samples)
    
    # Generate evaluation samples (focused on Shapley patterns)
    eval_samples = create_shapley_focused_samples(base_embeddings, 200)
    
    # Organize by interaction order
    order1_samples = []  # Samples good for order 1 Shapley
    order2_samples = []  # Samples good for order 2 Shapley
    
    # Split based on mask type
    for sample in train_samples:
        if sample['mask_type'] in ['single_token_masked', 'no_mask']:
            order1_samples.append(sample)
        else:
            order2_samples.append(sample)
    
    # Ensure balanced orders
    min_samples = min(len(order1_samples), len(order2_samples))
    order1_samples = order1_samples[:min_samples]
    order2_samples = order2_samples[:min_samples]
    
    dataset = {
        'sentence': sentence,
        'token_info': {
            'token_strings': tokens,
            'n_tokens': n_tokens
        },
        'generation_config': {
            'n_samples': n_samples,
            'embed_dim': embed_dim,
            'focus': 'shapley_computation',
            'masking_distribution': {
                'single_token': '40%',
                'pairwise': '30%', 
                'no_mask': '20%',
                'random': '10%'
            }
        },
        'data': {
            '1': {
                'samples': order1_samples,
                'n_samples': len(order1_samples),
                'description': 'Samples optimized for order 1 Shapley computation'
            },
            '2': {
                'samples': order2_samples,
                'n_samples': len(order2_samples),
                'description': 'Samples optimized for order 2 Shapley computation'
            }
        },
        'masked_evaluation': {
            'samples': eval_samples,
            'description': 'Evaluation samples for masked data R² computation'
        }
    }
    
    # Print statistics
    all_targets = [s['target_logit'] for s in train_samples]
    eval_targets = [s['target_logit'] for s in eval_samples]
    
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(train_samples)}")
    print(f"  Order 1 samples: {len(order1_samples)}")
    print(f"  Order 2 samples: {len(order2_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")
    
    print(f"\nTarget Statistics:")
    print(f"Training: mean={np.mean(all_targets):.4f}, std={np.std(all_targets):.4f}")
    print(f"Training: range=[{np.min(all_targets):.4f}, {np.max(all_targets):.4f}]")
    print(f"Evaluation: mean={np.mean(eval_targets):.4f}, std={np.std(eval_targets):.4f}")
    
    # Print mask type distribution for evaluation
    eval_mask_types = {}
    for sample in eval_samples:
        mask_type = sample['mask_type']
        eval_mask_types[mask_type] = eval_mask_types.get(mask_type, 0) + 1
    
    print(f"\nEvaluation mask distribution:")
    for mask_type, count in eval_mask_types.items():
        print(f"  {mask_type}: {count} samples")
    
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', type=str, default="The food was delicious")
    parser.add_argument('--output-dir', type=str, default='./shapley_datasets')
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--embed-dim', type=int, default=12)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokens = args.sentence.lower().replace(',', '').replace('.', '').split()
    
    dataset = create_shapley_dataset(args.sentence, tokens, 
                                   args.n_samples, args.embed_dim)
    
    safe_filename = args.sentence.replace(' ', '_').replace(',', '').replace('.', '')
    output_path = os.path.join(args.output_dir, f'{safe_filename}_shapley_optimized.json')
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nDataset saved to: {output_path}")

if __name__ == "__main__":
    main()
