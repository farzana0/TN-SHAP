#!/usr/bin/env python3
"""
Simple improved synthetic dataset generator for debugging.
"""

import numpy as np
import json
import argparse
import os

def create_simple_dataset(sentence: str, tokens: list, 
                         n_samples: int = 400,
                         embed_dim: int = 8) -> dict:
    """Create a simple synthetic dataset with clear patterns"""
    
    print(f"Creating simple dataset for: {sentence}")
    print(f"Tokens: {tokens}")
    print(f"Samples: {n_samples}, Embed dim: {embed_dim}")
    
    n_tokens = len(tokens)
    
    # Create structured base embeddings for each token
    base_embeddings = np.random.randn(n_tokens, embed_dim) * 0.3
    
    # Make each token have a unique signature
    for i in range(n_tokens):
        # Each token gets a strong signal in one dimension
        base_embeddings[i, i % embed_dim] = 1.0
    
    # Generate samples
    samples = []
    
    for i in range(n_samples):
        # Add noise to base embeddings
        embeddings = base_embeddings + np.random.randn(n_tokens, embed_dim) * 0.1
        
        # Create random mask (keep most tokens)
        mask = np.ones(n_tokens)
        if np.random.rand() < 0.3:  # 30% chance to mask something
            mask_idx = np.random.randint(n_tokens)
            mask[mask_idx] = 0
        
        # Simple target function: sum of first dimension of unmasked tokens
        # Plus interaction between first two unmasked tokens
        target = 0.0
        
        # Main effects
        for j in range(n_tokens):
            if mask[j] == 1:
                target += embeddings[j, 0] * 0.5
        
        # Simple interaction between first two tokens if both present
        if n_tokens >= 2 and mask[0] == 1 and mask[1] == 1:
            target += embeddings[0, 1] * embeddings[1, 0] * 0.3
        
        # Small noise
        target += np.random.randn() * 0.02
        
        samples.append({
            'embeddings': embeddings.tolist(),
            'mask': mask.tolist(),
            'target_logit': float(target),
            'sample_idx': i
        })
    
    # Create evaluation samples with specific mask patterns
    eval_samples = []
    
    for i in range(100):  # 100 eval samples
        embeddings = base_embeddings + np.random.randn(n_tokens, embed_dim) * 0.1
        
        # Different mask types
        if i < 25:  # No mask
            mask = np.ones(n_tokens)
            mask_type = 'no_mask'
        elif i < 50:  # Single token masked
            mask = np.ones(n_tokens)
            mask[i % n_tokens] = 0
            mask_type = 'single_mask'
        elif i < 75:  # Random mask
            mask = (np.random.rand(n_tokens) > 0.2).astype(float)
            if np.sum(mask) == 0:
                mask[0] = 1
            mask_type = 'random_mask'
        else:  # Half mask
            mask = np.ones(n_tokens)
            n_to_mask = max(1, n_tokens // 2)
            mask_indices = np.random.choice(n_tokens, n_to_mask, replace=False)
            mask[mask_indices] = 0
            mask_type = 'half_mask'
        
        # Same target function
        target = 0.0
        for j in range(n_tokens):
            if mask[j] == 1:
                target += embeddings[j, 0] * 0.5
        
        if n_tokens >= 2 and mask[0] == 1 and mask[1] == 1:
            target += embeddings[0, 1] * embeddings[1, 0] * 0.3
        
        target += np.random.randn() * 0.02
        
        eval_samples.append({
            'embeddings': embeddings.tolist(),
            'mask': mask.tolist(),
            'target_logit': float(target),
            'mask_type': mask_type,
            'n_masked': int(n_tokens - np.sum(mask))
        })
    
    # Organize dataset
    dataset = {
        'sentence': sentence,
        'token_info': {
            'token_strings': tokens,
            'n_tokens': n_tokens
        },
        'generation_config': {
            'n_samples': n_samples,
            'embed_dim': embed_dim,
            'noise_variance': 0.1
        },
        'data': {
            '1': {  # Order 1 (main effects only)
                'samples': samples[:n_samples//2],
                'n_samples': n_samples//2
            },
            '2': {  # Order 2 (with interactions)
                'samples': samples[n_samples//2:],
                'n_samples': n_samples//2
            }
        },
        'masked_evaluation': {
            'samples': eval_samples,
            'description': 'Samples for evaluating masked data R²'
        }
    }
    
    # Print statistics
    all_targets = [s['target_logit'] for s in samples]
    eval_targets = [s['target_logit'] for s in eval_samples]
    
    print(f"Training targets: mean={np.mean(all_targets):.4f}, std={np.std(all_targets):.4f}")
    print(f"Eval targets: mean={np.mean(eval_targets):.4f}, std={np.std(eval_targets):.4f}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', type=str, default="The food was delicious")
    parser.add_argument('--output-dir', type=str, default='./improved_datasets')
    parser.add_argument('--n-samples', type=int, default=400)
    parser.add_argument('--embed-dim', type=int, default=8)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokens = args.sentence.lower().replace(',', '').split()
    
    dataset = create_simple_dataset(args.sentence, tokens, args.n_samples, args.embed_dim)
    
    safe_filename = args.sentence.replace(' ', '_').replace(',', '')
    output_path = os.path.join(args.output_dir, f'{safe_filename}_simple.json')
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
