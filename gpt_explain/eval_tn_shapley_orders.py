#!/usr/bin/env python3
"""
Evaluate TN-tree models with higher-order Shapley values on GPT datasets.
Based on eval_all_highorders.py but adapted for our GPT dataset format and TN-tree models.

This script computes:
- Order 1 and 2 Shapley values using exact computation  
- Separate R² evaluation for masked vs unmasked data
- Performance metrics for different masking patterns
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Tuple, Optional

# Add paths for imports
sys.path.append('./higher_order_iclr')
from tntree_model import BinaryTensorTree

def ensure_json_serializable(obj):
    """Convert numpy arrays and other non-serializable types to JSON serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj

def load_dataset_and_model(result_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], BinaryTensorTree, Dict]:
    """Load dataset and trained model from result JSON"""
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    # Extract dataset path and load data
    dataset_path = results['dataset_path']
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Load dataset the same way as training
    print(f"Loading dataset: {dataset['sentence']}")
    print(f"Tokens: {dataset['token_info']['token_strings']}")
    
    X_list = []
    y_list = []
    
    for order, order_data in dataset['data'].items():
        samples = order_data['samples']
        print(f"Processing order {order}: {len(samples)} samples")
        
        for sample in samples:
            embeddings = sample['embeddings']  # Shape: [6, 10, 768]
            target_logit = sample['target_logit']
            
            # Convert to numpy array and loop through noise samples
            embeddings_array = np.array(embeddings)  # [6, 10, 768]
            n_noise_samples = embeddings_array.shape[0]
            
            for i in range(n_noise_samples):
                # Take first 16 dimensions of each token (same as training)
                X_sample = embeddings_array[i, :, :16]  # [10, 16]
                X_list.append(X_sample)
                y_list.append(target_logit)
    
    # Stack all samples
    X = np.stack(X_list)  # Shape: [n_samples, n_tokens, 16]
    y = np.array(y_list)
    
    tokens = dataset['token_info']['token_strings']
    
    # Reconstruct model from config
    model_config = results['model_config']
    n_tokens = len(tokens)
    embed_dim = 16
    
    # Create model with same architecture
    leaf_phys_dims = model_config['leaf_phys_dims']
    leaf_input_dims = model_config['leaf_phys_dims']  # Same as phys_dims
    
    model = BinaryTensorTree(
        leaf_phys_dims=leaf_phys_dims,
        leaf_input_dims=leaf_input_dims,
        ranks=model_config['rank'],
        assume_bias_when_matrix=True,
        device='cpu'
    )
    
    # Load model weights (need to find the corresponding .pt file)
    model_path = result_path.replace('.json', '.pt').replace('_tn_shapley_enhanced.json', '_tn_model_enhanced.pt')
    if not os.path.exists(model_path):
        # Try alternative naming pattern
        model_path = result_path.replace('_tn_shapley_enhanced.json', '_tn_model_enhanced.pt')
        if not os.path.exists(model_path):
            # Try with same base name but different extension
            base_path = result_path.replace('_tn_shapley_enhanced.json', '')
            model_path = base_path + '_tn_model_enhanced.pt'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded model weights from: {model_path}")
    else:
        print(f"Warning: Model weights not found. Tried: {model_path}")
        print("Model will use random weights - results will be meaningless!")
    
    return X, y, tokens, model, results

def exact_k_sii_zero_batch(predict_fn, x, k, max_subsets=None, batch_size=1024):
    """
    Compute exact k-SII (Shapley Interaction Index) with zero baseline.
    Optimized version with batching for efficiency.
    """
    print(f"Computing exact {k}-SII with zero baseline...")
    t0 = time.time()
    
    x = np.asarray(x, np.float32).ravel()
    d = x.size
    
    print(f"  Features: {d}")
    
    # Generate all possible k-subsets
    k_subsets = list(combinations(range(d), k))
    if max_subsets and len(k_subsets) > max_subsets:
        k_subsets = k_subsets[:max_subsets]
        print(f"  Limited to {max_subsets} subsets")
    
    print(f"  Computing {len(k_subsets)} {k}-subsets...")
    
    # For efficiency, we'll compute this in batches
    # For k=1: we need 2^(d-1) evaluations per subset
    # For k=2: we need 2^(d-2) evaluations per subset
    
    if d <= 10:  # Small enough for exact computation
        return exact_k_sii_zero_full(predict_fn, x, k, k_subsets)
    else:  # Use sampling approximation
        return exact_k_sii_zero_sampled(predict_fn, x, k, k_subsets, batch_size)

def exact_k_sii_zero_full(predict_fn, x, k, k_subsets):
    """Full exact computation for small feature spaces"""
    d = len(x)
    M = 1 << d  # 2^d
    
    # Precompute all function values
    masks = np.arange(M, dtype=np.uint32)
    bits = ((masks[:, None] >> np.arange(d, dtype=np.uint32)) & 1).astype(np.float32)
    
    print(f"  Evaluating {M} function values...")
    
    # Batch evaluation
    batch_size = min(1024, M)
    fvals = np.empty(M, np.float64)
    
    for s in range(0, M, batch_size):
        e = min(s + batch_size, M)
        X_batch = (bits[s:e] * x[None, :]).astype(np.float32)
        fvals[s:e] = predict_fn(X_batch).ravel().astype(np.float64)
    
    # Compute factorials
    fact = np.ones(d + 1, np.float64)
    for i in range(2, d + 1):
        fact[i] = fact[i - 1] * i
    
    # Weight computation  
    denom = fact[d - k + 1] if k <= d else 1.0
    w_by_s = np.array([fact[s] * fact[d - s - k] / denom 
                       for s in range(max(0, d - k + 1))], np.float64)
    
    k_sizes = np.unpackbits(masks.view(np.uint8)).reshape(-1, 32)[:, :d].sum(1)
    
    results = []
    for T in k_subsets:
        T_bit = sum(1 << t for t in T)
        
        # Find all subsets S such that S ∩ T = ∅
        mask_T = (masks & T_bit) == 0
        S_all = masks[mask_T]
        s_sizes = k_sizes[mask_T]
        
        if len(s_sizes) >= len(w_by_s):
            valid_indices = s_sizes < len(w_by_s)
            S_all = S_all[valid_indices]
            s_sizes = s_sizes[valid_indices]
        
        if len(s_sizes) == 0:
            results.append(0.0)
            continue
            
        wS = w_by_s[s_sizes]
        
        # Compute alternating sum over U ⊆ T
        delta = np.zeros_like(wS, np.float64)
        for r in range(k + 1):
            sign = (-1.0) ** (k - r)
            for U in combinations(T, r):
                U_bit = sum(1 << u for u in U)
                idx = S_all | U_bit
                delta += sign * fvals[idx]
        
        result = np.sum(wS * delta)
        results.append(float(result))
    
    computation_time = time.time() - t0
    print(f"  Computation time: {computation_time:.2f} seconds")
    
    return np.array(results), computation_time

def exact_k_sii_zero_sampled(predict_fn, x, k, k_subsets, batch_size=1024, n_samples=5000):
    """Sampled approximation for larger feature spaces"""
    t0 = time.time()
    d = len(x)
    results = []
    
    print(f"  Using sampling approximation with {n_samples} samples")
    
    for i, T in enumerate(k_subsets):
        if i % 10 == 0:
            print(f"    Processing subset {i+1}/{len(k_subsets)}")
        
        # Sample random subsets S disjoint from T
        T_set = set(T)
        available_features = [f for f in range(d) if f not in T_set]
        
        if len(available_features) == 0:
            results.append(0.0)
            continue
        
        # Generate random subsets
        subset_values = []
        for _ in range(n_samples):
            # Random subset size
            max_size = len(available_features)
            subset_size = np.random.randint(0, max_size + 1)
            
            if subset_size == 0:
                S = []
            else:
                S = np.random.choice(available_features, subset_size, replace=False).tolist()
            
            # Compute alternating sum for this S
            delta = 0.0
            for r in range(k + 1):
                sign = (-1.0) ** (k - r)
                for U in combinations(T, r):
                    # Create input: S ∪ U features set to x[i], others to 0
                    x_eval = np.zeros_like(x)
                    for idx in S + list(U):
                        x_eval[idx] = x[idx]
                    
                    pred_result = predict_fn(x_eval.reshape(1, -1))
                    pred = pred_result[0] if len(pred_result) > 0 else pred_result
                    delta += sign * pred
            
            subset_values.append(delta)
        
        # Average over samples
        result = np.mean(subset_values)
        results.append(float(result))
    
    computation_time = time.time() - t0
    return np.array(results), computation_time

def evaluate_masked_data_r2(model, X, y, tokens):
    """
    Evaluate R² performance on different masking patterns.
    This is crucial for understanding Shapley value quality.
    """
    print("\nEvaluating R² on different masking patterns...")
    
    n_samples, n_tokens, embed_dim = X.shape
    
    # Add bias dimension
    bias_dim = np.ones((n_samples, n_tokens, 1))
    X_with_bias = np.concatenate([X, bias_dim], axis=2)
    
    results = {}
    
    # Flatten for model input
    X_flat = X_with_bias.reshape(n_samples, -1)
    X_tensor = torch.FloatTensor(X_flat)
    
    model.eval()
    with torch.no_grad():
        # 1. Full data (no masking)
        y_pred_full = model(X_tensor).squeeze().numpy()
        r2_full = r2_score(y, y_pred_full)
        results['full_data'] = {
            'r2': float(r2_full),
            'mse': float(mean_squared_error(y, y_pred_full)),
            'description': 'No masking - all features present'
        }
        
        # 2. Single token masked (order-1 Shapley baseline)
        r2_single_masked = []
        for mask_idx in range(n_tokens):
            X_masked = X_with_bias.copy()
            # Set masked token features to 0 (but keep bias=1)
            X_masked[:, mask_idx, :-1] = 0  # Keep bias at index -1
            
            X_masked_flat = X_masked.reshape(n_samples, -1)
            X_masked_tensor = torch.FloatTensor(X_masked_flat)
            
            y_pred_masked = model(X_masked_tensor).squeeze().numpy()
            r2_masked = r2_score(y, y_pred_masked)
            r2_single_masked.append(r2_masked)
        
        results['single_token_masked'] = {
            'r2_per_token': [float(r2) for r2 in r2_single_masked],
            'r2_mean': float(np.mean(r2_single_masked)),
            'r2_std': float(np.std(r2_single_masked)),
            'description': 'One token masked at a time (for order-1 Shapley)'
        }
        
        # 3. Pairs of tokens masked (order-2 Shapley baseline)
        if n_tokens >= 2:
            r2_pair_masked = []
            for i, j in combinations(range(n_tokens), 2):
                X_masked = X_with_bias.copy()
                # Set both tokens to 0 (but keep bias=1)
                X_masked[:, i, :-1] = 0
                X_masked[:, j, :-1] = 0
                
                X_masked_flat = X_masked.reshape(n_samples, -1)
                X_masked_tensor = torch.FloatTensor(X_masked_flat)
                
                y_pred_masked = model(X_masked_tensor).squeeze().numpy()
                r2_masked = r2_score(y, y_pred_masked)
                r2_pair_masked.append(r2_masked)
            
            results['pair_tokens_masked'] = {
                'r2_per_pair': [float(r2) for r2 in r2_pair_masked],
                'r2_mean': float(np.mean(r2_pair_masked)),
                'r2_std': float(np.std(r2_pair_masked)),
                'n_pairs': len(r2_pair_masked),
                'description': 'Two tokens masked at a time (for order-2 Shapley)'
            }
        
        # 4. Random masking patterns
        r2_random_masked = []
        for _ in range(20):  # 20 random patterns
            # Randomly choose how many tokens to mask (1 to n_tokens-1)
            n_mask = np.random.randint(1, n_tokens)
            mask_indices = np.random.choice(n_tokens, n_mask, replace=False)
            
            X_masked = X_with_bias.copy()
            for idx in mask_indices:
                X_masked[:, idx, :-1] = 0  # Keep bias
            
            X_masked_flat = X_masked.reshape(n_samples, -1)
            X_masked_tensor = torch.FloatTensor(X_masked_flat)
            
            y_pred_masked = model(X_masked_tensor).squeeze().numpy()
            r2_masked = r2_score(y, y_pred_masked)
            r2_random_masked.append(r2_masked)
        
        results['random_masked'] = {
            'r2_values': [float(r2) for r2 in r2_random_masked],
            'r2_mean': float(np.mean(r2_random_masked)),
            'r2_std': float(np.std(r2_random_masked)),
            'description': 'Random masking patterns'
        }
        
        # 5. All tokens masked (zero baseline)
        X_zero = X_with_bias.copy()
        X_zero[:, :, :-1] = 0  # All features to 0, keep bias=1
        
        X_zero_flat = X_zero.reshape(n_samples, -1)
        X_zero_tensor = torch.FloatTensor(X_zero_flat)
        
        y_pred_zero = model(X_zero_tensor).squeeze().numpy()
        r2_zero = r2_score(y, y_pred_zero)
        
        results['all_masked'] = {
            'r2': float(r2_zero),
            'mse': float(mean_squared_error(y, y_pred_zero)),
            'prediction_mean': float(np.mean(y_pred_zero)),
            'prediction_std': float(np.std(y_pred_zero)),
            'description': 'All tokens masked (zero baseline for Shapley)'
        }
    
    # Print summary
    print(f"  Full data R²: {results['full_data']['r2']:.4f}")
    print(f"  Single token masked R²: {results['single_token_masked']['r2_mean']:.4f} ± {results['single_token_masked']['r2_std']:.4f}")
    if 'pair_tokens_masked' in results:
        print(f"  Pair tokens masked R²: {results['pair_tokens_masked']['r2_mean']:.4f} ± {results['pair_tokens_masked']['r2_std']:.4f}")
    print(f"  Random masked R²: {results['random_masked']['r2_mean']:.4f} ± {results['random_masked']['r2_std']:.4f}")
    print(f"  All masked (zero baseline) R²: {results['all_masked']['r2']:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate TN-tree with higher-order Shapley values')
    parser.add_argument('--result', type=str, required=True, help='Path to training result JSON file')
    parser.add_argument('--output-dir', type=str, default='./tn_shapley_eval', help='Output directory')
    parser.add_argument('--orders', type=int, nargs='+', default=[1, 2], help='Shapley orders to compute')
    parser.add_argument('--max-subsets', type=int, default=None, help='Max subsets per order (for efficiency)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("TN-TREE SHAPLEY EVALUATION")
    print("=" * 80)
    print(f"Result file: {args.result}")
    print(f"Output directory: {args.output_dir}")
    print(f"Orders: {args.orders}")
    
    try:
        # Load dataset and model
        print("\n" + "=" * 50)
        print("LOADING DATASET AND MODEL")
        print("=" * 50)
        
        X, y, tokens, model, training_results = load_dataset_and_model(args.result)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Targets shape: {y.shape}")
        print(f"Tokens: {tokens}")
        
        # Evaluate R² on different masking patterns
        print("\n" + "=" * 50)
        print("EVALUATING MASKED DATA R²")
        print("=" * 50)
        
        masking_results = evaluate_masked_data_r2(model, X, y, tokens)
        
        # Prepare model prediction function
        def predict_fn(X_input):
            """Prediction function for Shapley computation"""
            if X_input.ndim == 1:
                X_input = X_input.reshape(1, -1)
            X_tensor = torch.FloatTensor(X_input)
            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor).squeeze().numpy()
                # Ensure we always return an array, even for single predictions
                if predictions.ndim == 0:
                    predictions = np.array([predictions])
                return predictions
        
        # Compute Shapley values for each order
        print("\n" + "=" * 50)
        print("COMPUTING SHAPLEY VALUES")
        print("=" * 50)
        
        shapley_results = {}
        
        # Use a representative sample for Shapley computation (first sample)
        sample_idx = 0
        x_sample = X[sample_idx]  # Shape: [n_tokens, embed_dim]
        
        # Add bias and flatten
        bias_dim = np.ones((1, len(tokens), 1))
        x_sample_with_bias = np.concatenate([x_sample.reshape(1, len(tokens), -1), bias_dim], axis=2)
        x_flat = x_sample_with_bias.flatten()
        
        print(f"Computing Shapley for sample {sample_idx}")
        print(f"Sample shape: {x_flat.shape}")
        print(f"Target value: {y[sample_idx]:.4f}")
        
        for order in args.orders:
            print(f"\nComputing order-{order} Shapley values...")
            
            shapley_values, comp_time = exact_k_sii_zero_batch(
                predict_fn, x_flat, order, 
                max_subsets=args.max_subsets,
                batch_size=args.batch_size
            )
            
            print(f"  Computed {len(shapley_values)} values")
            print(f"  Values range: [{shapley_values.min():.6f}, {shapley_values.max():.6f}]")
            print(f"  Values mean: {shapley_values.mean():.6f}")
            print(f"  Values std: {shapley_values.std():.6f}")
            print(f"  Computation time: {comp_time:.2f} seconds")
            
            # Store results
            shapley_results[str(order)] = {
                'values': ensure_json_serializable(shapley_values),
                'computation_time': float(comp_time),
                'n_values': len(shapley_values),
                'sample_idx': sample_idx,
                'sample_target': float(y[sample_idx]),
                'statistics': {
                    'mean': float(shapley_values.mean()),
                    'std': float(shapley_values.std()),
                    'min': float(shapley_values.min()),
                    'max': float(shapley_values.max())
                }
            }
            
            # For order 1, map to tokens
            if order == 1 and len(shapley_values) >= len(tokens):
                token_shapley = {}
                features_per_token = (len(x_flat) - len(tokens)) // len(tokens)  # Exclude bias
                
                for i, token in enumerate(tokens):
                    # Get Shapley values for this token's features
                    start_idx = i * (features_per_token + 1)  # +1 for bias
                    end_idx = start_idx + features_per_token  # Don't include bias in Shapley
                    
                    if end_idx <= len(shapley_values):
                        token_values = shapley_values[start_idx:end_idx]
                        token_shapley[token] = {
                            'values': ensure_json_serializable(token_values),
                            'mean': float(token_values.mean()),
                            'sum': float(token_values.sum())
                        }
                
                shapley_results[str(order)]['token_shapley'] = token_shapley
        
        # Save results
        print("\n" + "=" * 50)
        print("SAVING RESULTS")
        print("=" * 50)
        
        # Prepare final results
        eval_results = {
            'source_result': args.result,
            'sentence_text': training_results.get('sentence_text', ''),
            'tokens': tokens,
            'evaluation_config': {
                'orders': args.orders,
                'max_subsets': args.max_subsets,
                'batch_size': args.batch_size,
                'sample_idx': sample_idx
            },
            'masking_evaluation': masking_results,
            'shapley_values': shapley_results,
            'training_metrics': training_results.get('final_metrics', {}),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save JSON results
        sentence = training_results.get('sentence_text', 'unknown')
        safe_filename = sentence.replace(' ', '_').replace('"', '').replace("'", '').replace('.', '')
        json_path = os.path.join(args.output_dir, f'{safe_filename}_shapley_eval.json')
        
        with open(json_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Evaluation results saved to: {json_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Sentence: {sentence}")
        print(f"Training R²: {training_results.get('final_metrics', {}).get('test_r2', 'N/A'):.4f}")
        print(f"Full data R²: {masking_results['full_data']['r2']:.4f}")
        print(f"Single token masked R²: {masking_results['single_token_masked']['r2_mean']:.4f}")
        print(f"Zero baseline R²: {masking_results['all_masked']['r2']:.4f}")
        
        for order in args.orders:
            if str(order) in shapley_results:
                stats = shapley_results[str(order)]['statistics']
                print(f"Order-{order} Shapley: {shapley_results[str(order)]['n_values']} values, "
                      f"mean={stats['mean']:.6f}, range=[{stats['min']:.6f}, {stats['max']:.6f}]")
        
        print(f"\nResults saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
