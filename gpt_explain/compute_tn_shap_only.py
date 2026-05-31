#!/usr/bin/env python3
"""
TN-SHAP evaluation script - uses only tensor network Shapley computation.
No exact computation, no KernelSHAP - only TN-based Shapley values.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# Add paths for imports
sys.path.append('./higher_order_iclr')
from tntree_model import BinaryTensorTree

def chebyshev_nodes_01(m, device='cpu'):
    """Generate Chebyshev nodes on [0,1] interval"""
    k = torch.arange(1, m + 1, dtype=torch.float32, device=device)
    nodes = 0.5 * (1 + torch.cos((2 * k - 1) * np.pi / (2 * m)))
    return nodes

def load_dataset_and_model(result_path: str):
    """Load dataset and trained model from result JSON"""
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    # Extract dataset path and load data
    dataset_path = results['dataset_path']
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loading dataset: {dataset['sentence']}")
    print(f"Tokens: {dataset['token_info']['token_strings']}")
    
    X_list = []
    y_list = []
    
    for order, order_data in dataset['data'].items():
        samples = order_data['samples']
        
        for sample in samples:
            embeddings = sample['embeddings']  # This might be 2D or 3D
            target_logit = sample['target_logit']
            
            # Handle different embedding formats
            embeddings_array = np.array(embeddings)
            
            if embeddings_array.ndim == 2:
                # Format: [n_tokens, embed_dim] - single sample per entry
                X_sample = embeddings_array[:, :16]  # Take first 16 dimensions
                X_list.append(X_sample)
                y_list.append(target_logit)
            elif embeddings_array.ndim == 3:
                # Format: [n_noise, n_tokens, embed_dim] - multiple noise samples
                n_noise_samples = embeddings_array.shape[0]
                for i in range(n_noise_samples):
                    X_sample = embeddings_array[i, :, :16]
                    X_list.append(X_sample)
                    y_list.append(target_logit)
            else:
                print(f"Warning: Unexpected embedding shape {embeddings_array.shape}")
                continue
    
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
    
    model = BinaryTensorTree(
        leaf_phys_dims=leaf_phys_dims,
        leaf_input_dims=leaf_phys_dims,
        ranks=model_config['rank'],
        assume_bias_when_matrix=True,
        device='cpu'
    )
    
    # Load model weights
    model_path = result_path.replace('_results.json', '_model_masked.pt').replace('_tn_masked_results.json', '_tn_model_masked.pt')
    if not os.path.exists(model_path):
        # Try alternative naming patterns
        base_path = result_path.replace('_tn_masked_results.json', '')
        model_path = base_path + '_tn_model_masked.pt'
        if not os.path.exists(model_path):
            # Try without "masked" suffix
            model_path = result_path.replace('.json', '.pt')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded model weights from: {model_path}")
    else:
        print(f"Warning: Model weights not found. Tried: {model_path}")
        print("Using random weights - results will be meaningless!")
    
    return X, y, tokens, model, results

@torch.no_grad()
def compute_tn_shapley_values(model, x_sample, tokens, device='cpu'):
    """
    Compute TN-SHAP values for order 1 (individual features) using the tensor network method.
    This is the core TN-SHAP computation without any baselines.
    """
    print("Computing TN-SHAP values (order 1)...")
    
    model.eval()
    n_tokens = len(tokens)
    embed_dim = x_sample.shape[-1] - 1  # Subtract bias dimension
    
    # Flatten the input sample
    x_flat = torch.as_tensor(x_sample, device=device, dtype=torch.float32).flatten()
    d = x_flat.numel()
    
    shapley_values = torch.zeros(d, device=device, dtype=torch.float32)
    
    print(f"  Computing Shapley values for {d} features across {n_tokens} tokens...")
    
    # For each feature, compute its Shapley value using polynomial approximation
    for feature_idx in range(d):
        if feature_idx % 20 == 0:
            print(f"    Processing feature {feature_idx}/{d}")
        
        # Number of Chebyshev nodes (polynomial degree)
        m = min(15, d)  # Reasonable polynomial degree
        
        # Generate Chebyshev nodes on [0,1]
        t = chebyshev_nodes_01(m, device=device)
        
        # Create input variations
        # X_g: scaled versions of the input
        X_g = t.unsqueeze(1) * x_flat.unsqueeze(0)  # Shape: [m, d]
        
        # X_h: same as X_g but with feature_idx set to 0
        X_h = X_g.clone()
        X_h[:, feature_idx] = 0.0
        
        # Combine for batch evaluation
        X_batch = torch.cat([X_g, X_h], dim=0)  # Shape: [2*m, d]
        
        # Get model predictions
        y_batch = model(X_batch).squeeze(-1)
        y_g, y_h = y_batch[:m], y_batch[m:]
        
        # Compute difference
        delta_y = y_g - y_h
        
        # Set up Vandermonde matrix for polynomial fitting
        V = torch.vander(t, N=m, increasing=True)  # Shape: [m, m]
        
        # Solve for polynomial coefficients
        try:
            # Solve V @ c = delta_y
            coeffs = torch.linalg.solve(V, delta_y.unsqueeze(1)).squeeze(1)
        except RuntimeError:
            # Fallback to least squares if singular
            coeffs = torch.linalg.lstsq(V, delta_y.unsqueeze(1)).solution.squeeze(1)
        
        # Compute Shapley value as integral of polynomial
        # Shapley value = integral of polynomial * (1/k) for k >= 1
        inv_k = torch.zeros_like(coeffs)
        inv_k[1:] = 1.0 / torch.arange(1, m, device=device, dtype=torch.float32)
        
        shapley_values[feature_idx] = torch.sum(coeffs * inv_k)
    
    # Convert to numpy and organize by tokens
    shapley_np = shapley_values.detach().cpu().numpy()
    
    # Organize Shapley values by tokens
    features_per_token = (embed_dim + 1)  # Including bias
    token_shapley = {}
    
    for token_idx, token in enumerate(tokens):
        start_idx = token_idx * features_per_token
        end_idx = start_idx + embed_dim  # Exclude bias from Shapley values
        
        if end_idx <= len(shapley_np):
            token_values = shapley_np[start_idx:end_idx]
            token_shapley[token] = {
                'values': token_values.tolist(),
                'mean': float(token_values.mean()),
                'sum': float(token_values.sum()),
                'abs_mean': float(np.abs(token_values).mean()),
                'token_index': token_idx
            }
    
    return shapley_np, token_shapley

@torch.no_grad() 
def compute_tn_shapley_interactions(model, x_sample, tokens, max_pairs=20, device='cpu'):
    """
    Compute TN-SHAP interaction values for order 2 (pairwise interactions).
    """
    print("Computing TN-SHAP interaction values (order 2)...")
    
    model.eval()
    n_tokens = len(tokens)
    embed_dim = x_sample.shape[-1] - 1
    
    x_flat = torch.as_tensor(x_sample, device=device, dtype=torch.float32).flatten()
    d = x_flat.numel()
    
    # For computational efficiency, focus on token-level interactions
    features_per_token = embed_dim + 1
    
    # Generate token pairs for interaction computation
    token_pairs = []
    for i in range(n_tokens):
        for j in range(i + 1, n_tokens):
            token_pairs.append((i, j))
    
    if len(token_pairs) > max_pairs:
        # Sample random pairs if too many
        import random
        token_pairs = random.sample(token_pairs, max_pairs)
    
    print(f"  Computing interactions for {len(token_pairs)} token pairs...")
    
    interaction_values = {}
    
    for pair_idx, (token_i, token_j) in enumerate(token_pairs):
        if pair_idx % 5 == 0:
            print(f"    Processing token pair {pair_idx+1}/{len(token_pairs)}: {tokens[token_i]} x {tokens[token_j]}")
        
        # Get feature indices for both tokens (excluding bias)
        features_i = list(range(token_i * features_per_token, token_i * features_per_token + embed_dim))
        features_j = list(range(token_j * features_per_token, token_j * features_per_token + embed_dim))
        
        # Compute interaction for representative features (mean across embedding dims)
        m = 10  # Smaller polynomial degree for interactions
        t = chebyshev_nodes_01(m, device=device)
        
        # Create four evaluation points for interaction:
        # f(S ∪ {i,j}) - f(S ∪ {i}) - f(S ∪ {j}) + f(S)
        interaction_values_dim = []
        
        for feat_i in features_i[:min(3, len(features_i))]:  # Sample a few features per token
            for feat_j in features_j[:min(3, len(features_j))]:
                
                # Scaled input versions
                X_base = t.unsqueeze(1) * x_flat.unsqueeze(0)
                
                # Four variations:
                X_both = X_base.clone()  # S ∪ {i,j}
                
                X_i_only = X_base.clone()  # S ∪ {i}
                X_i_only[:, feat_j] = 0.0
                
                X_j_only = X_base.clone()  # S ∪ {j} 
                X_j_only[:, feat_i] = 0.0
                
                X_neither = X_base.clone()  # S
                X_neither[:, feat_i] = 0.0
                X_neither[:, feat_j] = 0.0
                
                # Batch evaluation
                X_all = torch.cat([X_both, X_i_only, X_j_only, X_neither], dim=0)
                y_all = model(X_all).squeeze(-1)
                
                y_both = y_all[:m]
                y_i_only = y_all[m:2*m]
                y_j_only = y_all[2*m:3*m]
                y_neither = y_all[3*m:]
                
                # Interaction effect
                interaction_effect = y_both - y_i_only - y_j_only + y_neither
                
                # Fit polynomial and compute integral
                V = torch.vander(t, N=m, increasing=True)
                try:
                    coeffs = torch.linalg.solve(V, interaction_effect.unsqueeze(1)).squeeze(1)
                except RuntimeError:
                    coeffs = torch.linalg.lstsq(V, interaction_effect.unsqueeze(1)).solution.squeeze(1)
                
                # Compute interaction Shapley value
                inv_k = torch.zeros_like(coeffs)
                inv_k[1:] = 1.0 / torch.arange(1, m, device=device, dtype=torch.float32)
                
                interaction_val = torch.sum(coeffs * inv_k).item()
                interaction_values_dim.append(interaction_val)
        
        # Average interaction across feature dimensions
        mean_interaction = np.mean(interaction_values_dim) if interaction_values_dim else 0.0
        
        pair_key = f"{tokens[token_i]} x {tokens[token_j]}"
        interaction_values[pair_key] = {
            'value': float(mean_interaction),
            'token_i': token_i,
            'token_j': token_j,
            'token_i_name': tokens[token_i],
            'token_j_name': tokens[token_j]
        }
    
    return interaction_values

def evaluate_model_performance(model, X, y, tokens):
    """Evaluate TN-student R² on different data patterns"""
    print("Evaluating TN-student R² performance...")
    
    n_samples, n_tokens, embed_dim = X.shape
    
    # Add bias and flatten
    bias_dim = np.ones((n_samples, n_tokens, 1))
    X_with_bias = np.concatenate([X, bias_dim], axis=2)
    X_flat = X_with_bias.reshape(n_samples, -1)
    X_tensor = torch.FloatTensor(X_flat)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).squeeze().numpy()
        
        general_r2 = r2_score(y, y_pred)
        
        print(f"  TN-student general R²: {general_r2:.4f}")
        
        # Additional metrics
        performance = {
            'general_r2': float(general_r2),
            'n_samples': n_samples,
            'prediction_range': [float(y_pred.min()), float(y_pred.max())],
            'target_range': [float(y.min()), float(y.max())]
        }
    
    return performance

def save_results(shapley_values, token_shapley, interaction_values, performance, tokens, sentence, output_dir):
    """Save TN-SHAP results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate safe filename
    safe_filename = sentence.replace(' ', '_').replace('"', '').replace("'", '').replace('.', '').replace(',', '_')
    
    results = {
        'sentence': sentence,
        'tokens': tokens,
        'tn_student_performance': performance,
        'shapley_values': {
            'raw_values': shapley_values.tolist() if hasattr(shapley_values, 'tolist') else shapley_values,
            'by_token': token_shapley,
            'statistics': {
                'mean': float(np.mean(shapley_values)) if hasattr(shapley_values, 'mean') else 0.0,
                'std': float(np.std(shapley_values)) if hasattr(shapley_values, 'std') else 0.0,
                'min': float(np.min(shapley_values)) if hasattr(shapley_values, 'min') else 0.0,
                'max': float(np.max(shapley_values)) if hasattr(shapley_values, 'max') else 0.0
            }
        },
        'interaction_values': interaction_values,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    json_path = os.path.join(output_dir, f'{safe_filename}_tn_shap_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"TN-SHAP results saved to: {json_path}")
    
    return json_path

def main():
    parser = argparse.ArgumentParser(description='Compute TN-SHAP values and interactions')
    parser.add_argument('--result', type=str, required=True, help='Path to training result JSON')
    parser.add_argument('--output-dir', type=str, default='./tn_shap_results', help='Output directory')
    parser.add_argument('--max-pairs', type=int, default=20, help='Maximum token pairs for interactions')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index for Shapley computation')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TN-SHAP COMPUTATION")
    print("=" * 80)
    print(f"Training result: {args.result}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Load dataset and model
        X, y, tokens, model, training_results = load_dataset_and_model(args.result)
        print(f"Dataset shape: {X.shape}")
        
        # Evaluate model performance
        performance = evaluate_model_performance(model, X, y, tokens)
        
        # Use representative sample for Shapley computation
        sample_idx = min(args.sample_idx, len(X) - 1)
        x_sample = X[sample_idx]
        
        # Add bias dimension
        bias_dim = np.ones((len(tokens), 1))
        x_sample_with_bias = np.concatenate([x_sample, bias_dim], axis=1)
        
        print(f"\nComputing TN-SHAP for sample {sample_idx}")
        print(f"Target value: {y[sample_idx]:.4f}")
        
        # Compute TN-SHAP values (order 1)
        shapley_values, token_shapley = compute_tn_shapley_values(model, x_sample_with_bias, tokens)
        
        print(f"TN-SHAP values computed:")
        print(f"  Range: [{shapley_values.min():.6f}, {shapley_values.max():.6f}]")
        print(f"  Mean: {shapley_values.mean():.6f}")
        
        # Compute TN-SHAP interactions (order 2)
        interaction_values = compute_tn_shapley_interactions(model, x_sample_with_bias, tokens, args.max_pairs)
        
        print(f"TN-SHAP interactions computed: {len(interaction_values)} pairs")
        
        # Save results
        sentence = training_results.get('sentence_text', 'unknown')
        result_path = save_results(shapley_values, token_shapley, interaction_values, performance, tokens, sentence, args.output_dir)
        
        print(f"\nTN-SHAP computation complete!")
        print(f"TN-student general R²: {performance['general_r2']:.4f}")
        
        return result_path
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
