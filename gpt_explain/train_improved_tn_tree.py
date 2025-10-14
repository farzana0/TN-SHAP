#!/usr/bin/env python3
"""
Improved TN-tree training with separate R² evaluation for masked data.

This script:
1. Loads improved synthetic datasets
2. Trains TN-tree models with better architecture
3. Reports R² scores separately for full data and masked data
4. Computes TN-SHAP values with enhanced evaluation
5. Uses better training strategies for higher R² scores

Key improvements:
- Separate R² evaluation for masked vs unmasked data
- Better model architecture and training
- Enhanced convergence criteria
- Improved data preprocessing
- Detailed performance analysis

Created: September 26, 2025
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import sys
from typing import Dict, List, Tuple, Optional
import time

# Add the parent directory to sys.path to import tntree_model
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

def load_improved_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str, List[Dict]]:
    """Load improved synthetic dataset"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Dataset: {data['sentence']}")
    print(f"Tokens: {data['token_info']['token_strings']}")
    print(f"Number of tokens: {len(data['token_info']['token_strings'])}")
    
    X_list = []
    y_list = []
    masks_list = []
    
    # Load main training data
    for order, order_data in data['data'].items():
        samples = order_data['samples']
        print(f"Processing order {order}: {len(samples)} samples")
        
        for sample in samples:
            embeddings = np.array(sample['embeddings'])  # [n_tokens, embed_dim]
            mask = np.array(sample['mask'])  # [n_tokens]
            target_logit = sample['target_logit']
            
            X_list.append(embeddings)
            y_list.append(target_logit)
            masks_list.append(mask)
    
    # Stack all samples
    X = np.stack(X_list)  # [n_samples, n_tokens, embed_dim]
    y = np.array(y_list)
    masks = np.stack(masks_list)  # [n_samples, n_tokens]
    
    print(f"Combined input shape: {X.shape}")
    print(f"Combined targets shape: {y.shape}")
    print(f"Combined masks shape: {masks.shape}")
    print(f"Targets range: [{y.min():.4f}, {y.max():.4f}]")
    
    tokens = data['token_info']['token_strings']
    sentence_text = data['sentence']
    
    # Load masked evaluation data
    masked_eval_samples = []
    if 'masked_evaluation' in data:
        masked_eval_samples = data['masked_evaluation']['samples']
        print(f"Masked evaluation samples: {len(masked_eval_samples)}")
    
    return X, y, masks, tokens, sentence_text, masked_eval_samples

def prepare_data_for_training(X: np.ndarray, y: np.ndarray, masks: np.ndarray, 
                            normalize_targets: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """Prepare data for TN-tree training"""
    
    n_samples, n_tokens, embed_dim = X.shape
    print(f"\nPreparing data for training:")
    print(f"  Input shape: {X.shape}")
    print(f"  Targets: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Apply masks to embeddings (zero out masked positions)
    X_masked = X * masks[:, :, np.newaxis]  # Broadcasting masks
    
    # Add bias dimension per token
    bias_dim = np.ones((n_samples, n_tokens, 1))
    X_with_bias = np.concatenate([X_masked, bias_dim], axis=2)
    
    # Normalize targets if requested
    target_scaler = None
    y_processed = y.copy()
    if normalize_targets:
        target_scaler = StandardScaler()
        y_processed = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        print(f"  Normalized targets: mean={y_processed.mean():.4f}, std={y_processed.std():.4f}")
    
    print(f"  Final input shape: {X_with_bias.shape}")
    
    return X_with_bias, y_processed, target_scaler

def train_improved_tn_tree(X: np.ndarray, y: np.ndarray, tokens: List[str],
                          rank: int = 4, max_epochs: int = 100,
                          target_r2: float = 0.7, patience: int = 15,
                          learning_rate: float = 0.01) -> Tuple[BinaryTensorTree, Dict]:
    """Train TN-tree with improved settings"""
    
    n_samples, n_tokens, embed_dim_plus_bias = X.shape
    embed_dim = embed_dim_plus_bias - 1
    
    print(f"\nTraining improved TN-tree:")
    print(f"  Rank: {rank}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Target R²: {target_r2}")
    print(f"  Patience: {patience}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Samples: {n_samples}")
    print(f"  Tokens: {n_tokens}")
    print(f"  Embed dim per token: {embed_dim}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create TN-tree model with smaller rank to avoid overfitting
    leaf_phys_dims = [embed_dim + 1] * n_tokens
    leaf_input_dims = [embed_dim + 1] * n_tokens
    
    model = BinaryTensorTree(
        leaf_phys_dims=leaf_phys_dims,
        leaf_input_dims=leaf_input_dims,
        ranks=rank,  # Use smaller rank
        assume_bias_when_matrix=True,
        device='cpu'
    )
    
    print(f"  Model created with leaf_phys_dims: {leaf_phys_dims}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training setup with better regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                           weight_decay=1e-3, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=8, factor=0.7, verbose=True, min_lr=1e-6
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).reshape(X_train.shape[0], -1)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).reshape(X_test.shape[0], -1)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Training loop
    train_history = {
        'train_loss': [],
        'test_loss': [],
        'train_r2': [],
        'test_r2': [],
        'best_epoch': 0,
        'best_r2': -np.inf,
        'learning_rates': []
    }
    
    best_model_state = None
    epochs_without_improvement = 0
    
    print("\nTraining progress:")
    print("Epoch | Train Loss | Test Loss | Train R² | Test R² | LR")
    print("-" * 65)
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        train_pred = model(X_train_tensor)
        train_loss = criterion(train_pred.squeeze(), y_train_tensor)
        
        # Add L2 regularization manually if needed
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        train_loss += 1e-5 * l2_reg
        
        train_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor)
            test_loss = criterion(test_pred.squeeze(), y_test_tensor)
            
            # Calculate R² scores
            train_pred_np = train_pred.squeeze().detach().numpy()
            test_pred_np = test_pred.squeeze().detach().numpy()
            
            train_r2 = r2_score(y_train, train_pred_np)
            test_r2 = r2_score(y_test, test_pred_np)
        
        # Update learning rate
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        train_history['train_loss'].append(float(train_loss))
        train_history['test_loss'].append(float(test_loss))
        train_history['train_r2'].append(float(train_r2))
        train_history['test_r2'].append(float(test_r2))
        train_history['learning_rates'].append(current_lr)
        
        # Check for best model
        if test_r2 > train_history['best_r2']:
            train_history['best_r2'] = float(test_r2)
            train_history['best_epoch'] = epoch
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Print progress
        if epoch % 10 == 0 or epoch < 10:
            print(f"{epoch:5d} | {train_loss:.6f} | {test_loss:.6f} | {train_r2:.4f} | {test_r2:.4f} | {current_lr:.2e}")
        
        # Early stopping criteria
        if test_r2 >= target_r2:
            print(f"\nTarget R² {target_r2:.4f} achieved! Stopping early.")
            break
            
        if epochs_without_improvement >= patience:
            print(f"\nNo improvement for {patience} epochs. Stopping early.")
            break
            
        if current_lr < 1e-6:
            print(f"\nLearning rate too small ({current_lr:.2e}). Stopping.")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model from epoch {train_history['best_epoch']} with R² = {train_history['best_r2']:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_train_pred = model(X_train_tensor).squeeze().detach().numpy()
        final_test_pred = model(X_test_tensor).squeeze().detach().numpy()
        
        final_train_r2 = r2_score(y_train, final_train_pred)
        final_test_r2 = r2_score(y_test, final_test_pred)
        final_train_mse = mean_squared_error(y_train, final_train_pred)
        final_test_mse = mean_squared_error(y_test, final_test_pred)
    
    print(f"\nFinal Results:")
    print(f"  Train R²: {final_train_r2:.4f}, MSE: {final_train_mse:.6f}")
    print(f"  Test R²: {final_test_r2:.4f}, MSE: {final_test_mse:.6f}")
    
    # Add final metrics to history
    train_history['final_train_r2'] = float(final_train_r2)
    train_history['final_test_r2'] = float(final_test_r2)
    train_history['final_train_mse'] = float(final_train_mse)
    train_history['final_test_mse'] = float(final_test_mse)
    
    return model, train_history

def evaluate_masked_data_r2(model: BinaryTensorTree, masked_eval_samples: List[Dict], 
                           target_scaler: Optional[StandardScaler] = None) -> Dict:
    """Evaluate R² specifically on masked data samples"""
    
    print(f"\nEvaluating R² on masked data:")
    print(f"  Total masked samples: {len(masked_eval_samples)}")
    
    if not masked_eval_samples:
        print("  No masked evaluation samples available")
        return {}
    
    # Prepare masked data
    X_masked = []
    y_masked = []
    mask_info = []
    
    for sample in masked_eval_samples:
        embeddings = np.array(sample['embeddings'])  # [n_tokens, embed_dim]
        mask = np.array(sample['mask'])  # [n_tokens]
        target = sample['target_logit']
        
        # Apply mask
        masked_embeddings = embeddings * mask[:, np.newaxis]
        
        # Add bias
        n_tokens, embed_dim = embeddings.shape
        bias_dim = np.ones((n_tokens, 1))
        embeddings_with_bias = np.concatenate([masked_embeddings, bias_dim], axis=1)
        
        X_masked.append(embeddings_with_bias)
        y_masked.append(target)
        mask_info.append({
            'mask_type': sample['mask_type'],
            'n_masked': sample['n_masked']
        })
    
    X_masked = np.stack(X_masked)  # [n_samples, n_tokens, embed_dim+1]
    y_masked = np.array(y_masked)
    
    # Normalize targets if scaler was used during training
    if target_scaler is not None:
        y_masked_norm = target_scaler.transform(y_masked.reshape(-1, 1)).flatten()
    else:
        y_masked_norm = y_masked
    
    # Convert to tensor and predict
    X_masked_tensor = torch.FloatTensor(X_masked).reshape(X_masked.shape[0], -1)
    
    model.eval()
    with torch.no_grad():
        pred_masked = model(X_masked_tensor).squeeze().detach().numpy()
    
    # Calculate overall R²
    overall_r2 = r2_score(y_masked_norm, pred_masked)
    overall_mse = mean_squared_error(y_masked_norm, pred_masked)
    
    print(f"  Overall masked data R²: {overall_r2:.4f}")
    print(f"  Overall masked data MSE: {overall_mse:.6f}")
    
    # Calculate R² by mask type
    mask_type_results = {}
    mask_types = set(info['mask_type'] for info in mask_info)
    
    for mask_type in mask_types:
        indices = [i for i, info in enumerate(mask_info) if info['mask_type'] == mask_type]
        if len(indices) > 1:  # Need at least 2 samples for R²
            y_type = y_masked_norm[indices]
            pred_type = pred_masked[indices]
            
            r2_type = r2_score(y_type, pred_type)
            mse_type = mean_squared_error(y_type, pred_type)
            
            mask_type_results[mask_type] = {
                'r2': float(r2_type),
                'mse': float(mse_type),
                'n_samples': len(indices)
            }
            
            print(f"  {mask_type} R²: {r2_type:.4f} (MSE: {mse_type:.6f}, n={len(indices)})")
    
    # Calculate R² by number of masked tokens
    masked_count_results = {}
    masked_counts = set(info['n_masked'] for info in mask_info)
    
    for n_masked in sorted(masked_counts):
        indices = [i for i, info in enumerate(mask_info) if info['n_masked'] == n_masked]
        if len(indices) > 1:
            y_count = y_masked_norm[indices]
            pred_count = pred_masked[indices]
            
            r2_count = r2_score(y_count, pred_count)
            mse_count = mean_squared_error(y_count, pred_count)
            
            masked_count_results[str(n_masked)] = {
                'r2': float(r2_count),
                'mse': float(mse_count),
                'n_samples': len(indices)
            }
            
            print(f"  {n_masked} tokens masked R²: {r2_count:.4f} (MSE: {mse_count:.6f}, n={len(indices)})")
    
    return {
        'overall_r2': float(overall_r2),
        'overall_mse': float(overall_mse),
        'by_mask_type': mask_type_results,
        'by_masked_count': masked_count_results,
        'total_samples': len(masked_eval_samples)
    }

def compute_improved_tn_shap(model: BinaryTensorTree, X: np.ndarray, tokens: List[str],
                           target_scaler: Optional[StandardScaler] = None,
                           orders: List[int] = [1, 2]) -> Dict:
    """Compute TN-SHAP values with improved evaluation"""
    
    n_samples, n_tokens, embed_dim_plus_bias = X.shape
    
    print(f"\nComputing improved TN-SHAP values:")
    print(f"  Orders: {orders}")
    print(f"  Samples: {n_samples}, Tokens: {n_tokens}")
    
    shapley_results = {}
    
    for order in orders:
        print(f"\nComputing order {order} Shapley values...")
        start_time = time.time()
        
        with torch.no_grad():
            model.eval()
            X_tensor = torch.FloatTensor(X)
            
            if order == 1:
                # Single token contributions
                values = []
                interactions = []
                
                # Baseline: all zeros except bias (last dimension = 1)
                baseline = torch.zeros_like(X_tensor)
                baseline[:, :, -1] = 1.0  # Keep bias terms as 1
                baseline_flat = baseline.reshape(baseline.shape[0], -1)
                baseline_pred = model(baseline_flat).squeeze().detach().numpy()
                
                print(f"    Baseline prediction range: [{baseline_pred.min():.6f}, {baseline_pred.max():.6f}]")
                
                for i in range(n_tokens):
                    # Set only token i to its actual value (keeping bias = 1)
                    intervention = baseline.clone()
                    intervention[:, i, :] = X_tensor[:, i, :]
                    intervention[:, i, -1] = 1.0  # Ensure bias stays 1
                    intervention_flat = intervention.reshape(intervention.shape[0], -1)
                    intervention_pred = model(intervention_flat).squeeze().detach().numpy()
                    
                    # Shapley value for token i
                    shapley_val = np.mean(intervention_pred - baseline_pred)
                    values.append(shapley_val)
                    interactions.append([i])
                    
                    print(f"    Token {i} ({tokens[i]}) contribution: {shapley_val:.6f}")
                    
            elif order == 2:
                # Pairwise interactions
                values = []
                interactions = []
                
                # Baseline and single effects (same as order 1)
                baseline = torch.zeros_like(X_tensor)
                baseline[:, :, -1] = 1.0  # Keep bias terms as 1
                baseline_flat = baseline.reshape(baseline.shape[0], -1)
                baseline_pred = model(baseline_flat).squeeze().detach().numpy()
                
                single_effects = {}
                for i in range(n_tokens):
                    intervention = baseline.clone()
                    intervention[:, i, :] = X_tensor[:, i, :]
                    intervention[:, i, -1] = 1.0  # Ensure bias stays 1
                    intervention_flat = intervention.reshape(intervention.shape[0], -1)
                    intervention_pred = model(intervention_flat).squeeze().detach().numpy()
                    single_effects[i] = np.mean(intervention_pred - baseline_pred)
                
                # Pairwise effects
                for i in range(n_tokens):
                    for j in range(i + 1, n_tokens):
                        # Both tokens i and j
                        intervention_both = baseline.clone()
                        intervention_both[:, i, :] = X_tensor[:, i, :]
                        intervention_both[:, j, :] = X_tensor[:, j, :]
                        intervention_both[:, i, -1] = 1.0  # Ensure bias stays 1
                        intervention_both[:, j, -1] = 1.0  # Ensure bias stays 1
                        intervention_both_flat = intervention_both.reshape(intervention_both.shape[0], -1)
                        pred_both = model(intervention_both_flat).squeeze().detach().numpy()
                        
                        # Interaction = combined effect - individual effects
                        interaction_val = (np.mean(pred_both - baseline_pred) - 
                                         single_effects[i] - single_effects[j])
                        
                        values.append(interaction_val)
                        interactions.append([i, j])
                        
                        print(f"    Interaction {i}-{j} ({tokens[i]}-{tokens[j]}): {interaction_val:.6f}")
        
        values = np.array(values)
        
        computation_time = time.time() - start_time
        print(f"  Computation time: {computation_time:.2f} seconds")
        print(f"  Values range: [{values.min():.6f}, {values.max():.6f}]")
        print(f"  Values mean: {values.mean():.6f}, std: {values.std():.6f}")
        
        shapley_results[str(order)] = {
            'values': ensure_json_serializable(values),
            'interactions': ensure_json_serializable(interactions),
            'computation_time': computation_time,
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max())
        }
    
    return shapley_results

def main():
    parser = argparse.ArgumentParser(description='Improved TN-tree training with masked data evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Path to improved dataset JSON file')
    parser.add_argument('--output-dir', type=str, default='./improved_tn_results', help='Output directory')
    parser.add_argument('--rank', type=int, default=4, help='TN-tree rank')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--target-r2', type=float, default=0.7, help='Target R² score for early stopping')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--normalize-targets', action='store_true', help='Normalize target values')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("IMPROVED TN-TREE TRAINING WITH MASKED DATA EVALUATION")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Rank: {args.rank}, Max epochs: {args.max_epochs}")
    print(f"Target R²: {args.target_r2}, Patience: {args.patience}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Normalize targets: {args.normalize_targets}")
    
    try:
        # Load dataset
        print("\n" + "=" * 50)
        print("LOADING IMPROVED DATASET")
        print("=" * 50)
        
        X, y, masks, tokens, sentence_text, masked_eval_samples = load_improved_dataset(args.dataset)
        
        # Prepare data
        print("\n" + "=" * 50)
        print("PREPARING DATA")
        print("=" * 50)
        
        X_processed, y_processed, target_scaler = prepare_data_for_training(
            X, y, masks, normalize_targets=args.normalize_targets
        )
        
        # Train TN-tree
        print("\n" + "=" * 50)
        print("TRAINING IMPROVED TN-TREE")
        print("=" * 50)
        
        model, train_history = train_improved_tn_tree(
            X_processed, y_processed, tokens,
            rank=args.rank,
            max_epochs=args.max_epochs,
            target_r2=args.target_r2,
            patience=args.patience,
            learning_rate=args.learning_rate
        )
        
        # Evaluate on masked data
        print("\n" + "=" * 50)
        print("EVALUATING MASKED DATA R²")
        print("=" * 50)
        
        masked_r2_results = evaluate_masked_data_r2(model, masked_eval_samples, target_scaler)
        
        # Compute Shapley values
        print("\n" + "=" * 50)
        print("COMPUTING TN-SHAP VALUES")
        print("=" * 50)
        
        shapley_values = compute_improved_tn_shap(model, X_processed, tokens, target_scaler)
        
        # Save results
        print("\n" + "=" * 50)
        print("SAVING RESULTS")
        print("=" * 50)
        
        results = {
            'sentence_text': sentence_text,
            'tokens': tokens,
            'dataset_path': args.dataset,
            'model_config': {
                'rank': args.rank,
                'max_epochs': args.max_epochs,
                'target_r2': args.target_r2,
                'patience': args.patience,
                'learning_rate': args.learning_rate,
                'normalize_targets': args.normalize_targets,
                'leaf_phys_dims': [X_processed.shape[2]] * len(tokens)
            },
            'training_history': ensure_json_serializable(train_history),
            'masked_data_r2': masked_r2_results,
            'shapley_values': shapley_values,
            'final_metrics': {
                'train_r2': float(train_history['final_train_r2']),
                'test_r2': float(train_history['final_test_r2']),
                'train_mse': float(train_history['final_train_mse']),
                'test_mse': float(train_history['final_test_mse'])
            },
            'data_statistics': {
                'n_samples': len(X),
                'n_tokens': len(tokens),
                'embed_dim': X.shape[2],
                'target_range': [float(y.min()), float(y.max())],
                'target_mean': float(y.mean()),
                'target_std': float(y.std())
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save JSON results
        safe_filename = sentence_text.replace(' ', '_').replace(',', '').replace('.', '')
        json_path = os.path.join(args.output_dir, f'{safe_filename}_improved_results.json')
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {json_path}")
        
        # Save model
        model_path = os.path.join(args.output_dir, f'{safe_filename}_improved_model.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        
        # Print summary
        final_test_r2 = train_history['final_test_r2']
        masked_overall_r2 = masked_r2_results.get('overall_r2', 0.0)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE - SUMMARY")
        print("=" * 80)
        print(f"Final Test R²: {final_test_r2:.4f}")
        print(f"Masked Data R²: {masked_overall_r2:.4f}")
        print(f"Training Quality: {'EXCELLENT' if final_test_r2 >= 0.8 else 'GOOD' if final_test_r2 >= 0.5 else 'NEEDS IMPROVEMENT'}")
        print(f"Masked Data Quality: {'EXCELLENT' if masked_overall_r2 >= 0.8 else 'GOOD' if masked_overall_r2 >= 0.5 else 'NEEDS IMPROVEMENT'}")
        
        if masked_r2_results:
            print(f"\nMasked Data Breakdown:")
            for mask_type, results in masked_r2_results.get('by_mask_type', {}).items():
                print(f"  {mask_type}: R² = {results['r2']:.4f} (n={results['n_samples']})")
        
        print(f"\nResults saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
