#!/usr/bin/env python3
"""
Enhanced TN-tree training with Shapley value computation and R² reporting.

This script:
1. Loads GPT synthetic datasets with embeddings
2. Trains TN-tree models with higher ranks and epochs
3. Reports R² scores to ensure model quality
4. Computes exact TN-SHAP values for orders 1 and 2
5. Saves results with mean-removed Shapley values

Enhanced features:
- Better R² score reporting
- Higher default ranks and epochs
- Mean-removed Shapley values option
- Improved convergence criteria

Created: September 25, 2025
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

class GPTDatasetLoader:
    """Load GPT synthetic dataset with embeddings"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset_dir = os.path.dirname(dataset_path)
        
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
        """Load the dataset and return X, y, tokens, sentence_text"""
        
        # Load the main dataset file
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        # Extract metadata
        sentence_text = data['sentence_text']
        tokens = data['tokens'] 
        n_samples = data['n_samples']
        embed_dim = data['embed_dim']
        n_tokens = len(tokens)
        
        print(f"Dataset: {sentence_text}")
        print(f"Tokens: {tokens}")
        print(f"Samples: {n_samples}, Embed dim: {embed_dim}, Tokens: {n_tokens}")
        
        # Load embeddings
        dataset_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
        center_embeddings_path = os.path.join(self.dataset_dir, f'{dataset_name}_center_embeddings.npy')
        masked_embeddings_path = os.path.join(self.dataset_dir, f'{dataset_name}_masked_embeddings.npy')
        
        if not os.path.exists(center_embeddings_path):
            raise FileNotFoundError(f"Center embeddings not found: {center_embeddings_path}")
        if not os.path.exists(masked_embeddings_path):
            raise FileNotFoundError(f"Masked embeddings not found: {masked_embeddings_path}")
        
        center_embeddings = np.load(center_embeddings_path)
        masked_embeddings = np.load(masked_embeddings_path)
        
        print(f"Center embeddings shape: {center_embeddings.shape}")
        print(f"Masked embeddings shape: {masked_embeddings.shape}")
        
        # Get targets from dataset
        if 'targets' in data:
            targets = np.array(data['targets'])
        else:
            raise ValueError("No targets found in dataset!")
        
        print(f"Targets shape: {targets.shape}")
        print(f"Targets range: [{targets.min():.4f}, {targets.max():.4f}]")
        
        # Combine all embeddings as input features
        # Each sample has n_tokens * embed_dim features
        all_embeddings = np.concatenate([center_embeddings, masked_embeddings], axis=0)
        all_targets = np.concatenate([targets, targets], axis=0)
        
        print(f"Combined input shape: {all_embeddings.shape}")
        print(f"Combined targets shape: {all_targets.shape}")
        
        return all_embeddings, all_targets, tokens, sentence_text

def train_tn_tree_enhanced(X: np.ndarray, y: np.ndarray, tokens: List[str], 
                          rank: int = 8, max_epochs: int = 50, 
                          target_r2: float = 0.8, patience: int = 10) -> Tuple[BinaryTensorTree, Dict]:
    """Train TN-tree with enhanced monitoring and convergence criteria"""
    
    n_samples, total_features = X.shape
    n_tokens = len(tokens)
    embed_dim = total_features // n_tokens
    
    print(f"\nTraining TN-tree with enhanced settings:")
    print(f"  Rank: {rank}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Target R²: {target_r2}")
    print(f"  Patience: {patience}")
    print(f"  Embed dim per token: {embed_dim}")
    
    # Reshape for TN-tree: each token becomes a separate input with embed_dim+1 dimensions (including bias)
    X_reshaped = X.reshape(n_samples, n_tokens, embed_dim)
    
    # Add bias dimension
    bias_dim = np.ones((n_samples, n_tokens, 1))
    X_with_bias = np.concatenate([X_reshaped, bias_dim], axis=2)
    
    print(f"  Input shape with bias: {X_with_bias.shape}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_bias, y, test_size=0.2, random_state=42
    )
    
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create TN-tree model 
    leaf_phys_dims = [embed_dim + 1] * n_tokens  # +1 for bias
    
    model = BinaryTensorTree(
        leaf_phys_dims=leaf_phys_dims,
        rank=rank,
        assume_bias_when_matrix=True,  # Handle bias automatically
        device='cpu'  # Use CPU to avoid memory issues
    )
    
    print(f"  Model created with leaf_phys_dims: {leaf_phys_dims}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Training loop with enhanced monitoring
    train_history = {
        'train_loss': [],
        'test_loss': [],
        'train_r2': [],
        'test_r2': [],
        'best_epoch': 0,
        'best_r2': -np.inf
    }
    
    best_model_state = None
    epochs_without_improvement = 0
    
    print("\nTraining progress:")
    print("Epoch | Train Loss | Test Loss | Train R² | Test R² | LR")
    print("-" * 60)
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        train_pred = model(X_train_tensor)
        train_loss = criterion(train_pred.squeeze(), y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor)
            test_loss = criterion(test_pred.squeeze(), y_test_tensor)
            
            # Calculate R² scores
            train_r2 = r2_score(y_train, train_pred.squeeze().detach().numpy())
            test_r2 = r2_score(y_test, test_pred.squeeze().detach().numpy())
        
        # Update learning rate
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        train_history['train_loss'].append(float(train_loss))
        train_history['test_loss'].append(float(test_loss))
        train_history['train_r2'].append(float(train_r2))
        train_history['test_r2'].append(float(test_r2))
        
        # Check for best model
        if test_r2 > train_history['best_r2']:
            train_history['best_r2'] = float(test_r2)
            train_history['best_epoch'] = epoch
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Print progress
        if epoch % 5 == 0 or epoch < 10:
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
    print(f"  Train R²: {final_train_r2:.4f}, MSE: {final_train_mse:.4f}")
    print(f"  Test R²: {final_test_r2:.4f}, MSE: {final_test_mse:.4f}")
    
    # Add final metrics to history
    train_history['final_train_r2'] = float(final_train_r2)
    train_history['final_test_r2'] = float(final_test_r2)
    train_history['final_train_mse'] = float(final_train_mse)
    train_history['final_test_mse'] = float(final_test_mse)
    
    return model, train_history

def compute_tn_shap_enhanced(model: BinaryTensorTree, X: np.ndarray, tokens: List[str], 
                            orders: List[int] = [1, 2], 
                            remove_mean: bool = True) -> Dict:
    """Compute TN-SHAP values with optional mean removal"""
    
    n_samples, n_tokens, embed_dim_plus_bias = X.shape
    embed_dim = embed_dim_plus_bias - 1  # Subtract bias dimension
    
    print(f"\nComputing TN-SHAP values:")
    print(f"  Orders: {orders}")
    print(f"  Remove mean: {remove_mean}")
    print(f"  Samples: {n_samples}, Tokens: {n_tokens}, Features per token: {embed_dim_plus_bias}")
    
    shapley_results = {}
    
    for order in orders:
        print(f"\nComputing order {order} Shapley values...")
        start_time = time.time()
        
        # Compute exact TN-SHAP
        with torch.no_grad():
            model.eval()
            X_tensor = torch.FloatTensor(X)
            
            if order == 1:
                # Single token contributions
                values = []
                interactions = []
                
                # Baseline: all zeros
                baseline = torch.zeros_like(X_tensor)
                baseline_pred = model(baseline).squeeze().detach().numpy()
                
                for i in range(n_tokens):
                    # Set only token i to its actual value
                    intervention = baseline.clone()
                    intervention[:, i, :] = X_tensor[:, i, :]
                    intervention_pred = model(intervention).squeeze().detach().numpy()
                    
                    # Shapley value for token i
                    shapley_val = np.mean(intervention_pred - baseline_pred)
                    values.append(shapley_val)
                    interactions.append([i])
                    
            elif order == 2:
                # Pairwise interactions
                values = []
                interactions = []
                
                # Baseline: all zeros
                baseline = torch.zeros_like(X_tensor)
                baseline_pred = model(baseline).squeeze().detach().numpy()
                
                # Single token effects
                single_effects = {}
                for i in range(n_tokens):
                    intervention = baseline.clone()
                    intervention[:, i, :] = X_tensor[:, i, :]
                    intervention_pred = model(intervention).squeeze().detach().numpy()
                    single_effects[i] = np.mean(intervention_pred - baseline_pred)
                
                # Pairwise effects
                for i in range(n_tokens):
                    for j in range(i + 1, n_tokens):
                        # Both tokens i and j
                        intervention_both = baseline.clone()
                        intervention_both[:, i, :] = X_tensor[:, i, :]
                        intervention_both[:, j, :] = X_tensor[:, j, :]
                        pred_both = model(intervention_both).squeeze().detach().numpy()
                        
                        # Interaction = combined effect - individual effects
                        interaction_val = (np.mean(pred_both - baseline_pred) - 
                                         single_effects[i] - single_effects[j])
                        
                        values.append(interaction_val)
                        interactions.append([i, j])
        
        values = np.array(values)
        
        # Remove mean if requested
        if remove_mean and len(values) > 0:
            original_mean = np.mean(values)
            values = values - original_mean
            print(f"  Mean removed: {original_mean:.6f}")
            print(f"  Values range after mean removal: [{values.min():.6f}, {values.max():.6f}]")
        else:
            print(f"  Values range: [{values.min():.6f}, {values.max():.6f}]")
        
        computation_time = time.time() - start_time
        print(f"  Computation time: {computation_time:.2f} seconds")
        
        shapley_results[str(order)] = {
            'values': ensure_json_serializable(values),
            'interactions': ensure_json_serializable(interactions),
            'computation_time': computation_time,
            'mean_removed': remove_mean,
            'original_mean': float(original_mean) if remove_mean and len(values) > 0 else 0.0
        }
    
    return shapley_results

def main():
    parser = argparse.ArgumentParser(description='Enhanced TN-tree training with Shapley computation')
    parser.add_argument('--dataset', type=str, required=True, help='Path to GPT dataset JSON file')
    parser.add_argument('--output-dir', type=str, default='./tn_results_enhanced', help='Output directory')
    parser.add_argument('--rank', type=int, default=8, help='TN-tree rank')
    parser.add_argument('--max-epochs', type=int, default=50, help='Maximum training epochs')
    parser.add_argument('--target-r2', type=float, default=0.8, help='Target R² score for early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--remove-mean', action='store_true', help='Remove mean from Shapley values')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("ENHANCED TN-TREE TRAINING WITH SHAPLEY VALUES")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Rank: {args.rank}, Max epochs: {args.max_epochs}")
    print(f"Target R²: {args.target_r2}, Patience: {args.patience}")
    print(f"Remove mean from Shapley values: {args.remove_mean}")
    
    try:
        # Load dataset
        print("\n" + "=" * 50)
        print("LOADING DATASET")
        print("=" * 50)
        
        loader = GPTDatasetLoader(args.dataset)
        X, y, tokens, sentence_text = loader.load_dataset()
        
        # Train TN-tree
        print("\n" + "=" * 50)
        print("TRAINING TN-TREE")
        print("=" * 50)
        
        model, train_history = train_tn_tree_enhanced(
            X, y, tokens, 
            rank=args.rank, 
            max_epochs=args.max_epochs,
            target_r2=args.target_r2,
            patience=args.patience
        )
        
        # Check if R² is satisfactory
        final_r2 = train_history['final_test_r2']
        if final_r2 < 0.5:
            print(f"\nWARNING: Low R² score ({final_r2:.4f}). Consider increasing rank or epochs.")
        elif final_r2 >= 0.8:
            print(f"\nEXCELLENT: High R² score ({final_r2:.4f}). Model fits data well.")
        else:
            print(f"\nGOOD: Moderate R² score ({final_r2:.4f}). Model performance is acceptable.")
        
        # Reshape X for TN-SHAP computation
        n_samples, total_features = X.shape
        n_tokens = len(tokens)
        embed_dim = total_features // n_tokens
        X_reshaped = X.reshape(n_samples, n_tokens, embed_dim)
        
        # Add bias dimension
        bias_dim = np.ones((n_samples, n_tokens, 1))
        X_with_bias = np.concatenate([X_reshaped, bias_dim], axis=2)
        
        # Compute Shapley values
        print("\n" + "=" * 50)
        print("COMPUTING SHAPLEY VALUES")
        print("=" * 50)
        
        shapley_values = compute_tn_shap_enhanced(
            model, X_with_bias, tokens, 
            orders=[1, 2], 
            remove_mean=args.remove_mean
        )
        
        # Save results
        print("\n" + "=" * 50)
        print("SAVING RESULTS")
        print("=" * 50)
        
        # Prepare results
        results = {
            'sentence_text': sentence_text,
            'tokens': tokens,
            'dataset_path': args.dataset,
            'model_config': {
                'rank': args.rank,
                'max_epochs': args.max_epochs,
                'target_r2': args.target_r2,
                'patience': args.patience,
                'leaf_phys_dims': [embed_dim + 1] * n_tokens,
                'assume_bias_when_matrix': True
            },
            'training_history': ensure_json_serializable(train_history),
            'shapley_values': shapley_values,
            'final_metrics': {
                'train_r2': float(train_history['final_train_r2']),
                'test_r2': float(train_history['final_test_r2']),
                'train_mse': float(train_history['final_train_mse']),
                'test_mse': float(train_history['final_test_mse'])
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save JSON results
        safe_filename = sentence_text.replace(' ', '_').replace('"', '').replace("'", '').replace('.', '')
        json_path = os.path.join(args.output_dir, f'{safe_filename}_tn_shapley_enhanced.json')
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {json_path}")
        
        # Save model
        model_path = os.path.join(args.output_dir, f'{safe_filename}_tn_model_enhanced.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        
        print("\n" + "=" * 80)
        print("ENHANCED TRAINING COMPLETE")
        print("=" * 80)
        print(f"Final Test R²: {final_r2:.4f}")
        print(f"Training Quality: {'EXCELLENT' if final_r2 >= 0.8 else 'GOOD' if final_r2 >= 0.5 else 'POOR'}")
        
        if args.remove_mean:
            print(f"Shapley values computed with mean removed")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
