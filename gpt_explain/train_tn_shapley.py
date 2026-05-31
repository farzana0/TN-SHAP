#!/usr/bin/env python3
"""
Train TN-tree models on GPT synthetic datasets and compute TN-SHAP values.
Only TN model results - no Kernel    # Data comes as [B, n_tokens, embed_dim] - reshape to [B, n_tokens * embed_dim] for TN-tree
    B_train, n_tokens, embed_dim = X_train.shape
    B_val = X_val.shape[0]
    
    # Flatten token embeddings: [B, n_tokens, embed_dim] -> [B, n_tokens * embed_dim]
    X_train_flat = X_train.view(B_train, -1)  # [B, n_tokens * embed_dim]
    X_val_flat = X_val.view(B_val, -1)       # [B, n_tokens * embed_dim]
    
    total_input_dim = n_tokens * embed_dim
    print(f"Total input dimension: {total_input_dim} = {n_tokens} tokens × {embed_dim} dims")
    
    # Create TN-tree model with bias handling per token
    # Each token gets embed_dim+1 dimensions (embed_dim + 1 bias)
    model = BinaryTensorTree(
        leaf_phys_dims=[embed_dim + 1] * n_tokens,  # Each token leaf has embed_dim+1 physical dims
        leaf_input_dims=[embed_dim] * n_tokens,     # Each token leaf has embed_dim input features
        ranks=rank,
        out_dim=1,
        assume_bias_when_matrix=True,  # Automatically add bias dimension
        seed=seed,
        device=DEVICE,
        dtype=torch.float32
    ) or Exact methods.
Computes Shapley values for orders 1 and 2 using saved masked embeddings.

Created: September 25, 2025
"""

import os
import json
import time
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import our utilities
from embedding_loader_utils import load_center_embeddings, load_masked_data
from tntree_model import BinaryTensorTree

def ensure_json_serializable(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj

def ensure_dir(p: str): 
    os.makedirs(p, exist_ok=True)

def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

DEVICE = torch.device("cpu")  # Force CPU to avoid CUDA memory issues

def cosine_sim(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 and nb < 1e-12:
        return 1.0
    elif na < 1e-12 or nb < 1e-12:
        return 0.0
    else:
        return float(np.dot(a, b) / (na * nb))

class GPTDatasetLoader:
    """Load and prepare GPT synthetic datasets for TN-tree training"""
    
    def __init__(self, dataset_path: str, seed: int = 42):
        self.dataset_path = dataset_path
        self.seed = seed
        self.dataset_name = self._extract_dataset_name(dataset_path)
        
    def _extract_dataset_name(self, path: str) -> str:
        """Extract sentence identifier from dataset path"""
        filename = os.path.basename(path)
        if "sentence_1" in filename:
            return "The_food_was_cheap_fresh_and_tasty"
        elif "sentence_2" in filename:
            return "The_test_was_easy_and_simple"
        elif "sentence_3" in filename:
            return "The_product_is_not_very_reliable"
        elif "sentence_4" in filename:
            return "Great_just_what_I_needed"
        else:
            raise ValueError(f"Cannot determine sentence from {filename}")
    
    def load_dataset(self):
        """Load GPT synthetic dataset"""
        print(f"Loading dataset: {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        # Extract data based on the actual dataset structure
        if 'data' in data:
            # Collect all samples from all k-subsets
            X_list = []
            y_list = []
            
            for k_str, k_data in data['data'].items():
                k = int(k_str)
                for sample in k_data['samples']:
                    # Each sample has embeddings (shape: [n_noise_samples, n_tokens, embed_dim])
                    embeddings = np.array(sample['embeddings'])
                    n_noise_samples = embeddings.shape[0]
                    
                    # Keep the token structure - each sample is [n_tokens, embed_dim]
                    for i in range(n_noise_samples):
                        X_list.append(embeddings[i])  # Shape: [n_tokens, embed_dim]
                        y_list.append(sample['target_logit'])
            
            # Stack all samples
            X = np.stack(X_list)  # Shape: [n_samples, n_tokens, embed_dim]
            y = np.array(y_list)  # Shape: [n_samples]
            
            print(f"Loaded {X.shape[0]} samples with {X.shape[1]} tokens and {X.shape[2]} embedding dimensions")
            print(f"Target shape: {y.shape}")
            print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
            
        else:
            raise ValueError(f"Unrecognized dataset structure in {self.dataset_path}")
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Note: No standardization for 3D data - we'll handle this in the model
        
        return {
            'X_train': X_train.astype(np.float32),
            'X_val': X_val.astype(np.float32), 
            'y_train': y_train.astype(np.float32),
            'y_val': y_val.astype(np.float32),
            'n_tokens': X.shape[1],
            'embed_dim': X.shape[2]
        }

def train_tn_tree(data_dict: Dict, rank: int = 16, seed: int = 42, max_epochs: int = 2000):
    """Train TN-tree model on the dataset"""
    print(f"Training TN-tree with rank={rank}, seed={seed}")
    
    set_all_seeds(seed)
    
    # Get dimensions
    n_tokens = data_dict['n_tokens']
    embed_dim = data_dict['embed_dim'] 
    
    print(f"Data shape: n_tokens={n_tokens}, embed_dim={embed_dim}")
    
    # Prepare data - keep original embeddings without bias for model input
    X_train = torch.from_numpy(data_dict['X_train']).to(DEVICE)  # [batch, n_tokens, embed_dim]
    X_val = torch.from_numpy(data_dict['X_val']).to(DEVICE)
    y_train = torch.from_numpy(data_dict['y_train']).to(DEVICE)
    y_val = torch.from_numpy(data_dict['y_val']).to(DEVICE)
    
    # Flatten the input for the model: [batch, n_tokens, embed_dim] -> [batch, n_tokens * embed_dim]
    print("Flattening input data for model...")
    X_train_flat = X_train.view(X_train.shape[0], -1)  # [batch, n_tokens * embed_dim]
    X_val_flat = X_val.view(X_val.shape[0], -1)
    
    print(f"Flattened input shape: {X_train_flat.shape}")
    
    # Create TN-tree model - each token gets (embed_dim+1) physical dimensions but embed_dim input dims
    model = BinaryTensorTree(
        leaf_phys_dims=[embed_dim + 1] * n_tokens,  # Physical dims: each token has (embed_dim+1) dimensions
        leaf_input_dims=[embed_dim] * n_tokens,     # Input dims: each token has embed_dim input features
        ranks=rank,
        out_dim=1,
        assume_bias_when_matrix=True,  # Enable bias handling
        seed=seed,
        device=DEVICE,
        dtype=torch.float32
    )
    
    print(f"Model created: {model}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=50, verbose=True
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 100
    
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in tqdm(range(max_epochs), desc="Training"):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        y_pred_train = model(X_train_flat)
        train_loss = criterion(y_pred_train, y_train)
        
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val_flat)
            val_loss = criterion(y_pred_val, y_val)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 100 == 0:
            # Calculate R² scores for progress monitoring
            def r2_temp(y_true, y_pred):
                ss_res = torch.sum((y_true - y_pred) ** 2)
                ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
                return 1 - (ss_res / ss_tot)
            
            with torch.no_grad():
                train_r2_temp = r2_temp(y_train, y_pred_train)
                val_r2_temp = r2_temp(y_val, y_pred_val)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, train_R²={train_r2_temp:.4f}, val_R²={val_r2_temp:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train_flat)
        y_pred_val = model(X_val_flat)
        
        train_mse = torch.mean((y_pred_train - y_train) ** 2).item()
        val_mse = torch.mean((y_pred_val - y_val) ** 2).item()
        
        # Calculate R² scores
        def r2_score(y_true, y_pred):
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            ss_res = np.sum((y_true_np - y_pred_np) ** 2)
            ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        
        print(f"Final Train MSE: {train_mse:.6f}, R²: {train_r2:.6f}")
        print(f"Final Val MSE: {val_mse:.6f}, R²: {val_r2:.6f}")
        
        final_train_loss = criterion(y_pred_train, y_train).item()
        final_val_loss = criterion(y_pred_val, y_val).item()
    
    training_info = {
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_epochs': len(train_losses)
    }
    
    print(f"Training completed!")
    print(f"Final losses - Train: {final_train_loss:.6f}, Val: {final_val_loss:.6f}")
    
    return model, training_info

def compute_tn_shapley_values(model, sentence_prefix: str, baseline: str = "zero", orders: List[int] = [1, 2]):
    """
    Compute TN-SHAP values using saved masked embeddings.
    Only orders 1 and 2 as requested.
    """
    print(f"Computing TN-SHAP for {sentence_prefix}, baseline={baseline}, orders={orders}")
    
    # Load center embeddings and masked data
    center_data = load_center_embeddings(sentence_prefix, data_dir="gpt_explain")
    masked_data = load_masked_data(sentence_prefix, baseline, data_dir="gpt_explain")
    
    center_embeddings = center_data['embeddings']  # [num_tokens, 768]
    masked_embeddings = masked_data['masked_embeddings']  # [num_subsets, num_tokens, 768]
    subsets = masked_data['subsets']
    
    num_tokens = center_embeddings.shape[0]
    embedding_dim = center_embeddings.shape[1]
    
    print(f"  Tokens: {num_tokens}, Embedding dim: {embedding_dim}")
    print(f"  Number of subsets: {len(subsets)}")
    
    # Use embeddings without adding bias (model handles bias automatically)
    center_embeddings = center_embeddings.to(DEVICE)
    masked_embeddings = masked_embeddings.to(DEVICE)
    
    model.eval()
    
    # Compute center point prediction (original sentence)
    with torch.no_grad():
        # Flatten: [num_tokens, embed_dim] -> [num_tokens * embed_dim]
        center_flat = center_embeddings.view(-1).unsqueeze(0)  # [1, num_tokens * embed_dim]
        center_pred = model(center_flat).item()
    
    print(f"  Center point prediction: {center_pred:.6f}")
    
    # Compute predictions for all masked versions
    with torch.no_grad():
        masked_predictions = []
        for i in tqdm(range(len(subsets)), desc="Computing masked predictions"):
            # Flatten: [num_tokens, embed_dim] -> [num_tokens * embed_dim]
            masked_flat = masked_embeddings[i].view(-1).unsqueeze(0)  # [1, num_tokens * embed_dim]
            masked_pred = model(masked_flat).item()
            masked_predictions.append(masked_pred)
    
    masked_predictions = np.array(masked_predictions)
    print(f"  Masked predictions range: [{masked_predictions.min():.6f}, {masked_predictions.max():.6f}]")
    
    # Compute Shapley values for each order
    shapley_results = {}
    
    for order in orders:
        print(f"  Computing Shapley values for order {order}")
        
        if order == 1:
            # Single token Shapley values
            single_shapley = np.zeros(num_tokens)
            
            for token_idx in range(num_tokens):
                # Find subset that masks only this token
                target_subset = [token_idx]
                
                # Find the index in our subsets list
                subset_idx = None
                for i, subset in enumerate(subsets):
                    if list(subset) == target_subset:
                        subset_idx = i
                        break
                
                if subset_idx is not None:
                    # Shapley value = center_pred - masked_pred
                    single_shapley[token_idx] = center_pred - masked_predictions[subset_idx]
                else:
                    print(f"    Warning: Could not find subset {target_subset}")
            
            shapley_results[f'order_{order}'] = {
                'values': single_shapley.tolist(),
                'indices': list(range(num_tokens)),
                'description': 'Single token Shapley values'
            }
            
        elif order == 2:
            # Pairwise Shapley values
            pairwise_shapley = {}
            
            for i in range(num_tokens):
                for j in range(i + 1, num_tokens):
                    target_subset = sorted([i, j])
                    
                    # Find the index in our subsets list
                    subset_idx = None
                    for k, subset in enumerate(subsets):
                        if list(sorted(subset)) == target_subset:
                            subset_idx = k
                            break
                    
                    if subset_idx is not None:
                        # Interaction Shapley value = center_pred - masked_pred
                        pair_shapley = center_pred - masked_predictions[subset_idx]
                        pairwise_shapley[f'{i}_{j}'] = pair_shapley
                    else:
                        print(f"    Warning: Could not find subset {target_subset}")
            
            shapley_results[f'order_{order}'] = {
                'values': pairwise_shapley,
                'description': 'Pairwise interaction Shapley values'
            }
    
    # Add metadata
    shapley_results['metadata'] = {
        'sentence': center_data['sentence'],
        'tokens': center_data['tokens'].tolist() if hasattr(center_data['tokens'], 'tolist') else list(center_data['tokens']),
        'num_tokens': int(num_tokens),
        'embedding_dim': int(embedding_dim),
        'baseline': baseline,
        'center_prediction': float(center_pred),
        'model_type': 'TN-tree',
        'orders_computed': list(orders)
    }
    
    return shapley_results

def main():
    parser = argparse.ArgumentParser(description='Train TN-tree on GPT datasets and compute Shapley values')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Path to dataset JSON file')
    parser.add_argument('--rank', type=int, default=32, 
                       help='TN-tree rank')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--max-epochs', type=int, default=2000, 
                       help='Maximum training epochs')
    parser.add_argument('--output-dir', type=str, default='./tn_results', 
                       help='Output directory')
    parser.add_argument('--baseline', type=str, default='zero', choices=['zero', 'mean'],
                       help='Baseline for masking')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TN-TREE TRAINING AND SHAPLEY VALUE COMPUTATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Rank: {args.rank}")
    print(f"Seed: {args.seed}")
    print(f"Baseline: {args.baseline}")
    print(f"Device: {DEVICE}")
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Load dataset
    loader = GPTDatasetLoader(args.dataset, args.seed)
    data_dict = loader.load_dataset()
    
    # Train TN-tree model
    print("\n" + "="*50)
    print("TRAINING TN-TREE MODEL")
    print("="*50)
    
    start_time = time.time()
    model, training_info = train_tn_tree(data_dict, args.rank, args.seed, args.max_epochs)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    model_path = os.path.join(args.output_dir, f'{loader.dataset_name}_tn_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': data_dict['X_train'].shape[1],
            'ranks': args.rank,
            'out_dim': 1,
            'bias': True,
            'seed': args.seed
        },
        'training_info': training_info,
        'training_time': training_time
    }, model_path)
    
    print(f"Model saved: {model_path}")
    
    # Compute Shapley values
    print("\n" + "="*50)
    print("COMPUTING TN-SHAPLEY VALUES")
    print("="*50)
    
    start_time = time.time()
    shapley_results = compute_tn_shapley_values(
        model, loader.dataset_name, args.baseline, orders=[1, 2]
    )
    shapley_time = time.time() - start_time
    
    print(f"Shapley computation completed in {shapley_time:.2f} seconds")
    
    # Save results
    results_path = os.path.join(args.output_dir, f'{loader.dataset_name}_tn_shapley_{args.baseline}.json')
    
    # Add timing info
    shapley_results['timing'] = {
        'training_time': training_time,
        'shapley_time': shapley_time,
        'total_time': training_time + shapley_time
    }
    
    with open(results_path, 'w') as f:
        json.dump(ensure_json_serializable(shapley_results), f, indent=2)
    
    print(f"Results saved: {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Dataset: {loader.dataset_name}")
    print(f"Final train loss: {training_info['final_train_loss']:.6f}")
    print(f"Final val loss: {training_info['final_val_loss']:.6f}")
    print(f"Training time: {training_time:.2f}s")
    print(f"Shapley computation time: {shapley_time:.2f}s")
    
    if 'order_1' in shapley_results:
        single_values = np.array(shapley_results['order_1']['values'])
        print(f"Single token Shapley values:")
        print(f"  Mean: {single_values.mean():.6f}")
        print(f"  Std: {single_values.std():.6f}")
        print(f"  Range: [{single_values.min():.6f}, {single_values.max():.6f}]")
    
    if 'order_2' in shapley_results:
        pair_values = list(shapley_results['order_2']['values'].values())
        pair_values = np.array(pair_values)
        print(f"Pairwise Shapley values:")
        print(f"  Count: {len(pair_values)}")
        print(f"  Mean: {pair_values.mean():.6f}")
        print(f"  Std: {pair_values.std():.6f}")
        print(f"  Range: [{pair_values.min():.6f}, {pair_values.max():.6f}]")
    
    print("\nTN-tree training and Shapley computation completed successfully!")

if __name__ == "__main__":
    main()
