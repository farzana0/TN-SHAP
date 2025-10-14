#!/usr/bin/env python3
"""
Enhanced TN-tree training with heavy mask injection for better Shapley computation.
Focuses on single token masks and pairwise masks to improve TN-SHAP quality.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.append('./higher_order_iclr')
from tntree_model import BinaryTensorTree

def inject_mask_patterns(X, y, tokens, mask_probability=0.5, pairwise_multiplier=5):
    """
    Inject various masking patterns into the training data with HEAVY pairwise injection.
    
    Args:
        X: Input embeddings [n_samples, n_tokens, embed_dim]
        y: Target values [n_samples]
        tokens: List of token strings
        mask_probability: Base probability of applying masking
        pairwise_multiplier: How many times more pairwise masks to inject
    
    Returns:
        X_augmented: Augmented input data
        y_augmented: Corresponding target values
        mask_info: Information about applied masks
    """
    n_samples, n_tokens, embed_dim = X.shape
    
    X_augmented = [X.copy()]  # Start with original data
    y_augmented = [y.copy()]
    mask_info = [{'type': 'original', 'masked_tokens': []} for _ in range(n_samples)]
    
    print(f"Starting HEAVY mask injection with {mask_probability} base probability...")
    print(f"Pairwise multiplier: {pairwise_multiplier}x (HEAVY pairwise injection)")
    
    # 1. Single token masks (crucial for order-1 Shapley)
    single_injections = 0
    for token_idx in range(n_tokens):
        if np.random.random() > mask_probability:
            continue
            
        print(f"  Injecting single token masks for token {token_idx} ({tokens[token_idx]})")
        
        X_masked = X.copy()
        # Mask the specific token (set embeddings to 0, keep bias if present)
        X_masked[:, token_idx, :-1] = 0  # Assuming last dim is bias
        
        X_augmented.append(X_masked)
        y_augmented.append(y.copy())
        mask_info.extend([{'type': 'single', 'masked_tokens': [token_idx]} for _ in range(n_samples)])
        single_injections += 1
    
    # 2. HEAVY Pairwise token masks (crucial for order-2 Shapley)
    print(f"  HEAVY PAIRWISE INJECTION (multiplier={pairwise_multiplier}x):")
    pairwise_injections = 0
    
    # Generate ALL possible pairs (not limited)
    all_pairs = [(i, j) for i in range(n_tokens) for j in range(i + 1, n_tokens)]
    print(f"    Total possible pairs: {len(all_pairs)}")
    
    # Inject each pair multiple times with high probability
    for pair_round in range(pairwise_multiplier):
        print(f"    Pairwise injection round {pair_round + 1}/{pairwise_multiplier}")
        
        for i, j in all_pairs:
            # Higher probability for pairwise masks
            pairwise_prob = min(0.9, mask_probability * 1.5)  # Boost pairwise probability
            
            if np.random.random() > pairwise_prob:
                continue
                
            print(f"      Round {pair_round+1}: Injecting pair ({i},{j}) - {tokens[i]} x {tokens[j]}")
            
            X_masked = X.copy()
            # Mask both tokens
            X_masked[:, i, :-1] = 0
            X_masked[:, j, :-1] = 0
            
            X_augmented.append(X_masked)
            y_augmented.append(y.copy())
            mask_info.extend([{'type': 'pair', 'masked_tokens': [i, j], 'round': pair_round+1} for _ in range(n_samples)])
            pairwise_injections += 1
    
    # 3. Triple token masks (for higher-order interactions)
    triple_injections = 0
    if n_tokens >= 3:
        print(f"  Injecting triple token masks for higher-order interactions:")
        n_triple_samples = min(10, n_tokens)  # Sample some triples
        
        for _ in range(n_triple_samples):
            if np.random.random() > mask_probability * 0.7:  # Slightly lower prob for triples
                continue
                
            # Randomly select 3 tokens
            triple_tokens = np.random.choice(n_tokens, 3, replace=False).tolist()
            print(f"    Injecting triple mask for tokens {triple_tokens}")
            
            X_masked = X.copy()
            for token_idx in triple_tokens:
                X_masked[:, token_idx, :-1] = 0
                
            X_augmented.append(X_masked)
            y_augmented.append(y.copy())
            mask_info.extend([{'type': 'triple', 'masked_tokens': triple_tokens} for _ in range(n_samples)])
            triple_injections += 1
    
    # 4. Reduced random masks (less emphasis on random patterns)
    random_injections = 0
    n_random_masks = min(3, n_tokens)  # Fewer random masks
    for _ in range(n_random_masks):
        if np.random.random() > mask_probability * 0.5:  # Lower prob for random
            continue
            
        # Randomly select 1-3 tokens to mask
        n_mask = np.random.randint(1, min(4, n_tokens + 1))
        masked_tokens = np.random.choice(n_tokens, n_mask, replace=False).tolist()
        
        print(f"  Injecting random mask for tokens {masked_tokens}")
        
        X_masked = X.copy()
        for token_idx in masked_tokens:
            X_masked[:, token_idx, :-1] = 0
            
        X_augmented.append(X_masked)
        y_augmented.append(y.copy())
        mask_info.extend([{'type': 'random', 'masked_tokens': masked_tokens} for _ in range(n_samples)])
        random_injections += 1
    
    # Combine all augmented data
    X_final = np.concatenate(X_augmented, axis=0)
    y_final = np.concatenate(y_augmented, axis=0)
    
    print(f"HEAVY mask injection complete:")
    print(f"  Original samples: {n_samples}")
    print(f"  Single token injections: {single_injections}")
    print(f"  PAIRWISE injections: {pairwise_injections} (HEAVY!)")
    print(f"  Triple token injections: {triple_injections}")
    print(f"  Random injections: {random_injections}")
    print(f"  Total augmented samples: {X_final.shape[0]}")
    print(f"  Augmentation ratio: {X_final.shape[0] / n_samples:.1f}x")
    print(f"  Pairwise ratio: {pairwise_injections / single_injections:.1f}x more pairwise than single" if single_injections > 0 else "")
    
    return X_final, y_final, mask_info

def load_dataset(dataset_path):
    """Load dataset from JSON file"""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loading dataset: {dataset['sentence']}")
    print(f"Tokens: {dataset['token_info']['token_strings']}")
    
    X_list = []
    y_list = []
    
    for order, order_data in dataset['data'].items():
        samples = order_data['samples']
        print(f"Processing order {order}: {len(samples)} samples")
        
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
    
    X = np.stack(X_list)  # Shape: [n_samples, n_tokens, 16]
    y = np.array(y_list)
    tokens = dataset['token_info']['token_strings']
    
    return X, y, tokens, dataset

def train_tn_tree(X, y, tokens, config):
    """Train TN-tree model with enhanced monitoring"""
    n_samples, n_tokens, embed_dim = X.shape
    
    # Add bias dimension
    bias_dim = np.ones((n_samples, n_tokens, 1))
    X_with_bias = np.concatenate([X, bias_dim], axis=2)
    input_dim = embed_dim + 1
    
    # Create model architecture
    leaf_phys_dims = [input_dim] * n_tokens  # Each token is a leaf
    model = BinaryTensorTree(
        leaf_phys_dims=leaf_phys_dims,
        leaf_input_dims=leaf_phys_dims,
        ranks=config['rank'],
        assume_bias_when_matrix=True,
        device='cpu'
    )
    
    # Convert to tensors and flatten
    X_flat = X_with_bias.reshape(n_samples, -1)
    X_tensor = torch.FloatTensor(X_flat)
    y_tensor = torch.FloatTensor(y)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    best_r2 = -float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    
    print(f"\nStarting training for {config['max_epochs']} epochs...")
    
    for epoch in range(config['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Mini-batch training
        batch_size = min(config['batch_size'], X_train.shape[0])
        n_batches = (X_train.shape[0] + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, X_train.shape[0])
            
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= n_batches
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            y_train_pred = model(X_train).squeeze()
            y_val_pred = model(X_val).squeeze()
            
            val_loss = criterion(y_val_pred, y_val).item()
            
            # Compute R² scores
            train_r2 = r2_score(y_train.numpy(), y_train_pred.numpy())
            val_r2 = r2_score(y_val.numpy(), y_val_pred.numpy())
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Progress reporting
        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
        
        # Early stopping based on R²
        if val_r2 > best_r2:
            best_r2 = val_r2
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch} (best val R²: {best_r2:.4f})")
            break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train).squeeze()
        y_val_pred = model(X_val).squeeze()
        
        final_train_r2 = r2_score(y_train.numpy(), y_train_pred.numpy())
        final_val_r2 = r2_score(y_val.numpy(), y_val_pred.numpy())
        
        final_train_mse = mean_squared_error(y_train.numpy(), y_train_pred.numpy())
        final_val_mse = mean_squared_error(y_val.numpy(), y_val_pred.numpy())
    
    print(f"\nFinal Results:")
    print(f"  Train R²: {final_train_r2:.4f}, MSE: {final_train_mse:.4f}")
    print(f"  Val R²: {final_val_r2:.4f}, MSE: {final_val_mse:.4f}")
    
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_r2s': train_r2s,
        'val_r2s': val_r2s
    }
    
    final_metrics = {
        'train_r2': final_train_r2,
        'val_r2': final_val_r2,
        'train_mse': final_train_mse,
        'val_mse': final_val_mse,
        'best_val_r2': best_r2,
        'epochs_trained': epoch + 1
    }
    
    return model, training_history, final_metrics, leaf_phys_dims, input_dim

def save_results(model, training_history, final_metrics, dataset_info, tokens, config, leaf_phys_dims, input_dim, output_dir):
    """Save training results and model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate safe filename
    sentence = dataset_info.get('sentence', 'unknown')
    safe_filename = sentence.replace(' ', '_').replace('"', '').replace("'", '').replace('.', '').replace(',', '_')
    
    # Save model
    model_path = os.path.join(output_dir, f'{safe_filename}_tn_model_masked.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save results JSON
    results = {
        'sentence_text': sentence,
        'tokens': tokens,
        'dataset_path': config['dataset'],
        'model_config': {
            'leaf_phys_dims': leaf_phys_dims,
            'rank': config['rank'],
            'input_dim': input_dim
        },
        'training_config': {
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'max_epochs': config['max_epochs'],
            'patience': config['patience'],
            'mask_probability': config['mask_probability']
        },
        'final_metrics': final_metrics,
        'training_history': training_history,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    json_path = os.path.join(output_dir, f'{safe_filename}_tn_masked_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training curves plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_history['train_losses'], label='Train Loss')
    plt.plot(training_history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(training_history['train_r2s'], label='Train R²')
    plt.plot(training_history['val_r2s'], label='Val R²')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.title('R² Score Progress')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{safe_filename}_training_curves_masked.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved:")
    print(f"  Model: {model_path}")
    print(f"  Results: {json_path}")
    print(f"  Training curves: {plot_path}")
    
    return json_path

def main():
    parser = argparse.ArgumentParser(description='Train TN-tree with heavy mask injection')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset JSON')
    parser.add_argument('--output-dir', type=str, default='./tn_results_masked', help='Output directory')
    parser.add_argument('--rank', type=int, default=8, help='TN-tree rank')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--mask-probability', type=float, default=0.7, help='Mask injection probability')
    parser.add_argument('--pairwise-multiplier', type=int, default=5, help='Pairwise mask injection multiplier (higher = more pairwise masks)')
    
    args = parser.parse_args()
    
    config = {
        'dataset': args.dataset,
        'rank': args.rank,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'mask_probability': args.mask_probability,
        'pairwise_multiplier': args.pairwise_multiplier
    }
    
    print("=" * 80)
    print("TN-TREE TRAINING WITH MASK INJECTION")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Rank: {args.rank}")
    print(f"Mask probability: {args.mask_probability}")
    
    # Load dataset
    X, y, tokens, dataset_info = load_dataset(args.dataset)
    print(f"\nOriginal dataset shape: {X.shape}")
    
    # Inject mask patterns
    X_augmented, y_augmented, mask_info = inject_mask_patterns(X, y, tokens, args.mask_probability, args.pairwise_multiplier)
    print(f"Augmented dataset shape: {X_augmented.shape}")
    
    # Train model
    print(f"\nTraining TN-tree model...")
    model, training_history, final_metrics, leaf_phys_dims, input_dim = train_tn_tree(X_augmented, y_augmented, tokens, config)
    
    # Save results
    result_path = save_results(model, training_history, final_metrics, dataset_info, tokens, config, leaf_phys_dims, input_dim, args.output_dir)
    
    print(f"\nTraining complete! Final validation R²: {final_metrics['val_r2']:.4f}")
    
    return result_path

if __name__ == "__main__":
    main()
