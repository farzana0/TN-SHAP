#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Comprehensive TN-Shapley Example: Synthetic Data with Strong Pairwise Interactions

This example demonstrates:
1. Synthetic data generation with d=100 and strong pairwise interactions
2. Ground truth GAM-based Shapley value calculations
3. Tensor Network fitting with R² > 0.9
4. Mask augmentation for training and interpolation
5. Comparison with ground truth
6. SHAPIQ baseline evaluation

Based on groundtruth_multilinear_generator.py but extended for comprehensive TN analysis.

Created: September 28, 2025
"""

import os
import json
import time
import math
import argparse
import warnings
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add paths for TN imports
import sys

try:
    from tensornetwork import TN, set_seed
    from greedy_adaptive_tn import fit_tensor_network_fixed_structure
    TN_AVAILABLE = True
    print("✅ TensorNetwork modules imported successfully")
except ImportError as e:
    TN_AVAILABLE = False
    print(f"❌ TensorNetwork import failed: {e}")

try:
    from src.tntree_model import BinaryTensorTree
    TNTREE_AVAILABLE = True
    print("✅ BinaryTensorTree imported successfully")
except ImportError as e:
    TNTREE_AVAILABLE = False
    print(f"❌ BinaryTensorTree import failed: {e}")

warnings.filterwarnings('ignore')

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-12))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return np.dot(a, b) / (na * nb)

# ===============================================================================
# SYNTHETIC DATA GENERATOR WITH STRONG PAIRWISE INTERACTIONS
# ===============================================================================

class SyntheticDataGenerator:
    """Generate synthetic data with controllable pairwise interactions."""
    
    def __init__(self, d: int, n_strong_pairs: int = 20, n_weak_pairs: int = 30, 
                 noise_std: float = 0.1, seed: int = 42):
        """
        Args:
            d: Number of features
            n_strong_pairs: Number of strong pairwise interactions
            n_weak_pairs: Number of weak pairwise interactions  
            noise_std: Standard deviation of noise
            seed: Random seed
        """
        self.d = d
        self.n_strong_pairs = n_strong_pairs
        self.n_weak_pairs = n_weak_pairs
        self.noise_std = noise_std
        self.seed = seed
        
        set_all_seeds(seed)
        self._generate_ground_truth()
    
    def _generate_ground_truth(self):
        """Generate ground truth coefficients for pairwise interactions."""
        self.coefficients = {}
        
        # Sample pairs for strong interactions
        all_pairs = [(i, j) for i in range(self.d) for j in range(i+1, self.d)]
        np.random.shuffle(all_pairs)
        
        # Strong pairwise interactions
        strong_pairs = all_pairs[:self.n_strong_pairs]
        for i, j in strong_pairs:
            self.coefficients[(i, j)] = np.random.normal(0, 2.0)  # Strong coefficients
        
        # Weak pairwise interactions
        weak_pairs = all_pairs[self.n_strong_pairs:self.n_strong_pairs + self.n_weak_pairs]
        for i, j in weak_pairs:
            self.coefficients[(i, j)] = np.random.normal(0, 0.5)  # Weak coefficients
        
        # Linear terms (moderate strength)
        self.linear_coeffs = np.random.normal(0, 1.0, self.d)
        
        # Store interaction pairs for analysis
        self.strong_pairs = strong_pairs
        self.weak_pairs = weak_pairs
        
        print(f"Ground truth generated:")
        print(f"  Strong pairs: {len(strong_pairs)} (coeff std: 2.0)")
        print(f"  Weak pairs: {len(weak_pairs)} (coeff std: 0.5)")
        print(f"  Linear terms std: 1.0")
    
    def generate_function(self, X: np.ndarray) -> np.ndarray:
        """Generate target values based on ground truth function."""
        n_samples = X.shape[0]
        y = np.zeros(n_samples)
        
        # Linear terms
        y += X @ self.linear_coeffs
        
        # Pairwise interactions
        for (i, j), coeff in self.coefficients.items():
            y += coeff * X[:, i] * X[:, j]
        
        # Add noise
        y += np.random.normal(0, self.noise_std, n_samples)
        
        return y
    
    def generate_dataset(self, n_samples: int, x_dist: str = 'normal') -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic dataset."""
        if x_dist == 'normal':
            X = np.random.normal(0, 1, (n_samples, self.d))
        elif x_dist == 'uniform':
            X = np.random.uniform(-1, 1, (n_samples, self.d))
        elif x_dist == 'binary':
            X = np.random.binomial(1, 0.5, (n_samples, self.d)).astype(float)
        else:
            raise ValueError(f"Unknown distribution: {x_dist}")
        
        y = self.generate_function(X)
        return X, y
    
    def compute_ground_truth_shapley(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ground truth Shapley values analytically."""
        x = np.asarray(x)
        
        # First-order Shapley values (linear terms + half of interaction terms)
        shapley_k1 = self.linear_coeffs.copy()
        
        # Add contributions from pairwise interactions
        for (i, j), coeff in self.coefficients.items():
            shapley_k1[i] += 0.5 * coeff * x[j]  # Half the interaction to feature i
            shapley_k1[j] += 0.5 * coeff * x[i]  # Half the interaction to feature j
        
        # Second-order Shapley values (pure interaction terms)
        shapley_k2 = []
        pair_indices = []
        for i in range(self.d):
            for j in range(i+1, self.d):
                if (i, j) in self.coefficients:
                    shapley_k2.append(self.coefficients[(i, j)] * x[i] * x[j])
                else:
                    shapley_k2.append(0.0)
                pair_indices.append((i, j))
        
        return np.array(shapley_k1), np.array(shapley_k2)

# ===============================================================================
# GAM-BASED SHAPLEY COMPUTATION
# ===============================================================================

class GAMShapleyComputer:
    """Compute Shapley values using Generalized Additive Model approach."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, model_func: callable):
        """
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target vector [n_samples]
            model_func: Function that takes X and returns predictions
        """
        self.X = X
        self.y = y
        self.model_func = model_func
        self.n_samples, self.d = X.shape
        self.baseline = np.zeros(self.d)  # Zero baseline
    
    def compute_marginal_contributions(self, x_point: np.ndarray) -> np.ndarray:
        """Compute marginal contributions for each feature."""
        x_point = np.asarray(x_point)
        baseline_pred = self.model_func(self.baseline.reshape(1, -1))[0]
        
        marginal_contribs = []
        for i in range(self.d):
            x_with_i = self.baseline.copy()
            x_with_i[i] = x_point[i]
            pred_with_i = self.model_func(x_with_i.reshape(1, -1))[0]
            marginal_contribs.append(pred_with_i - baseline_pred)
        
        return np.array(marginal_contribs)
    
    def compute_pairwise_interactions(self, x_point: np.ndarray) -> np.ndarray:
        """Compute pairwise interaction Shapley values."""
        x_point = np.asarray(x_point)
        baseline_pred = self.model_func(self.baseline.reshape(1, -1))[0]
        
        interactions = []
        for i in range(self.d):
            for j in range(i+1, self.d):
                # Individual contributions
                x_i = self.baseline.copy()
                x_i[i] = x_point[i]
                pred_i = self.model_func(x_i.reshape(1, -1))[0]
                
                x_j = self.baseline.copy()
                x_j[j] = x_point[j]
                pred_j = self.model_func(x_j.reshape(1, -1))[0]
                
                # Joint contribution
                x_ij = self.baseline.copy()
                x_ij[i] = x_point[i]
                x_ij[j] = x_point[j]
                pred_ij = self.model_func(x_ij.reshape(1, -1))[0]
                
                # Interaction = joint - individual contributions
                interaction = (pred_ij - baseline_pred) - (pred_i - baseline_pred) - (pred_j - baseline_pred)
                interactions.append(interaction)
        
        return np.array(interactions)

# ===============================================================================
# TENSOR NETWORK FITTING
# ===============================================================================

class TensorNetworkFitter:
    """Fit tensor networks to synthetic data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        self.X = X
        self.y = y
        self.seed = seed
        self.n_samples, self.d = X.shape
        
        set_all_seeds(seed)
    
    def create_tn_structure(self, connectivity_type: str = 'sparse') -> nx.Graph:
        """Create tensor network structure."""
        structure = nx.Graph()
        structure.add_nodes_from(range(self.d))
        
        if connectivity_type == 'sparse':
            # Create sparse structure with reasonable connectivity
            for i in range(self.d):
                # Connect to next few nodes (cyclic)
                for offset in [1, 2, 3]:
                    j = (i + offset) % self.d
                    if i != j:
                        structure.add_edge(i, j)
            
            # Add some random long-range connections
            n_long_range = min(20, self.d * 2)
            pairs = [(i, j) for i in range(self.d) for j in range(i+3, self.d)]
            np.random.shuffle(pairs)
            for i, j in pairs[:n_long_range]:
                structure.add_edge(i, j)
                
        elif connectivity_type == 'dense':
            # Dense structure (careful with memory!)
            max_edges = min(200, self.d * (self.d - 1) // 4)  # Limit for memory
            all_pairs = [(i, j) for i in range(self.d) for j in range(i+1, self.d)]
            np.random.shuffle(all_pairs)
            for i, j in all_pairs[:max_edges]:
                structure.add_edge(i, j)
        
        print(f"TN structure created: {len(structure.nodes())} nodes, {len(structure.edges())} edges")
        return structure
    
    def fit_adaptive_tn(self, bond_dim: int = 8, epochs: int = 200, 
                       batch_size: int = 32, lr: float = 0.01) -> Optional[Any]:
        """Fit tensor network using adaptive TN module."""
        if not TN_AVAILABLE:
            print("❌ TensorNetwork modules not available")
            return None
        
        try:
            structure = self.create_tn_structure('sparse')
            
            print(f"Fitting Adaptive TN:")
            print(f"  Bond dimension: {bond_dim}")
            print(f"  Epochs: {epochs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Learning rate: {lr}")
            
            # Use first sample for structure (no batch dimension)
            tn = fit_tensor_network_fixed_structure(
                X=torch.tensor(self.X[0], dtype=torch.float32),
                Y=torch.tensor(self.y, dtype=torch.float32),
                structure=structure,
                bond_dim=bond_dim,
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr
            )
            
            return tn
            
        except Exception as e:
            print(f"❌ Adaptive TN fitting failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def fit_binary_tn(self, rank: int = 8, epochs: int = 200, lr: float = 0.01) -> Optional[Any]:
        """Fit binary tensor tree model."""
        if not TNTREE_AVAILABLE:
            print("❌ BinaryTensorTree not available")
            return None
        
        try:
            print(f"Fitting Binary TN Tree:")
            print(f"  Rank: {rank}")
            print(f"  Epochs: {epochs}")
            print(f"  Learning rate: {lr}")
            
            # Prepare data - reshape for tensor tree (each feature as separate "token")
            X_reshaped = self.X.reshape(self.n_samples, self.d, 1)  # [samples, features, 1]
            
            # Add bias dimension
            bias_dim = np.ones((self.n_samples, self.d, 1))
            X_with_bias = np.concatenate([X_reshaped, bias_dim], axis=2)  # [samples, features, 2]
            
            # Create model
            model = BinaryTensorTree(
                leaf_phys_dims=[2] * self.d,  # Each feature has 2 dimensions (value + bias)
                ranks=rank,
                out_dim=1,
                seed=self.seed,
                dtype=torch.float32
            )
            
            # Convert to tensors
            X_tensor = torch.tensor(X_with_bias, dtype=torch.float32)
            y_tensor = torch.tensor(self.y, dtype=torch.float32).unsqueeze(-1)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_tensor, y_tensor, test_size=0.2, random_state=self.seed
            )
            
            # Training setup
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            best_val_r2 = -float('inf')
            patience = 20
            no_improve = 0
            
            print("Training Binary TN...")
            for epoch in range(epochs):
                model.train()
                
                # Training
                optimizer.zero_grad()
                train_pred = model(X_train).squeeze()
                train_loss = criterion(train_pred, y_train.squeeze())
                train_loss.backward()
                optimizer.step()
                
                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_pred = model(X_val).squeeze()
                        val_loss = criterion(val_pred, y_val.squeeze())
                        val_r2 = r2_score_np(y_val.squeeze().numpy(), val_pred.numpy())
                        
                        print(f"  Epoch {epoch}: Train Loss: {train_loss.item():.6f}, "
                              f"Val Loss: {val_loss.item():.6f}, Val R²: {val_r2:.4f}")
                        
                        if val_r2 > best_val_r2:
                            best_val_r2 = val_r2
                            no_improve = 0
                        else:
                            no_improve += 1
                            if no_improve >= patience:
                                print(f"  Early stopping at epoch {epoch}")
                                break
            
            print(f"  Final validation R²: {best_val_r2:.4f}")
            return model if best_val_r2 > 0.8 else None
            
        except Exception as e:
            print(f"❌ Binary TN fitting failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# ===============================================================================
# SHAPIQ BASELINE
# ===============================================================================

def run_shapiq_baseline(data_generator: SyntheticDataGenerator, X_test: np.ndarray, 
                       x_point: np.ndarray, budget: int = 2000) -> Dict[str, Any]:
    """Run SHAPIQ baseline for comparison."""
    try:
        import shapiq
        print(f"Running SHAPIQ baseline (budget: {budget})...")
        
        # Create model wrapper
        def model_func(X):
            return data_generator.generate_function(X)
        
        class ModelWrapper:
            def __init__(self, func):
                self.func = func
            
            def predict(self, X):
                return self.func(X)
        
        model = ModelWrapper(model_func)
        
        # Background data (zeros)
        X_bg = np.zeros((1, data_generator.d))
        
        # Run SHAPIQ for different orders
        results = {}
        
        for k, index_name in [(1, 'SV'), (2, 'SII')]:
            for approximator in ['regression', 'permutation']:
                try:
                    explainer = shapiq.TabularExplainer(
                        model=model,
                        data=X_bg,
                        approximator=approximator,
                        index=index_name,
                        max_order=k,
                        random_state=42
                    )
                    
                    start_time = time.time()
                    explanation = explainer.explain(x_point.reshape(1, -1), budget=budget)
                    runtime = time.time() - start_time
                    
                    if k == 1:
                        if index_name == "SV":
                            values = np.array(explanation.get_values()).flatten()
                        else:
                            values = np.array(explanation.get_n_order_values(1)).flatten()
                    else:  # k == 2
                        tensor = explanation.get_n_order_values(k)
                        values = []
                        for i in range(data_generator.d):
                            for j in range(i+1, data_generator.d):
                                values.append(tensor[i, j])
                        values = np.array(values)
                    
                    method_name = f"shapiq_{approximator}_k{k}_{index_name}"
                    results[method_name] = {
                        'values': values,
                        'runtime': runtime,
                        'method': method_name
                    }
                    
                    print(f"  {method_name}: {len(values)} values, {runtime:.2f}s")
                    
                except Exception as e:
                    print(f"  ❌ {approximator} k={k} failed: {e}")
        
        return results
        
    except ImportError:
        print("❌ SHAPIQ not available")
        return {}
    except Exception as e:
        print(f"❌ SHAPIQ failed: {e}")
        return {}

# ===============================================================================
# MASK AUGMENTATION
# ===============================================================================

def generate_mask_augmented_data(X: np.ndarray, y: np.ndarray, 
                               n_masks: int = 1000, mask_prob: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate mask-augmented training data."""
    n_samples, d = X.shape
    
    # Generate random masks
    masks = np.random.binomial(1, 1 - mask_prob, (n_masks, d)).astype(bool)
    
    # Apply masks to random samples
    X_masked = []
    y_masked = []
    mask_info = []
    
    for mask in masks:
        # Random sample
        idx = np.random.randint(n_samples)
        x_orig = X[idx].copy()
        
        # Apply mask (set masked features to 0)
        x_masked = x_orig.copy()
        x_masked[~mask] = 0
        
        X_masked.append(x_masked)
        y_masked.append(y[idx])  # Keep original target
        mask_info.append(mask)
    
    X_masked = np.array(X_masked)
    y_masked = np.array(y_masked)
    mask_info = np.array(mask_info)
    
    print(f"Generated {n_masks} mask-augmented samples")
    print(f"  Average mask rate: {(1 - mask_info.mean()):.2f}")
    
    return X_masked, y_masked, mask_info

# ===============================================================================
# MAIN EXPERIMENT
# ===============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive TN-Shapley Example')
    parser.add_argument('--d', type=int, default=100, help='Number of features')
    parser.add_argument('--n_samples', type=int, default=2000, help='Number of samples')
    parser.add_argument('--n_strong_pairs', type=int, default=20, help='Number of strong interactions')
    parser.add_argument('--n_weak_pairs', type=int, default=30, help='Number of weak interactions')
    parser.add_argument('--noise_std', type=float, default=0.1, help='Noise standard deviation')
    parser.add_argument('--tn_bond_dim', type=int, default=6, help='TN bond dimension')
    parser.add_argument('--tn_rank', type=int, default=8, help='Binary TN rank')
    parser.add_argument('--tn_epochs', type=int, default=200, help='TN training epochs')
    parser.add_argument('--shapiq_budget', type=int, default=2000, help='SHAPIQ budget')
    parser.add_argument('--use_masks', action='store_true', help='Use mask augmentation')
    parser.add_argument('--n_masks', type=int, default=500, help='Number of mask samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--outdir', type=str, default='tn_shapley_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    ensure_dir(args.outdir)
    set_all_seeds(args.seed)
    
    print("🚀 Comprehensive TN-Shapley Experiment")
    print("="*60)
    print(f"Features: {args.d}")
    print(f"Samples: {args.n_samples}")
    print(f"Strong pairs: {args.n_strong_pairs}")
    print(f"Weak pairs: {args.n_weak_pairs}")
    print(f"Seed: {args.seed}")
    
    # ============= STEP 1: GENERATE SYNTHETIC DATA =============
    print(f"\n📊 STEP 1: Generating synthetic data...")
    
    data_gen = SyntheticDataGenerator(
        d=args.d,
        n_strong_pairs=args.n_strong_pairs,
        n_weak_pairs=args.n_weak_pairs,
        noise_std=args.noise_std,
        seed=args.seed
    )
    
    X, y = data_gen.generate_dataset(args.n_samples, x_dist='normal')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"  Training set: {X_train_scaled.shape}")
    print(f"  Test set: {X_test_scaled.shape}")
    print(f"  Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # ============= STEP 2: MASK AUGMENTATION (Optional) =============
    if args.use_masks:
        print(f"\n🎭 STEP 2: Generating mask-augmented data...")
        X_masked, y_masked, masks = generate_mask_augmented_data(
            X_train_scaled, y_train_scaled, n_masks=args.n_masks
        )
        
        # Combine original and masked data
        X_train_combined = np.vstack([X_train_scaled, X_masked])
        y_train_combined = np.hstack([y_train_scaled, y_masked])
        
        print(f"  Combined training set: {X_train_combined.shape}")
    else:
        X_train_combined = X_train_scaled
        y_train_combined = y_train_scaled
        masks = None
        print(f"\n🎭 STEP 2: Skipping mask augmentation")
    
    # ============= STEP 3: TENSOR NETWORK FITTING =============
    print(f"\n🧠 STEP 3: Fitting Tensor Networks...")
    
    tn_fitter = TensorNetworkFitter(X_train_combined, y_train_combined, seed=args.seed)
    
    # Try Binary TN first (usually more stable)
    binary_tn = tn_fitter.fit_binary_tn(rank=args.tn_rank, epochs=args.tn_epochs)
    
    # Try Adaptive TN
    adaptive_tn = None
    if TN_AVAILABLE and args.d <= 50:  # Limit for memory reasons
        adaptive_tn = tn_fitter.fit_adaptive_tn(
            bond_dim=args.tn_bond_dim, 
            epochs=args.tn_epochs
        )
    
    # Select best model
    best_tn = None
    best_tn_type = None
    
    if binary_tn is not None:
        # Evaluate binary TN
        X_test_for_binary = X_test_scaled.reshape(len(X_test_scaled), args.d, 1)
        bias_test = np.ones((len(X_test_scaled), args.d, 1))
        X_test_with_bias = np.concatenate([X_test_for_binary, bias_test], axis=2)
        
        with torch.no_grad():
            binary_tn.eval()
            y_pred_binary = binary_tn(torch.tensor(X_test_with_bias, dtype=torch.float32))
            y_pred_binary = y_pred_binary.squeeze().numpy()
            
        r2_binary = r2_score_np(y_test_scaled, y_pred_binary)
        print(f"  Binary TN R²: {r2_binary:.4f}")
        
        if r2_binary > 0.9:
            best_tn = binary_tn
            best_tn_type = 'binary'
            print(f"✅ Binary TN achieves R² > 0.9: {r2_binary:.4f}")
    
    if adaptive_tn is not None:
        # Evaluate adaptive TN
        adaptive_tn.eval()
        y_pred_adaptive = []
        
        with torch.no_grad():
            for i in range(len(X_test_scaled)):
                adaptive_tn.update_input_data(torch.tensor(X_test_scaled[i], dtype=torch.float32))
                pred = adaptive_tn.contract()
                y_pred_adaptive.append(pred.item())
        
        y_pred_adaptive = np.array(y_pred_adaptive)
        r2_adaptive = r2_score_np(y_test_scaled, y_pred_adaptive)
        print(f"  Adaptive TN R²: {r2_adaptive:.4f}")
        
        if r2_adaptive > 0.9 and (best_tn is None or r2_adaptive > r2_binary):
            best_tn = adaptive_tn
            best_tn_type = 'adaptive'
            print(f"✅ Adaptive TN achieves R² > 0.9: {r2_adaptive:.4f}")
    
    if best_tn is None:
        print("❌ No TN model achieved R² > 0.9")
        print("Proceeding with ground truth analysis only...")
    
    # ============= STEP 4: SHAPLEY VALUE COMPUTATION =============
    print(f"\n🎯 STEP 4: Computing Shapley values...")
    
    # Choose evaluation point (center of test set)
    x_eval = np.mean(X_test_scaled, axis=0)
    print(f"  Evaluation point: mean of test set")
    
    # Ground truth Shapley values
    print("  Computing ground truth Shapley values...")
    gt_shapley_k1, gt_shapley_k2 = data_gen.compute_ground_truth_shapley(
        scaler_X.inverse_transform(x_eval.reshape(1, -1)).flatten()
    )
    
    print(f"    Order 1: {len(gt_shapley_k1)} values, range [{gt_shapley_k1.min():.4f}, {gt_shapley_k1.max():.4f}]")
    print(f"    Order 2: {len(gt_shapley_k2)} values, range [{gt_shapley_k2.min():.4f}, {gt_shapley_k2.max():.4f}]")
    
    # GAM-based Shapley computation
    print("  Computing GAM-based Shapley values...")
    
    def model_func(X_input):
        """Model function for GAM computation."""
        X_orig = scaler_X.inverse_transform(X_input)
        return data_gen.generate_function(X_orig)
    
    gam_computer = GAMShapleyComputer(X_train_scaled, y_train_scaled, model_func)
    gam_shapley_k1 = gam_computer.compute_marginal_contributions(x_eval)
    gam_shapley_k2 = gam_computer.compute_pairwise_interactions(x_eval)
    
    print(f"    GAM Order 1: range [{gam_shapley_k1.min():.4f}, {gam_shapley_k1.max():.4f}]")
    print(f"    GAM Order 2: range [{gam_shapley_k2.min():.4f}, {gam_shapley_k2.max():.4f}]")
    
    # TN-based Shapley computation (if model available)
    tn_shapley_k1 = None
    tn_shapley_k2 = None
    
    if best_tn is not None and best_tn_type == 'binary':
        print("  Computing TN-based Shapley values...")
        try:
            # Simple TN-SHAP computation
            x_eval_reshaped = x_eval.reshape(1, args.d, 1)
            bias_eval = np.ones((1, args.d, 1))
            x_eval_with_bias = np.concatenate([x_eval_reshaped, bias_eval], axis=2)
            
            baseline = np.zeros_like(x_eval_with_bias)
            baseline[:, :, -1] = 1  # Keep bias
            
            with torch.no_grad():
                best_tn.eval()
                baseline_pred = best_tn(torch.tensor(baseline, dtype=torch.float32)).item()
                
                # Order 1: Individual feature contributions
                tn_shapley_k1 = []
                for i in range(args.d):
                    intervention = baseline.copy()
                    intervention[:, i, 0] = x_eval[i]  # Set feature i
                    intervention_pred = best_tn(torch.tensor(intervention, dtype=torch.float32)).item()
                    tn_shapley_k1.append(intervention_pred - baseline_pred)
                
                tn_shapley_k1 = np.array(tn_shapley_k1)
                
                # Order 2: Pairwise interactions (sample a subset)
                max_pairs = min(100, args.d * (args.d - 1) // 2)  # Limit computation
                pairs = [(i, j) for i in range(args.d) for j in range(i+1, args.d)]
                np.random.shuffle(pairs)
                selected_pairs = pairs[:max_pairs]
                
                tn_shapley_k2 = []
                for i, j in selected_pairs:
                    # Individual contributions
                    interv_i = baseline.copy()
                    interv_i[:, i, 0] = x_eval[i]
                    pred_i = best_tn(torch.tensor(interv_i, dtype=torch.float32)).item()
                    
                    interv_j = baseline.copy()
                    interv_j[:, j, 0] = x_eval[j]
                    pred_j = best_tn(torch.tensor(interv_j, dtype=torch.float32)).item()
                    
                    # Joint contribution
                    interv_ij = baseline.copy()
                    interv_ij[:, i, 0] = x_eval[i]
                    interv_ij[:, j, 0] = x_eval[j]
                    pred_ij = best_tn(torch.tensor(interv_ij, dtype=torch.float32)).item()
                    
                    # Interaction
                    interaction = (pred_ij - baseline_pred) - (pred_i - baseline_pred) - (pred_j - baseline_pred)
                    tn_shapley_k2.append(interaction)
                
                tn_shapley_k2 = np.array(tn_shapley_k2)
                
            print(f"    TN Order 1: range [{tn_shapley_k1.min():.4f}, {tn_shapley_k1.max():.4f}]")
            print(f"    TN Order 2: {len(tn_shapley_k2)} pairs, range [{tn_shapley_k2.min():.4f}, {tn_shapley_k2.max():.4f}]")
            
        except Exception as e:
            print(f"    ❌ TN Shapley computation failed: {e}")
    
    # ============= STEP 5: SHAPIQ BASELINE =============
    if args.shapiq_budget > 0:
        print(f"\n📈 STEP 5: Running SHAPIQ baseline...")
        shapiq_results = run_shapiq_baseline(
            data_gen, X_test_scaled, 
            scaler_X.inverse_transform(x_eval.reshape(1, -1)).flatten(),
            budget=args.shapiq_budget
        )
    else:
        shapiq_results = {}
        print(f"\n📈 STEP 5: Skipping SHAPIQ baseline")
    
    # ============= STEP 6: COMPARISON AND ANALYSIS =============
    print(f"\n📊 STEP 6: Comparison and Analysis...")
    
    results = {
        'ground_truth': {
            'shapley_k1': gt_shapley_k1.tolist(),
            'shapley_k2': gt_shapley_k2.tolist(),
            'strong_pairs': data_gen.strong_pairs,
            'weak_pairs': data_gen.weak_pairs,
            'linear_coeffs': data_gen.linear_coeffs.tolist()
        },
        'gam': {
            'shapley_k1': gam_shapley_k1.tolist(),
            'shapley_k2': gam_shapley_k2.tolist()
        },
        'tensor_network': {
            'type': best_tn_type,
            'r2_achieved': r2_binary if best_tn_type == 'binary' else (r2_adaptive if best_tn_type == 'adaptive' else None),
            'shapley_k1': tn_shapley_k1.tolist() if tn_shapley_k1 is not None else None,
            'shapley_k2': tn_shapley_k2.tolist() if tn_shapley_k2 is not None else None
        },
        'shapiq': shapiq_results,
        'evaluation_point': x_eval.tolist(),
        'experiment_config': {
            'd': args.d,
            'n_samples': args.n_samples,
            'n_strong_pairs': args.n_strong_pairs,
            'n_weak_pairs': args.n_weak_pairs,
            'noise_std': args.noise_std,
            'seed': args.seed,
            'use_masks': args.use_masks
        }
    }
    
    # Compute comparisons
    comparisons = {}
    
    # GAM vs Ground Truth
    cos_gam_gt_k1 = cosine_similarity(gam_shapley_k1, gt_shapley_k1)
    r2_gam_gt_k1 = r2_score_np(gt_shapley_k1, gam_shapley_k1)
    
    cos_gam_gt_k2 = cosine_similarity(gam_shapley_k2, gt_shapley_k2)
    r2_gam_gt_k2 = r2_score_np(gt_shapley_k2, gam_shapley_k2)
    
    comparisons['gam_vs_gt'] = {
        'cosine_k1': cos_gam_gt_k1,
        'r2_k1': r2_gam_gt_k1,
        'cosine_k2': cos_gam_gt_k2,
        'r2_k2': r2_gam_gt_k2
    }
    
    print(f"  GAM vs Ground Truth:")
    print(f"    Order 1 - Cosine: {cos_gam_gt_k1:.4f}, R²: {r2_gam_gt_k1:.4f}")
    print(f"    Order 2 - Cosine: {cos_gam_gt_k2:.4f}, R²: {r2_gam_gt_k2:.4f}")
    
    # TN vs Ground Truth
    if tn_shapley_k1 is not None:
        cos_tn_gt_k1 = cosine_similarity(tn_shapley_k1, gt_shapley_k1)
        r2_tn_gt_k1 = r2_score_np(gt_shapley_k1, tn_shapley_k1)
        
        cos_tn_gam_k1 = cosine_similarity(tn_shapley_k1, gam_shapley_k1)
        r2_tn_gam_k1 = r2_score_np(gam_shapley_k1, tn_shapley_k1)
        
        comparisons['tn_vs_gt'] = {
            'cosine_k1': cos_tn_gt_k1,
            'r2_k1': r2_tn_gt_k1
        }
        
        comparisons['tn_vs_gam'] = {
            'cosine_k1': cos_tn_gam_k1,
            'r2_k1': r2_tn_gam_k1
        }
        
        print(f"  TN vs Ground Truth:")
        print(f"    Order 1 - Cosine: {cos_tn_gt_k1:.4f}, R²: {r2_tn_gt_k1:.4f}")
        print(f"  TN vs GAM:")
        print(f"    Order 1 - Cosine: {cos_tn_gam_k1:.4f}, R²: {r2_tn_gam_k1:.4f}")
    
    results['comparisons'] = comparisons
    
    # ============= STEP 7: SAVE RESULTS =============
    print(f"\n💾 STEP 7: Saving results...")
    
    # Save main results
    results_file = os.path.join(args.outdir, 'tn_shapley_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save data
    data_file = os.path.join(args.outdir, 'data.npz')
    np.savez(
        data_file,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train_scaled,
        y_test=y_test_scaled,
        x_eval=x_eval,
        masks=masks if masks is not None else np.array([])
    )
    
    # Create summary plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Order 1 comparison
        axes[0, 0].scatter(gt_shapley_k1, gam_shapley_k1, alpha=0.6, label='GAM')
        if tn_shapley_k1 is not None:
            axes[0, 0].scatter(gt_shapley_k1, tn_shapley_k1, alpha=0.6, label='TN')
        axes[0, 0].plot([gt_shapley_k1.min(), gt_shapley_k1.max()], 
                       [gt_shapley_k1.min(), gt_shapley_k1.max()], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('Ground Truth Shapley (Order 1)')
        axes[0, 0].set_ylabel('Predicted Shapley (Order 1)')
        axes[0, 0].set_title('Order 1 Shapley Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Order 2 comparison
        axes[0, 1].scatter(gt_shapley_k2, gam_shapley_k2, alpha=0.6, label='GAM')
        if tn_shapley_k2 is not None and len(tn_shapley_k2) <= len(gt_shapley_k2):
            # Only plot available TN pairs
            axes[0, 1].scatter(gt_shapley_k2[:len(tn_shapley_k2)], tn_shapley_k2, alpha=0.6, label='TN')
        axes[0, 1].plot([gt_shapley_k2.min(), gt_shapley_k2.max()], 
                       [gt_shapley_k2.min(), gt_shapley_k2.max()], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('Ground Truth Shapley (Order 2)')
        axes[0, 1].set_ylabel('Predicted Shapley (Order 2)')
        axes[0, 1].set_title('Order 2 Shapley Values')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature importance ranking
        top_k = min(20, args.d)
        gt_top_idx = np.argsort(np.abs(gt_shapley_k1))[-top_k:]
        gam_top_idx = np.argsort(np.abs(gam_shapley_k1))[-top_k:]
        
        axes[1, 0].barh(range(top_k), np.abs(gt_shapley_k1[gt_top_idx]), alpha=0.7, label='Ground Truth')
        axes[1, 0].barh(range(top_k), np.abs(gam_shapley_k1[gt_top_idx]), alpha=0.7, label='GAM')
        axes[1, 0].set_xlabel('Absolute Shapley Value')
        axes[1, 0].set_ylabel('Feature Rank')
        axes[1, 0].set_title(f'Top {top_k} Features by Importance')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary metrics
        summary_text = f"""
Experiment Summary:
• Features: {args.d}
• Strong pairs: {args.n_strong_pairs}
• Weak pairs: {args.n_weak_pairs}
• TN R²: {results['tensor_network']['r2_achieved']:.4f if results['tensor_network']['r2_achieved'] else 'N/A'}

GAM vs GT:
• Order 1 Cosine: {cos_gam_gt_k1:.3f}
• Order 1 R²: {r2_gam_gt_k1:.3f}
• Order 2 Cosine: {cos_gam_gt_k2:.3f}
• Order 2 R²: {r2_gam_gt_k2:.3f}

TN vs GT:
• Order 1 Cosine: {comparisons.get('tn_vs_gt', {}).get('cosine_k1', 0):.3f}
• Order 1 R²: {comparisons.get('tn_vs_gt', {}).get('r2_k1', 0):.3f}
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plot_file = os.path.join(args.outdir, 'tn_shapley_analysis.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Plot saved: {plot_file}")
        
    except Exception as e:
        print(f"  ❌ Plotting failed: {e}")
    
    # Final summary
    print(f"\n🎉 EXPERIMENT COMPLETED!")
    print("="*60)
    print(f"Results saved to: {args.outdir}")
    print(f"Main results: {results_file}")
    print(f"Data: {data_file}")
    
    if best_tn is not None:
        print(f"✅ Tensor Network R² > 0.9: {results['tensor_network']['r2_achieved']:.4f}")
    else:
        print(f"❌ Tensor Network did not achieve R² > 0.9")
    
    print(f"✅ GAM Shapley analysis completed")
    print(f"✅ Ground truth comparison completed")
    
    if shapiq_results:
        print(f"✅ SHAPIQ baseline completed ({len(shapiq_results)} methods)")
    else:
        print(f"⚠️ SHAPIQ baseline skipped")

if __name__ == "__main__":
    main()
