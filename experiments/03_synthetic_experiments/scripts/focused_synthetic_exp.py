#!/usr/bin/env python3
"""
Focused multilinear Shapley experiment with three scenarios:
1. Low-rank multilinear ground truth from tensor tree
2. Higher-order tree ground truth vs same-rank training tree
3. Generic multilinear function vs tensor tree training
"""

import numpy as np
import torch
import torch.nn as nn
import itertools
import math
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score
from tntree_model import BinaryTensorTree, make_balanced_binary_tensor_tree

class TensorTreeGenerator:
    """Generate ground truth multilinear functions from tensor trees."""
    
    def __init__(self, n_features: int, ranks: int, seed: int = 42):
        self.n_features = n_features
        self.ranks = ranks
        self.seed = seed
        
        # Create ground truth tensor tree
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        phys_dims = [2] * n_features  # Binary selectors
        self.gt_tree = make_balanced_binary_tensor_tree(
            leaf_phys_dims=phys_dims,
            ranks=ranks,
            out_dim=1,
            assume_bias_when_matrix=True,
            seed=seed
        )
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate ground truth tensor tree."""
        self.gt_tree.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y = self.gt_tree.forward(X_tensor)
            
            # Ensure consistent output format
            if y.dim() == 0:  # scalar
                return np.array([y.item()])
            else:
                result = y.squeeze().detach().cpu().numpy()
                if result.ndim == 0:  # 0d array
                    return np.array([result.item()])
                return result

class GenericMultilinearGenerator:
    """Generate generic multilinear functions with explicit coefficients."""
    
    def __init__(self, n_features: int, max_order: int = 3, seed: int = 42):
        self.n_features = n_features
        self.max_order = max_order
        self.seed = seed
        np.random.seed(seed)
        
        # Generate coefficients for subsets up to max_order
        self.coefficients = {}
        
        # Constant term (zero baseline)
        self.coefficients[()] = 0.0
        
        # Linear terms (moderate strength)
        for i in range(n_features):
            self.coefficients[(i,)] = np.random.normal(0, 1.0)
        
        # Pairwise interactions (stronger)
        for i in range(n_features):
            for j in range(i+1, n_features):
                self.coefficients[(i, j)] = np.random.normal(0, 2.0)
        
        # Triplet interactions (if max_order >= 3)
        if max_order >= 3:
            for subset in itertools.combinations(range(n_features), 3):
                self.coefficients[subset] = np.random.normal(0, 1.5)
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate multilinear function."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        y = np.zeros(n_samples)
        
        for subset, coeff in self.coefficients.items():
            if len(subset) == 0:
                y += coeff
            else:
                term = np.ones(n_samples)
                for idx in subset:
                    term *= X[:, idx]
                y += coeff * term
        
        return y

class ExactShapleyCalculator:
    """Calculate exact Shapley values via enumeration."""
    
    def __init__(self, generator):
        self.generator = generator
        if hasattr(generator, 'n_features'):
            self.n_features = generator.n_features
        else:
            # Infer from generator
            test_input = np.zeros(8)  # assume 8 features for now
            try:
                self.generator.evaluate(test_input)
                self.n_features = 8
            except:
                self.n_features = 4
    
    def _coalition_value(self, x: np.ndarray, coalition: Tuple[int, ...]) -> float:
        """Zero-baseline coalition value."""
        x_masked = np.zeros(self.n_features)
        for idx in coalition:
            x_masked[idx] = x[idx]
        
        result = self.generator.evaluate(x_masked.reshape(1, -1))
        
        # Handle different output types
        if isinstance(result, (int, float)):
            return float(result)
        elif hasattr(result, 'item'):  # numpy scalar
            return float(result.item())
        elif hasattr(result, '__len__') and len(result) > 0:
            return float(result[0])
        elif hasattr(result, '__iter__'):
            result_array = np.array(list(result))
            if result_array.size > 0:
                return float(result_array.flat[0])
        
        # Last resort - try to convert directly
        return float(result)
    
    def shapley_values(self, x: np.ndarray) -> np.ndarray:
        """Compute exact Shapley values."""
        shapley = np.zeros(self.n_features)
        n = self.n_features
        
        for i in range(n):
            value = 0.0
            for coalition_size in range(n):
                for coalition in itertools.combinations([j for j in range(n) if j != i], coalition_size):
                    coalition_with_i = tuple(sorted(coalition + (i,)))
                    coalition_without_i = coalition
                    
                    # Correct Shapley weight
                    weight = (math.factorial(coalition_size) * 
                             math.factorial(n - coalition_size - 1) / 
                             math.factorial(n))
                    
                    marginal = (self._coalition_value(x, coalition_with_i) - 
                               self._coalition_value(x, coalition_without_i))
                    
                    value += weight * marginal
            
            shapley[i] = value
        
        return shapley
    
    def shapley_interactions(self, x: np.ndarray, order: int) -> Dict[Tuple[int, ...], float]:
        """Compute Shapley interactions of given order."""
        interactions = {}
        
        if order == 1:
            shap_vals = self.shapley_values(x)
            for i in range(self.n_features):
                interactions[(i,)] = shap_vals[i]
            return interactions
        
        # For multilinear functions, use inclusion-exclusion
        for subset in itertools.combinations(range(self.n_features), order):
            total_value = 0.0
            
            for k in range(len(subset) + 1):
                for sub_subset in itertools.combinations(subset, k):
                    val = self._coalition_value(x, sub_subset)
                    sign = (-1) ** (len(subset) - k)
                    total_value += sign * val
            
            interactions[subset] = total_value
        
        return interactions

class TensorNetworkShapleyCalculator:
    """Calculate Shapley using trained tensor network."""
    
    def __init__(self, trained_tree: BinaryTensorTree):
        self.tensor_tree = trained_tree
        self.n_features = len(trained_tree.leaf_ids)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using tensor network."""
        self.tensor_tree.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_pred = self.tensor_tree.forward(X_tensor)
            
            # Handle different output shapes consistently
            if y_pred.dim() == 0:  # scalar tensor
                return np.array([y_pred.item()])
            else:
                result = y_pred.squeeze().detach().cpu().numpy()
                if result.ndim == 0:  # 0d numpy array
                    return np.array([result.item()])
                return result
    
    def shapley_values_input_scaling(self, x: np.ndarray) -> np.ndarray:
        """Compute Shapley values using input scaling."""
        x = np.asarray(x, dtype=np.float64)
        d = x.shape[0]
        
        # Use t = 0, 1, 2, ..., d-1
        t = np.arange(d, dtype=np.float64)
        
        # Vandermonde system
        V = np.vander(t, N=d, increasing=True)
        V_inv = np.linalg.solve(V, np.eye(d))
        weights = 1.0 / (np.arange(1, d + 1, dtype=np.float64))
        
        phi = np.zeros(d, dtype=np.float64)
        
        for i in range(d):
            others_mask = np.ones(d, dtype=bool)
            others_mask[i] = False
            
            # s=1: feature i unchanged, others scaled by t
            X1 = np.repeat(x[None, :], d, axis=0)
            X1[:, others_mask] *= t[:, None]
            
            # s=0: feature i zeroed, others scaled by t
            X0 = np.repeat(x[None, :], d, axis=0)
            X0[:, others_mask] *= t[:, None]
            X0[:, i] = 0.0
            
            G1 = self._predict(X1)
            G0 = self._predict(X0)
            H = G1 - G0
            
            if H.ndim == 0:
                H = np.array([H])
            
            c_i = V_inv @ H
            phi[i] = float((c_i * weights).sum())
        
        return phi
    
    def shapley_interactions(self, x: np.ndarray, order: int) -> Dict[Tuple[int, ...], float]:
        """Compute Shapley interactions."""
        interactions = {}
        
        if order == 1:
            shap_vals = self.shapley_values_input_scaling(x)
            for i in range(self.n_features):
                interactions[(i,)] = shap_vals[i]
            return interactions
        
        # Higher-order interactions via inclusion-exclusion
        for subset in itertools.combinations(range(self.n_features), order):
            total_value = 0.0
            
            for k in range(len(subset) + 1):
                for sub_subset in itertools.combinations(subset, k):
                    x_eval = np.zeros(self.n_features)
                    for idx in sub_subset:
                        x_eval[idx] = x[idx]
                    
                    val = self._predict(x_eval.reshape(1, -1))
                    if hasattr(val, '__len__') and len(val) > 0:
                        val = val[0]
                    elif hasattr(val, 'item'):
                        val = val.item()
                    
                    sign = (-1) ** (len(subset) - k)
                    total_value += sign * val
            
            interactions[subset] = total_value
        
        return interactions

def train_tensor_tree(generator, n_features: int, ranks: int, n_samples: int = 10000, epochs: int = 1000):
    """Train tensor tree to approximate generator."""
    print(f"Training tensor tree (rank={ranks}) with {n_samples} samples...")
    
    # Generate training data
    X_train = np.random.normal(0, 1, (n_samples, n_features))
    y_train = generator.evaluate(X_train)
    
    # Create and train tensor tree
    phys_dims = [2] * n_features
    tensor_tree = make_balanced_binary_tensor_tree(
        leaf_phys_dims=phys_dims,
        ranks=ranks,
        out_dim=1,
        assume_bias_when_matrix=True,
        seed=42
    )
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(tensor_tree.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        y_pred = tensor_tree.forward(X_tensor).squeeze()
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
        
        if epoch % 100 == 0:
            r2 = r2_score(y_tensor.detach().numpy(), y_pred.detach().numpy())
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}, R² = {r2:.6f}")
        
        if patience > 50:
            break
    
    final_r2 = r2_score(y_tensor.detach().numpy(), y_pred.detach().numpy())
    print(f"  Final R² = {final_r2:.6f}")
    
    return tensor_tree, final_r2

def run_experiment(scenario_name: str, gt_generator, train_ranks: int, n_features: int = 4):
    """Run single experiment scenario."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    
    # Test point
    x_test = np.random.normal(0, 1, n_features)
    print(f"Test point: {x_test}")
    
    # Ground truth Shapley values
    print("\nComputing ground truth Shapley values...")
    exact_calc = ExactShapleyCalculator(gt_generator)
    
    gt_results = {}
    for order in range(1, 4):  # Orders 1, 2, 3
        gt_interactions = exact_calc.shapley_interactions(x_test, order)
        gt_results[order] = gt_interactions
        print(f"  Order {order}: {len(gt_interactions)} interactions")
    
    # Train tensor network
    print(f"\nTraining tensor network (rank={train_ranks})...")
    trained_tree, train_r2 = train_tensor_tree(gt_generator, n_features, train_ranks)
    
    if train_r2 < 0.95:
        print(f"WARNING: Training R² = {train_r2:.6f} < 0.95")
    
    # Compute TN Shapley values
    print("\nComputing tensor network Shapley values...")
    tn_calc = TensorNetworkShapleyCalculator(trained_tree)
    
    tn_results = {}
    order_r2s = {}
    
    for order in range(1, 4):
        tn_interactions = tn_calc.shapley_interactions(x_test, order)
        tn_results[order] = tn_interactions
        
        # Compare with ground truth
        gt_vals = [gt_results[order][subset] for subset in gt_results[order].keys()]
        tn_vals = [tn_interactions[subset] for subset in gt_results[order].keys()]
        
        if gt_vals:
            r2 = r2_score(gt_vals, tn_vals)
            order_r2s[order] = r2
            print(f"  Order {order} R² = {r2:.6f}")
            
            # Show examples
            for i, (subset, gt_val) in enumerate(list(gt_results[order].items())[:3]):
                tn_val = tn_interactions[subset]
                error = abs(gt_val - tn_val)
                print(f"    {subset}: GT={gt_val:.6f}, TN={tn_val:.6f}, err={error:.6f}")
    
    return {
        'train_r2': train_r2,
        'order_r2s': order_r2s,
        'gt_results': gt_results,
        'tn_results': tn_results
    }

def main():
    """Run all three experiment scenarios."""
    n_features = 4
    
    results = {}
    
    # Scenario 1: Low-rank tree ground truth
    print("Creating low-rank tensor tree ground truth...")
    gt_generator_1 = TensorTreeGenerator(n_features, ranks=4, seed=42)
    results['scenario_1'] = run_experiment(
        "Low-rank tensor tree GT (rank=4) vs Training (rank=4)",
        gt_generator_1, train_ranks=4, n_features=n_features
    )
    
    # Scenario 2: Higher-order tree ground truth vs same-rank training
    print("\nCreating higher-order tensor tree ground truth...")
    gt_generator_2 = TensorTreeGenerator(n_features, ranks=8, seed=43)
    results['scenario_2'] = run_experiment(
        "Higher-order tree GT (rank=8) vs Training (rank=4)",
        gt_generator_2, train_ranks=4, n_features=n_features
    )
    
    # Scenario 3: Generic multilinear vs tensor tree training
    print("\nCreating generic multilinear ground truth...")
    gt_generator_3 = GenericMultilinearGenerator(n_features, max_order=3, seed=44)
    results['scenario_3'] = run_experiment(
        "Generic multilinear GT vs Training (rank=6)",
        gt_generator_3, train_ranks=6, n_features=n_features
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for scenario, result in results.items():
        print(f"\n{scenario.upper()}:")
        print(f"  Training R² = {result['train_r2']:.6f}")
        for order, r2 in result['order_r2s'].items():
            status = "✓" if r2 > 0.9 else "⚠" if r2 > 0.7 else "✗"
            print(f"  Order {order} R² = {r2:.6f} {status}")
    
    return results

if __name__ == "__main__":
    results = main()
    print("\nExperiment completed!")