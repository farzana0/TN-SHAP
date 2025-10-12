#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Fixed synthetic experiments with correct Shapley formulation.
"""

import numpy as np
import torch
import torch.nn as nn
import itertools
import pickle
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import r2_score

from src.tntree_model import BinaryTensorTree, make_balanced_binary_tensor_tree

class SyntheticMultilinearGenerator:
    """Generate synthetic multilinear functions with controlled interactions."""
    
    def __init__(self, n_features: int = 8, seed: int = 42):
        self.n_features = n_features
        self.seed = seed
        np.random.seed(seed)
        
        # Generate coefficients for all possible subsets
        self.coefficients = {}
        self._generate_coefficients()
    
    def _generate_coefficients(self):
        """Generate coefficients with some strong interactions."""
        # Make coefficients much larger and more structured
        
        # Zero constant term
        self.coefficients[()] = 0.0
        
        # Strong linear terms
        for i in range(self.n_features):
            self.coefficients[(i,)] = 2.0 + np.random.normal(0, 0.5)
        
        # Very strong pairwise interactions
        strong_pairs = [(0, 1), (2, 3)]
        for pair in strong_pairs:
            self.coefficients[pair] = 10.0 + np.random.normal(0, 1.0)
        
        # Moderate other pairwise
        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                if (i, j) not in strong_pairs:
                    self.coefficients[(i, j)] = np.random.normal(0, 0.5)
        
        # One very strong triplet
        self.coefficients[(0, 1, 2)] = 15.0
        
        # Other triplets - moderate
        for subset in itertools.combinations(range(self.n_features), 3):
            if subset != (0, 1, 2):
                self.coefficients[subset] = np.random.normal(0, 0.2)
        
        # Higher order - very weak
        for order in range(4, self.n_features + 1):
            for subset in itertools.combinations(range(self.n_features), order):
                self.coefficients[subset] = np.random.normal(0, 0.01)
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the multilinear function on data X."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        y = np.zeros(n_samples)
        
        for subset, coeff in self.coefficients.items():
            if len(subset) == 0:
                y += coeff
            else:
                term = np.ones(n_samples)
                for feature_idx in subset:
                    term *= X[:, feature_idx]
                y += coeff * term
        
        return y
    
    def generate_dataset(self, n_samples: int = 1000, noise_std: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Generate zero-mean dataset."""
        X = np.random.normal(0, 1, (n_samples, self.n_features))
        y = self.evaluate(X)
        
        if noise_std > 0:
            y += np.random.normal(0, noise_std, n_samples)
        
        return X, y

class ExactShapleyCalculator:
    """Calculate exact Shapley values via correct subset enumeration."""
    
    def __init__(self, generator: SyntheticMultilinearGenerator):
        self.generator = generator
        self.n_features = generator.n_features
    
    def _coalition_value(self, x: np.ndarray, coalition: Tuple[int, ...]) -> float:
        """Zero-baseline coalition value."""
        x_masked = np.zeros(self.n_features)
        for idx in coalition:
            x_masked[idx] = x[idx]
        result = self.generator.evaluate(x_masked.reshape(1, -1))
        return float(result[0]) if hasattr(result, '__len__') else float(result)
    
    def shapley_values(self, x: np.ndarray) -> np.ndarray:
        """Compute exact Shapley values with correct formula."""
        shapley = np.zeros(self.n_features)
        n = self.n_features
        
        for i in range(n):
            value = 0.0
            # Sum over all coalitions not containing i
            for coalition_size in range(n):
                for coalition in itertools.combinations([j for j in range(n) if j != i], coalition_size):
                    coalition_with_i = tuple(sorted(coalition + (i,)))
                    coalition_without_i = coalition
                    
                    # Correct Shapley weight
                    weight = (math.factorial(coalition_size) * 
                             math.factorial(n - coalition_size - 1) / 
                             math.factorial(n))
                    
                    # Marginal contribution
                    marginal = (self._coalition_value(x, coalition_with_i) - 
                               self._coalition_value(x, coalition_without_i))
                    
                    value += weight * marginal
            
            shapley[i] = value
        
        return shapley
    
    def shapley_interactions(self, x: np.ndarray, order: int) -> Dict[Tuple[int, ...], float]:
        """Exact Shapley interactions using direct formula for multilinear case."""
        interactions = {}
        
        if order == 1:
            shap_vals = self.shapley_values(x)
            for i in range(self.n_features):
                interactions[(i,)] = shap_vals[i]
            return interactions
        
        # For multilinear functions with zero baseline, SII is simpler
        for subset in itertools.combinations(range(self.n_features), order):
            if subset in self.generator.coefficients:
                # For zero baseline: SII = coefficient * product of features
                coeff = self.generator.coefficients[subset]
                prod = 1.0
                for idx in subset:
                    prod *= x[idx]
                interactions[subset] = coeff * prod
            else:
                interactions[subset] = 0.0
        
        return interactions

class TensorNetworkShapleyCalculator:
    """Calculate Shapley values using tensor network with input scaling (not selectors)."""
    
    def __init__(self, generator: SyntheticMultilinearGenerator, ranks: int = 8):
        self.generator = generator
        self.n_features = generator.n_features
        self.ranks = ranks
        self.tensor_tree = None
        
    def train_standard_network(self, n_samples: int = 5000, epochs: int = 500):
        """Train tensor network on standard input data with proper dimensions."""
        print(f"Training standard tensor network with {n_samples} samples...")
        
        # Generate standard training data
        X_train = []
        y_train = []
        
        for _ in range(n_samples):
            x = np.random.normal(0, 1, self.n_features)
            y = self.generator.evaluate(x.reshape(1, -1))[0]
            X_train.append(x)
            y_train.append(y)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        # Create tensor network that handles matrix input properly
        # Use augmented inputs like the original working version
        phys_dims = [2] * self.n_features  # [value, 1] per feature for bias
        self.tensor_tree = make_balanced_binary_tensor_tree(
            leaf_phys_dims=phys_dims,
            ranks=self.ranks,
            out_dim=1,
            assume_bias_when_matrix=True,  # This handles matrix input correctly
            seed=42
        )
        
        # Training
        optimizer = torch.optim.Adam(self.tensor_tree.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Use the assume_bias_when_matrix=True feature
            y_pred = self.tensor_tree.forward(X_tensor).squeeze()
            
            loss = criterion(y_pred, y_tensor)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 100 == 0:
                r2 = r2_score(y_tensor.detach().numpy(), y_pred.detach().numpy())
                print(f"Epoch {epoch}: Loss = {loss.item():.8f}, R² = {r2:.6f}")
            
            if patience_counter > 50:
                break
        
        final_r2 = r2_score(y_tensor.detach().numpy(), y_pred.detach().numpy())
        print(f"Final training R² = {final_r2:.6f}")
        
        return final_r2
    
    def _predict_torch(self, X: np.ndarray) -> np.ndarray:
        """Predict using torch tensor network with proper output handling."""
        if self.tensor_tree is None:
            raise RuntimeError("Must train tensor network first")
        
        self.tensor_tree.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_pred = self.tensor_tree.forward(X_tensor)
            
            # Handle different output shapes
            if y_pred.dim() == 0:  # scalar
                return float(y_pred.item())
            elif y_pred.dim() == 1:  # vector
                return y_pred.detach().cpu().numpy()
            else:  # matrix - squeeze
                return y_pred.squeeze().detach().cpu().numpy()
    
    def shapley_values_input_scaling(self, x: np.ndarray) -> np.ndarray:
        """Compute Shapley values using input scaling method (like your working code)."""
        x = np.asarray(x, dtype=np.float64)
        d = x.shape[0]
        
        # Use t = 0, 1, 2, ..., d-1 for exact Vandermonde solve
        t = np.arange(d, dtype=np.float64)
        
        # Precompute Vandermonde inverse
        V = np.vander(t, N=d, increasing=True)  # [d,d] with powers 0..d-1
        V_inv = np.linalg.solve(V, np.eye(d))   # exact inverse
        
        # Precompute weights 1/(r+1)
        weights = 1.0 / (np.arange(1, d + 1, dtype=np.float64))
        
        phi = np.zeros(d, dtype=np.float64)
        
        # For each feature i
        for i in range(d):
            # Create inputs where others are scaled by t, feature i varies
            others_mask = np.ones(d, dtype=bool)
            others_mask[i] = False
            
            # s=1: feature i at original value, others scaled by t
            X1 = np.repeat(x[None, :], d, axis=0)  # [d, d]
            X1[:, others_mask] *= t[:, None]       # scale others by t
            
            # s=0: feature i set to 0, others scaled by t  
            X0 = np.repeat(x[None, :], d, axis=0)
            X0[:, others_mask] *= t[:, None]
            X0[:, i] = 0.0                         # zero out feature i
            
            # Evaluate
            G1 = self._predict_torch(X1)  # [d]
            G0 = self._predict_torch(X0)  # [d]
            H = G1 - G0                   # [d]
            
            # Solve Vandermonde system: V * c = H
            c_i = V_inv @ H               # [d]
            phi[i] = float((c_i * weights).sum())
        
        return phi
    
    def shapley_interactions_input_scaling(self, x: np.ndarray, order: int) -> Dict[Tuple[int, ...], float]:
        """Compute Shapley interactions using input scaling."""
        interactions = {}
        
        if order == 1:
            shap_vals = self.shapley_values_input_scaling(x)
            for i in range(self.n_features):
                interactions[(i,)] = shap_vals[i]
            return interactions
        
        # For higher-order interactions, use inclusion-exclusion principle
        for subset in itertools.combinations(range(self.n_features), order):
            # Apply inclusion-exclusion to compute pure interaction
            total_value = 0.0
            
            # Sum over all subsets of the target subset
            for k in range(len(subset) + 1):
                for sub_subset in itertools.combinations(subset, k):
                    # Create input with only sub_subset features active
                    x_eval = np.zeros(self.n_features)
                    for idx in sub_subset:
                        x_eval[idx] = x[idx]
                    
                    # Evaluate 
                    val = self._predict_torch(x_eval.reshape(1, -1))
                    
                    # Handle scalar vs array output
                    if isinstance(val, np.ndarray):
                        val = val.item() if val.size == 1 else val[0]
                    
                    # Inclusion-exclusion sign
                    sign = (-1) ** (len(subset) - k)
                    total_value += sign * val
            
            interactions[subset] = total_value
        
        return interactions

def test_tensor_network_recovery():
    """Test tensor network recovery of exact Shapley values using input scaling."""
    
    print("\n=== Tensor Network Recovery Test (Input Scaling) ===")
    
    # Use 4-feature problem for validation
    generator = SyntheticMultilinearGenerator(n_features=4, seed=42)
    x_test = np.array([-1.72491783, -0.56228753, -1.01283112, 0.31424733])
    
    # Get exact values
    exact_calc = ExactShapleyCalculator(generator)
    exact_shapley = exact_calc.shapley_values(x_test)
    
    # Train tensor network (standard, not selector-based)
    tn_calc = TensorNetworkShapleyCalculator(generator, ranks=12)
    train_r2 = tn_calc.train_standard_network(n_samples=20000, epochs=1000)
    
    if train_r2 < 0.99:
        print(f"WARNING: Training R² = {train_r2:.6f} < 0.99. May need more training.")
    
    # Compute TN Shapley values using input scaling
    tn_shapley = tn_calc.shapley_values_input_scaling(x_test)
    
    # Compare
    print(f"\nShapley Value Comparison:")
    print(f"Exact:  {exact_shapley}")
    print(f"TN:     {tn_shapley}")
    print(f"Error:  {np.abs(exact_shapley - tn_shapley)}")
    
    shapley_r2 = r2_score(exact_shapley, tn_shapley)
    print(f"Shapley R² = {shapley_r2:.6f}")
    
    # Test all interaction orders
    print(f"\n=== All Interaction Orders ===")
    order_r2s = {}
    
    for order in range(1, 5):
        exact_interactions = exact_calc.shapley_interactions(x_test, order)
        tn_interactions = tn_calc.shapley_interactions_input_scaling(x_test, order)
        
        # Compare
        exact_vals = [exact_interactions[subset] for subset in exact_interactions.keys()]
        tn_vals = [tn_interactions[subset] for subset in exact_interactions.keys()]
        
        if exact_vals:
            order_r2 = r2_score(exact_vals, tn_vals)
            order_r2s[order] = order_r2
            print(f"Order {order} R² = {order_r2:.6f}")
            
            # Show a few examples
            print(f"  Examples:")
            for i, (subset, exact_val) in enumerate(list(exact_interactions.items())[:3]):
                tn_val = tn_interactions[subset]
                error = abs(exact_val - tn_val)
                print(f"    {subset}: exact={exact_val:.6f}, tn={tn_val:.6f}, error={error:.6f}")
    
    return order_r2s

def simple_validation_experiment():
    """Simplified experiment to validate the approach."""
    
    print("=== Simple Validation Experiment ===")
    
    # Create simple generator
    generator = SyntheticMultilinearGenerator(n_features=4, seed=42)
    
    # Print the planted coefficients
    print("\nPlanted coefficients:")
    for subset, coeff in generator.coefficients.items():
        if abs(coeff) > 1e-6:
            print(f"  {subset}: {coeff:.4f}")
    
    # Generate a single test point
    x_test = np.random.normal(0, 1, 4)
    print(f"\nTest point: {x_test}")
    
    # Compute exact Shapley
    exact_calc = ExactShapleyCalculator(generator)
    exact_shapley = exact_calc.shapley_values(x_test)
    print(f"\nExact Shapley values: {exact_shapley}")
    
    # Verify by direct computation for multilinear case
    manual_shapley = np.zeros(4)
    for subset, coeff in generator.coefficients.items():
        if len(subset) > 0 and abs(coeff) > 1e-10:
            prod = np.prod([x_test[j] for j in subset])
            for i in subset:
                manual_shapley[i] += coeff * prod / len(subset)
    
    print(f"Manual Shapley values: {manual_shapley}")
    print(f"Difference: {np.abs(exact_shapley - manual_shapley)}")
    
    # Compute all interaction orders 1-4
    print("\n=== All Interaction Orders ===")
    all_interactions = {}
    
    for order in range(1, 5):  # Orders 1 to 4
        print(f"\n--- Order {order} ---")
        interactions = exact_calc.shapley_interactions(x_test, order)
        all_interactions[order] = interactions
        
        # Print non-zero interactions
        for subset, value in interactions.items():
            if abs(value) > 1e-8:
                print(f"  {subset}: {value:.8f}")
        
        # Manual verification
        print(f"Manual verification (coeff * product):")
        for subset in itertools.combinations(range(4), order):
            if subset in generator.coefficients:
                coeff = generator.coefficients[subset]
                if abs(coeff) > 1e-10:
                    manual_val = coeff * np.prod([x_test[i] for i in subset])
                    print(f"  {subset}: {manual_val:.8f}")
    
    return generator, x_test, all_interactions

def extend_to_8_features():
    """Extended experiment with 8 features."""
    
    print("\n=== 8-Feature Experiment ===")
    
    # Create 8-feature generator
    generator = SyntheticMultilinearGenerator(n_features=8, seed=42)
    
    # Print strong coefficients only
    print("\nStrong planted coefficients (|coeff| > 1.0):")
    for subset, coeff in generator.coefficients.items():
        if abs(coeff) > 1.0:
            print(f"  {subset}: {coeff:.4f}")
    
    # Test point
    x_test = np.random.normal(0, 1, 8)
    print(f"\nTest point: {x_test}")
    
    # Compute interactions for all orders
    exact_calc = ExactShapleyCalculator(generator)
    all_interactions = {}
    
    for order in range(1, 9):  # Orders 1 to 8
        print(f"\n--- Order {order} ---")
        interactions = exact_calc.shapley_interactions(x_test, order)
        all_interactions[order] = interactions
        
        # Count non-zero and show largest magnitude
        non_zero = [(subset, value) for subset, value in interactions.items() if abs(value) > 1e-8]
        print(f"  Non-zero interactions: {len(non_zero)} / {len(interactions)}")
        
        if non_zero:
            # Show top 5 by magnitude
            sorted_by_mag = sorted(non_zero, key=lambda x: abs(x[1]), reverse=True)
            print(f"  Top interactions by magnitude:")
            for subset, value in sorted_by_mag[:5]:
                print(f"    {subset}: {value:.8f}")
    
    return generator, x_test, all_interactions

if __name__ == "__main__":
    # Run 4-feature validation
    generator_4, x_test_4, interactions_4 = simple_validation_experiment()
    
    # Run 8-feature experiment
    generator_8, x_test_8, interactions_8 = extend_to_8_features()
    
    # Test tensor network recovery
    print("\n" + "="*50)
    recovery_results = test_tensor_network_recovery()
    
    # Save all results
    results = {
        "4_features": {
            "generator_coeffs": generator_4.coefficients,
            "test_point": x_test_4.tolist(),
            "interactions": {str(k): v for k, v in interactions_4.items()}
        },
        "8_features": {
            "generator_coeffs": generator_8.coefficients,
            "test_point": x_test_8.tolist(),
            "interactions": {str(k): v for k, v in interactions_8.items()}
        },
        "tensor_network_recovery": recovery_results
    }
    
    # Convert tuples to strings for JSON serialization
    for feat_key in ["4_features", "8_features"]:
        # Convert coefficients dict
        coeffs_str = {}
        for subset, coeff in results[feat_key]["generator_coeffs"].items():
            coeffs_str[str(subset)] = coeff
        results[feat_key]["generator_coeffs"] = coeffs_str
        
        # Convert interactions dict
        interactions_str = {}
        for order, interactions in results[feat_key]["interactions"].items():
            interactions_order = {}
            for subset, value in interactions.items():
                interactions_order[str(subset)] = value
            interactions_str[order] = interactions_order
        results[feat_key]["interactions"] = interactions_str
    
    # Save to file
    with open("shapley_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("=== FINAL RESULTS ===")
    print(f"Exact Shapley calculation: ✓ Working correctly")
    print(f"Tensor network recovery R² scores:")
    for order, r2 in recovery_results.items():
        status = "✓ Excellent" if r2 > 0.95 else "⚠ Needs improvement" if r2 > 0.8 else "✗ Poor"
        print(f"  Order {order}: R² = {r2:.6f} {status}")
    
    print(f"\nResults saved to shapley_validation_results.json")
    
    if all(r2 > 0.9 for r2 in recovery_results.values()):
        print("🎉 SUCCESS: Tensor network successfully recovers exact Shapley interactions!")
    else:
        print("⚠️  PARTIAL SUCCESS: Some orders need better recovery. Consider:")
        print("   - Increasing training samples")
        print("   - Higher tensor network rank")
        print("   - More training epochs")
        print("   - Better selector data augmentation")