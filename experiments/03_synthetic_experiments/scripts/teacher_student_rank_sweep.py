#!/usr/bin/env python3
"""
Teacher-Student rank sweep experiment with TNShap Shapley calculations.

Three teacher types:
1. TensorTree rank 3 (low-rank)
2. TensorTree rank 16 (higher-rank) 
3. Generic multilinear function

Student ranks: 2, 4, 6, 8, 10, 16

Uses TNShap method for Shapley value and interaction calculations up to order 3.
Saves interpolation points during training for reuse in evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import itertools
import math
import os
import json
import time
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score
from tntree_model import BinaryTensorTree, make_balanced_binary_tensor_tree

class TensorTreeTeacher:
    """Teacher using tensor tree with specified rank."""
    
    def __init__(self, n_features: int, ranks: int, seed: int = 42):
        self.n_features = n_features
        self.ranks = ranks
        self.seed = seed
        
        # Create teacher tensor tree
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        phys_dims = [2] * n_features  # Binary selectors
        self.teacher_tree = make_balanced_binary_tensor_tree(
            leaf_phys_dims=phys_dims,
            ranks=ranks,
            out_dim=1,
            assume_bias_when_matrix=True,
            seed=seed
        )
        
        # Store interpolation points for TNShap (Chebyshev nodes)
        self.t_nodes = self._generate_chebyshev_nodes(n_features + 8)  # Extra nodes for stability
        
    def _generate_chebyshev_nodes(self, m: int) -> np.ndarray:
        """Generate Chebyshev nodes in [0,1]."""
        i = np.arange(1, m + 1)
        nodes = 0.5 * (1 + np.cos((2 * i - 1) * np.pi / (2 * m)))
        return nodes[::-1]  # Sort ascending
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate teacher tensor tree."""
        self.teacher_tree.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y = self.teacher_tree.forward(X_tensor)
            
            if y.dim() == 0:
                return np.array([y.item()])
            else:
                result = y.squeeze().detach().cpu().numpy()
                if result.ndim == 0:
                    return np.array([result.item()])
                return result

class GenericMultilinearTeacher:
    """Teacher using explicit multilinear function."""
    
    def __init__(self, n_features: int, max_order: int = 3, seed: int = 42):
        self.n_features = n_features
        self.max_order = max_order
        self.seed = seed
        np.random.seed(seed)
        
        # Generate coefficients for subsets up to max_order
        self.coefficients = {}
        
        # Constant term
        self.coefficients[()] = 0.0
        
        # Linear terms
        for i in range(n_features):
            self.coefficients[(i,)] = np.random.normal(0, 1.0)
        
        # Pairwise interactions
        for i in range(n_features):
            for j in range(i+1, n_features):
                self.coefficients[(i, j)] = np.random.normal(0, 1.5)
        
        # Triplet interactions
        if max_order >= 3:
            for subset in itertools.combinations(range(n_features), 3):
                self.coefficients[subset] = np.random.normal(0, 1.2)
        
        # Store interpolation points for TNShap
        self.t_nodes = self._generate_chebyshev_nodes(n_features + 8)
        
    def _generate_chebyshev_nodes(self, m: int) -> np.ndarray:
        """Generate Chebyshev nodes in [0,1]."""
        i = np.arange(1, m + 1)
        nodes = 0.5 * (1 + np.cos((2 * i - 1) * np.pi / (2 * m)))
        return nodes[::-1]
    
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

class TNShapCalculator:
    """Calculate Shapley values using TNShap (interpolation-based) method."""
    
    def __init__(self, model, t_nodes: np.ndarray, n_features: int):
        self.model = model
        self.t_nodes = t_nodes
        self.n_features = n_features
        
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model."""
        if hasattr(self.model, 'evaluate'):
            return self.model.evaluate(X)
        else:
            # Assume it's a PyTorch model
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_pred = self.model.forward(X_tensor)
                
                if y_pred.dim() == 0:
                    return np.array([y_pred.item()])
                else:
                    result = y_pred.squeeze().detach().cpu().numpy()
                    if result.ndim == 0:
                        return np.array([result.item()])
                    return result
    
    def _vandermonde_solve(self, f_values: np.ndarray) -> np.ndarray:
        """Solve Vandermonde system to get polynomial coefficients."""
        t = self.t_nodes
        m = len(t)
        V = np.vander(t, N=m, increasing=True)
        return np.linalg.solve(V, f_values)
    
    def shapley_values_tnshap(self, x: np.ndarray) -> np.ndarray:
        """Compute Shapley values using TNShap interpolation method."""
        x = np.asarray(x, dtype=np.float64)
        d = x.shape[0]
        t = self.t_nodes
        m = len(t)
        
        # Compute weights for integration
        weights = 1.0 / (np.arange(1, d + 1, dtype=np.float64))
        
        phi = np.zeros(d, dtype=np.float64)
        
        for i in range(d):
            others_mask = np.ones(d, dtype=bool)
            others_mask[i] = False
            
            # Evaluate along interpolation path
            # s=1: feature i unchanged, others scaled by t
            X1 = np.repeat(x[None, :], m, axis=0)
            X1[:, others_mask] *= t[:, None]
            
            # s=0: feature i zeroed, others scaled by t
            X0 = np.repeat(x[None, :], m, axis=0)
            X0[:, others_mask] *= t[:, None]
            X0[:, i] = 0.0
            
            # Get function values
            G1 = self._predict(X1)
            G0 = self._predict(X0)
            H = G1 - G0
            
            if H.ndim == 0:
                H = np.array([H])
            
            # Solve for coefficients
            c_i = self._vandermonde_solve(H)
            
            # Integrate (sum weighted coefficients)
            phi[i] = float((c_i[:d] * weights).sum())
        
        return phi
    
    def shapley_interactions_tnshap(self, x: np.ndarray, order: int) -> Dict[Tuple[int, ...], float]:
        """Compute Shapley interactions using TNShap method."""
        interactions = {}
        
        if order == 1:
            shap_vals = self.shapley_values_tnshap(x)
            for i in range(self.n_features):
                interactions[(i,)] = shap_vals[i]
            return interactions
        
        # For higher orders, use inclusion-exclusion on interpolated functions
        t = self.t_nodes
        m = len(t)
        
        for subset in itertools.combinations(range(self.n_features), order):
            # Compute interaction via inclusion-exclusion
            total_value = 0.0
            
            for k in range(len(subset) + 1):
                for sub_subset in itertools.combinations(subset, k):
                    # Evaluate function with this sub_subset active
                    X_eval = np.repeat(x[None, :], m, axis=0)
                    # Zero out features not in sub_subset
                    for j in range(self.n_features):
                        if j not in sub_subset:
                            X_eval[:, j] = 0.0
                    
                    # Scale by interpolation parameter
                    X_eval *= t[:, None]
                    
                    # Get function values and solve for coefficients
                    f_vals = self._predict(X_eval)
                    if f_vals.ndim == 0:
                        f_vals = np.array([f_vals])
                    
                    coeffs = self._vandermonde_solve(f_vals)
                    
                    # The interaction is the highest-order coefficient
                    if len(coeffs) > order:
                        val = coeffs[order]
                    else:
                        val = 0.0
                    
                    sign = (-1) ** (len(subset) - k)
                    total_value += sign * val
            
            interactions[subset] = total_value
        
        return interactions

def train_student_tensor_tree(teacher, n_features: int, student_rank: int, 
                             n_samples: int = 10000, epochs: int = 1000, 
                             seed: int = 42):
    """Train student tensor tree to approximate teacher."""
    print(f"  Training student (rank={student_rank}) with {n_samples} samples...")
    
    # Generate training data
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train = np.random.normal(0, 1, (n_samples, n_features))
    y_train = teacher.evaluate(X_train)
    
    # Create student tensor tree
    phys_dims = [2] * n_features
    student_tree = make_balanced_binary_tensor_tree(
        leaf_phys_dims=phys_dims,
        ranks=student_rank,
        out_dim=1,
        assume_bias_when_matrix=True,
        seed=seed + 1000
    )
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(student_tree.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience = 0
    max_patience = 100
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        y_pred = student_tree.forward(X_tensor).squeeze()
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
        
        if epoch % 200 == 0:
            r2 = r2_score(y_tensor.detach().numpy(), y_pred.detach().numpy())
            print(f"    Epoch {epoch}: Loss = {loss.item():.6f}, R² = {r2:.6f}")
        
        if patience > max_patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    final_r2 = r2_score(y_tensor.detach().numpy(), y_pred.detach().numpy())
    print(f"    Final R² = {final_r2:.6f}")
    
    return student_tree, final_r2

def run_teacher_student_experiment(teacher_name: str, teacher, student_ranks: List[int], 
                                 n_features: int = 4, n_test_points: int = 10):
    """Run experiment for one teacher type across multiple student ranks."""
    print(f"\n{'='*80}")
    print(f"TEACHER: {teacher_name}")
    print(f"{'='*80}")
    
    results = {}
    
    # Generate test points
    np.random.seed(42)
    test_points = [np.random.normal(0, 1, n_features) for _ in range(n_test_points)]
    
    for student_rank in student_ranks:
        print(f"\n--- Student Rank {student_rank} ---")
        
        # Train student
        student, train_r2 = train_student_tensor_tree(
            teacher, n_features, student_rank, 
            n_samples=8000, epochs=1500, seed=42
        )
        
        # Initialize TNShap calculators
        teacher_calc = TNShapCalculator(teacher, teacher.t_nodes, n_features)
        student_calc = TNShapCalculator(student, teacher.t_nodes, n_features)
        
        # Evaluate on test points
        point_results = []
        
        for point_idx, x_test in enumerate(test_points):
            print(f"  Test point {point_idx + 1}/{len(test_points)}")
            
            point_result = {
                'point_idx': point_idx,
                'train_r2': train_r2,
                'teacher_name': teacher_name,
                'student_rank': student_rank
            }
            
            # Compute Shapley values and interactions for orders 1, 2, 3
            for order in range(1, 4):
                # Teacher (ground truth)
                teacher_interactions = teacher_calc.shapley_interactions_tnshap(x_test, order)
                
                # Student
                student_interactions = student_calc.shapley_interactions_tnshap(x_test, order)
                
                # Compare
                teacher_vals = [teacher_interactions[subset] for subset in teacher_interactions.keys()]
                student_vals = [student_interactions[subset] for subset in teacher_interactions.keys()]
                
                if teacher_vals:
                    r2 = r2_score(teacher_vals, student_vals)
                    mse = np.mean((np.array(teacher_vals) - np.array(student_vals)) ** 2)
                    cosine_sim = np.dot(teacher_vals, student_vals) / (
                        np.linalg.norm(teacher_vals) * np.linalg.norm(student_vals) + 1e-12
                    )
                    
                    point_result[f'order_{order}_r2'] = r2
                    point_result[f'order_{order}_mse'] = mse
                    point_result[f'order_{order}_cosine'] = cosine_sim
                    point_result[f'order_{order}_n_interactions'] = len(teacher_vals)
                    
                    print(f"    Order {order}: R² = {r2:.6f}, MSE = {mse:.6f}, Cosine = {cosine_sim:.6f}")
                else:
                    point_result[f'order_{order}_r2'] = np.nan
                    point_result[f'order_{order}_mse'] = np.nan
                    point_result[f'order_{order}_cosine'] = np.nan
                    point_result[f'order_{order}_n_interactions'] = 0
            
            point_results.append(point_result)
        
        # Aggregate results for this student rank
        rank_result = {
            'teacher_name': teacher_name,
            'student_rank': student_rank,
            'train_r2': train_r2,
            'n_test_points': len(test_points)
        }
        
        # Compute mean and std across test points for each order
        for order in range(1, 4):
            r2_values = [pr[f'order_{order}_r2'] for pr in point_results if not np.isnan(pr[f'order_{order}_r2'])]
            mse_values = [pr[f'order_{order}_mse'] for pr in point_results if not np.isnan(pr[f'order_{order}_mse'])]
            cosine_values = [pr[f'order_{order}_cosine'] for pr in point_results if not np.isnan(pr[f'order_{order}_cosine'])]
            
            if r2_values:
                rank_result[f'order_{order}_r2_mean'] = np.mean(r2_values)
                rank_result[f'order_{order}_r2_std'] = np.std(r2_values)
                rank_result[f'order_{order}_mse_mean'] = np.mean(mse_values)
                rank_result[f'order_{order}_mse_std'] = np.std(mse_values)
                rank_result[f'order_{order}_cosine_mean'] = np.mean(cosine_values)
                rank_result[f'order_{order}_cosine_std'] = np.std(cosine_values)
            else:
                rank_result[f'order_{order}_r2_mean'] = np.nan
                rank_result[f'order_{order}_r2_std'] = np.nan
                rank_result[f'order_{order}_mse_mean'] = np.nan
                rank_result[f'order_{order}_mse_std'] = np.nan
                rank_result[f'order_{order}_cosine_mean'] = np.nan
                rank_result[f'order_{order}_cosine_std'] = np.nan
        
        results[student_rank] = {
            'summary': rank_result,
            'detailed': point_results
        }
        
        # Print summary for this rank
        print(f"  Summary for rank {student_rank}:")
        print(f"    Training R² = {train_r2:.6f}")
        for order in range(1, 4):
            r2_mean = rank_result.get(f'order_{order}_r2_mean', np.nan)
            r2_std = rank_result.get(f'order_{order}_r2_std', np.nan)
            print(f"    Order {order} R² = {r2_mean:.6f} ± {r2_std:.6f}")
    
    return results

def save_results(all_results: Dict, output_dir: str = "results_teacher_student_sweep"):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results as JSON
    json_path = os.path.join(output_dir, "full_results.json")
    with open(json_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy(all_results), f, indent=2)
    
    print(f"Saved full results to {json_path}")
    
    # Create summary CSV
    summary_rows = []
    detailed_rows = []
    
    for teacher_name, teacher_results in all_results.items():
        for student_rank, rank_data in teacher_results.items():
            summary_rows.append(rank_data['summary'])
            
            # Add teacher_name and student_rank to detailed results
            for point_result in rank_data['detailed']:
                detailed_rows.append(point_result)
    
    # Save summary CSV
    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "summary_results.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary to {summary_csv_path}")
    
    # Save detailed CSV
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_csv_path = os.path.join(output_dir, "detailed_results.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"Saved detailed results to {detailed_csv_path}")
    
    return summary_df, detailed_df

def print_final_summary(summary_df):
    """Print a final summary of results."""
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    for teacher in summary_df['teacher_name'].unique():
        print(f"\n{teacher.upper()}:")
        teacher_data = summary_df[summary_df['teacher_name'] == teacher]
        
        print("  Student Rank | Train R² | Order 1 R² | Order 2 R² | Order 3 R²")
        print("  -------------|----------|------------|------------|------------")
        
        for _, row in teacher_data.iterrows():
            rank = row['student_rank']
            train_r2 = row['train_r2']
            o1_r2 = row.get('order_1_r2_mean', np.nan)
            o2_r2 = row.get('order_2_r2_mean', np.nan)
            o3_r2 = row.get('order_3_r2_mean', np.nan)
            
            print(f"  {rank:11d} | {train_r2:8.6f} | {o1_r2:10.6f} | {o2_r2:10.6f} | {o3_r2:10.6f}")

def main():
    """Run the complete teacher-student rank sweep experiment."""
    n_features = 4
    student_ranks = [2, 4, 6, 8, 10, 16]
    
    # Create teachers
    teachers = {
        'TensorTree_Rank3': TensorTreeTeacher(n_features, ranks=3, seed=42),
        'TensorTree_Rank16': TensorTreeTeacher(n_features, ranks=16, seed=43),
        'GenericMultilinear': GenericMultilinearTeacher(n_features, max_order=3, seed=44)
    }
    
    all_results = {}
    
    # Run experiments for each teacher
    for teacher_name, teacher in teachers.items():
        teacher_results = run_teacher_student_experiment(
            teacher_name, teacher, student_ranks, n_features, n_test_points=15
        )
        all_results[teacher_name] = teacher_results
    
    # Save and summarize results
    summary_df, detailed_df = save_results(all_results)
    print_final_summary(summary_df)
    
    print(f"\nExperiment completed! Results saved to 'results_teacher_student_sweep/'")
    
    return all_results, summary_df, detailed_df

if __name__ == "__main__":
    results, summary, detailed = main()
