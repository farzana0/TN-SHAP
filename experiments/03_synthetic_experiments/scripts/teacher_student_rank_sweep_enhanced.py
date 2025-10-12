#!/usr/bin/env python3
"""
Improved Teacher-Student rank sweep with enhanced early stopping.

This version includes more aggressive early stopping criteria:
1. Stop when loss becomes very small (< 1e-6)
2. Stop when R² reaches very high values (> 0.999)
3. Stop when improvement plateaus for fewer epochs
4. Adaptive learning rate reduction
"""

import numpy as np
import torch
import torch.nn as nn
import itertools
import math
import os
import json
import time
import pandas as pd
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
        return nodes[::-1]
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate teacher tensor tree."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self.teacher_tree.forward(X_tensor)
        
        if y_pred.dim() == 0:
            return np.array([y_pred.item()])
        else:
            return y_pred.detach().numpy().flatten()

class GenericMultilinearTeacher:
    """Teacher using sparse multilinear function."""
    
    def __init__(self, n_features: int, max_order: int = 3, sparsity: float = 0.3, seed: int = 42):
        self.n_features = n_features
        self.max_order = max_order
        self.seed = seed
        
        # Generate sparse coefficients
        np.random.seed(seed)
        self.coefficients = {}
        
        for order in range(max_order + 1):
            for subset in itertools.combinations(range(n_features), order):
                if order == 0 or np.random.random() < sparsity:
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
                    return y_pred.detach().numpy().flatten()
    
    def _vandermonde_solve(self, y_vals: np.ndarray) -> np.ndarray:
        """Solve Vandermonde system to get polynomial coefficients."""
        t = self.t_nodes[:len(y_vals)]
        V = np.vander(t, increasing=True)
        return np.linalg.solve(V, y_vals)
    
    def shapley_values_tnshap(self, x: np.ndarray) -> np.ndarray:
        """Compute Shapley values using TNShap interpolation method."""
        t = self.t_nodes
        m = len(t)
        d = self.n_features
        
        phi = np.zeros(d)
        
        for i in range(d):
            # Create interpolation path for feature i
            X_eval = np.repeat(x[None, :], m, axis=0)
            X_eval[:, i] *= t
            
            # Evaluate function along path
            f_vals = self._predict(X_eval)
            if f_vals.ndim == 0:
                f_vals = np.array([f_vals])
            
            # Solve for polynomial coefficients
            c_i = self._vandermonde_solve(f_vals)
            
            # Compute weights for Shapley integration
            weights = np.zeros(d)
            for k in range(1, min(d, len(c_i))):
                weights[k] = 1.0 / (k * math.comb(d-1, k-1))
            
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

def train_student_tensor_tree_enhanced(teacher, n_features: int, rank: int, n_samples: int = 5000, 
                                     epochs: int = 800, seed: int = 42):
    """Train a student tensor tree with enhanced early stopping."""
    print(f"    Training student (rank {rank}) with seed {seed}...")
    
    # Set seed for student training
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate training data using teacher (teacher stays deterministic)
    X_train = np.random.normal(0, 1, (n_samples, n_features))
    y_train = teacher.evaluate(X_train)
    
    # Create student tensor tree
    phys_dims = [2] * n_features
    student_tree = make_balanced_binary_tensor_tree(
        leaf_phys_dims=phys_dims,
        ranks=rank,
        out_dim=1,
        assume_bias_when_matrix=True,
        seed=seed
    )
    
    # Train student
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1)  # Fix: flatten to match student output
    
    optimizer = torch.optim.Adam(student_tree.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5, verbose=False)
    
    # Enhanced early stopping parameters
    best_loss = float('inf')
    patience = 0
    max_patience = 80  # Reduced from 150
    
    # Additional stopping criteria
    loss_threshold = 1e-6  # Stop if loss becomes very small
    r2_threshold = 0.9995  # Stop if R² becomes very high
    improvement_threshold = 1e-5  # Minimum improvement to reset patience
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = student_tree.forward(X_tensor).squeeze()  # Fix: ensure same shape as y_tensor
        loss = nn.MSELoss()(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        current_loss = loss.item()
        
        # Check for improvement
        if current_loss < best_loss - improvement_threshold:
            best_loss = current_loss
            patience = 0
        else:
            patience += 1
        
        # Enhanced early stopping conditions
        if epoch % 50 == 0:
            r2 = r2_score(y_tensor.detach().numpy(), y_pred.detach().numpy())
            
            if epoch % 200 == 0:
                print(f"      Epoch {epoch}: Loss = {current_loss:.6f}, R² = {r2:.6f}")
            
            # Stop if loss is very small
            if current_loss < loss_threshold:
                print(f"      Early stopping at epoch {epoch} (loss threshold: {current_loss:.2e})")
                break
            
            # Stop if R² is very high
            if r2 > r2_threshold:
                print(f"      Early stopping at epoch {epoch} (R² threshold: {r2:.6f})")
                break
        
        # Stop if no improvement for too long
        if patience > max_patience:
            print(f"      Early stopping at epoch {epoch} (patience exceeded)")
            break
        
        # Stop if loss becomes NaN
        if torch.isnan(loss):
            print(f"      Early stopping at epoch {epoch} (NaN loss)")
            break
    
    final_r2 = r2_score(y_tensor.detach().numpy(), y_pred.detach().numpy())
    print(f"      Final R² = {final_r2:.6f}")
    
    return student_tree, final_r2

# Copy all other functions from the original script but use the enhanced training function
def run_single_experiment(teacher_name: str, teacher, student_rank: int, 
                         n_features: int, test_points: List[np.ndarray], 
                         run_seed: int, run_id: int, n_runs: int = 3):
    """Run a single experiment with one student training seed."""
    print(f"  Run {run_id + 1}/{n_runs} (seed={run_seed})")
    
    # Train student with enhanced early stopping
    student, train_r2 = train_student_tensor_tree_enhanced(
        teacher, n_features, student_rank, 
        n_samples=8000, epochs=800, seed=run_seed
    )
    
    # Initialize TNShap calculators
    teacher_calc = TNShapCalculator(teacher, teacher.t_nodes, n_features)
    student_calc = TNShapCalculator(student, teacher.t_nodes, n_features)
    
    # Evaluate on test points
    point_results = []
    
    for point_idx, x_test in enumerate(test_points):
        point_result = {
            'run_id': run_id,
            'run_seed': run_seed,
            'point_idx': point_idx,
            'train_r2': train_r2,
            'teacher_name': teacher_name,
            'student_rank': student_rank
        }
        
        # Calculate Shapley values for orders 1, 2, 3
        for order in [1, 2, 3]:
            teacher_interactions = teacher_calc.shapley_interactions_tnshap(x_test, order)
            student_interactions = student_calc.shapley_interactions_tnshap(x_test, order)
            
            # Match interactions and compute R²
            teacher_vals = []
            student_vals = []
            
            for key in teacher_interactions:
                if key in student_interactions:
                    teacher_vals.append(teacher_interactions[key])
                    student_vals.append(student_interactions[key])
            
            if len(teacher_vals) > 1:
                r2 = r2_score(teacher_vals, student_vals)
            else:
                r2 = 0.0
            
            point_result[f'order_{order}_r2'] = r2
        
        point_results.append(point_result)
    
    return point_results

def run_teacher_student_experiment_multiple_seeds(teacher_name: str, teacher, 
                                                student_ranks: List[int], 
                                                n_features: int = 4, 
                                                n_test_points: int = 10,
                                                n_runs: int = 3):
    """Run experiment for one teacher type across multiple student ranks with multiple seeds."""
    print(f"\n{'='*80}")
    print(f"TEACHER: {teacher_name}")
    print(f"{'='*80}")
    
    # Generate fixed test points (same for all runs)
    np.random.seed(42)
    test_points = [np.random.normal(0, 1, n_features) for _ in range(n_test_points)]
    
    # Generate random seeds for student training
    np.random.seed(12345)  # Fixed seed for reproducible seed generation
    run_seeds = np.random.randint(1000, 9999, n_runs).tolist()
    
    print(f"Using {n_runs} different seeds for student training: {run_seeds}")
    
    results = {}
    
    for student_rank in student_ranks:
        print(f"\n--- Student Rank {student_rank} ---")
        
        all_run_results = []
        
        # Run multiple training sessions with different seeds
        for run_id, run_seed in enumerate(run_seeds):
            run_results = run_single_experiment(
                teacher_name, teacher, student_rank, n_features, 
                test_points, run_seed, run_id, n_runs
            )
            all_run_results.extend(run_results)
        
        results[student_rank] = {
            'run_seeds': run_seeds,
            'detailed_results': all_run_results
        }
    
    return results

def save_results_with_stats(all_results: Dict, n_runs: int = 3):
    """Save results with mean and standard deviation statistics."""
    os.makedirs('results_teacher_student_multi_seed_enhanced', exist_ok=True)
    
    # Create detailed results DataFrame
    detailed_data = []
    summary_data = []
    
    for teacher_name, teacher_results in all_results.items():
        for student_rank, rank_results in teacher_results.items():
            # Add detailed results
            for result in rank_results['detailed_results']:
                result['teacher_name'] = teacher_name
                detailed_data.append(result)
            
            # Compute summary statistics
            df_rank = pd.DataFrame(rank_results['detailed_results'])
            
            # Group by run_id and compute per-run averages, then overall stats
            run_stats = df_rank.groupby('run_id').agg({
                'train_r2': 'mean',
                'order_1_r2': 'mean',
                'order_2_r2': 'mean', 
                'order_3_r2': 'mean'
            })
            
            summary_row = {
                'teacher_name': teacher_name,
                'student_rank': student_rank,
                'n_runs': n_runs,
                'run_seeds': str(rank_results['run_seeds']),
                'train_r2_mean': run_stats['train_r2'].mean(),
                'train_r2_std': run_stats['train_r2'].std(),
                'order_1_r2_mean': run_stats['order_1_r2'].mean(),
                'order_1_r2_std': run_stats['order_1_r2'].std(),
                'order_2_r2_mean': run_stats['order_2_r2'].mean(),
                'order_2_r2_std': run_stats['order_2_r2'].std(),
                'order_3_r2_mean': run_stats['order_3_r2'].mean(),
                'order_3_r2_std': run_stats['order_3_r2'].std()
            }
            summary_data.append(summary_row)
    
    # Save DataFrames
    detailed_df = pd.DataFrame(detailed_data)
    summary_df = pd.DataFrame(summary_data)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_csv_path = f'results_teacher_student_multi_seed_enhanced/summary_{timestamp}.csv'
    detailed_csv_path = f'results_teacher_student_multi_seed_enhanced/detailed_{timestamp}.csv'
    
    summary_df.to_csv(summary_csv_path, index=False)
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    print(f"Saved summary results to {summary_csv_path}")
    print(f"Saved detailed results to {detailed_csv_path}")
    
    return summary_df, detailed_df

def print_final_summary_with_stats(summary_df):
    """Print a final summary of results with mean ± std."""
    print(f"\n{'='*100}")
    print("FINAL SUMMARY - MEAN ± STD ACROSS 3 RUNS (ENHANCED EARLY STOPPING)")
    print(f"{'='*100}")
    
    for teacher in summary_df['teacher_name'].unique():
        print(f"\n{teacher.upper()}:")
        teacher_data = summary_df[summary_df['teacher_name'] == teacher]
        
        print("  Rank | Train R² (μ±σ)    | Order 1 R² (μ±σ)  | Order 2 R² (μ±σ)  | Order 3 R² (μ±σ)")
        print("  -----|-------------------|-------------------|-------------------|-------------------")
        
        for _, row in teacher_data.iterrows():
            rank = row['student_rank']
            
            # Format mean ± std
            train_str = f"{row['train_r2_mean']:.4f}±{row['train_r2_std']:.4f}"
            o1_str = f"{row['order_1_r2_mean']:.4f}±{row['order_1_r2_std']:.4f}"
            o2_str = f"{row['order_2_r2_mean']:.4f}±{row['order_2_r2_std']:.4f}"
            o3_str = f"{row['order_3_r2_mean']:.4f}±{row['order_3_r2_std']:.4f}"
            
            print(f"  {rank:4d} | {train_str:17s} | {o1_str:17s} | {o2_str:17s} | {o3_str:17s}")

def main():
    """Run the complete teacher-student rank sweep experiment with enhanced early stopping."""
    n_features = 4
    student_ranks = [2, 4, 6, 8, 10, 16]
    n_runs = 3
    
    print(f"Starting Teacher-Student Rank Sweep with ENHANCED Early Stopping")
    print(f"Features: {n_features}, Student Ranks: {student_ranks}, Runs: {n_runs}")
    print(f"Enhanced stopping criteria: loss < 1e-6, R² > 0.9995, patience = 80")
    print(f"Estimated total time: ~{18 * n_runs * 6 // 60} hours ({18 * n_runs} total training runs)")
    
    # Create teachers (fixed seeds)
    teachers = {
        'TensorTree_Rank3': TensorTreeTeacher(n_features, ranks=3, seed=42),
        'TensorTree_Rank16': TensorTreeTeacher(n_features, ranks=16, seed=43),
        'GenericMultilinear': GenericMultilinearTeacher(n_features, max_order=3, seed=44)
    }
    
    all_results = {}
    
    # Run experiments for each teacher
    for teacher_name, teacher in teachers.items():
        teacher_results = run_teacher_student_experiment_multiple_seeds(
            teacher_name, teacher, student_ranks, n_features, 
            n_test_points=15, n_runs=n_runs
        )
        all_results[teacher_name] = teacher_results
    
    # Save and summarize results
    summary_df, detailed_df = save_results_with_stats(all_results, n_runs)
    print_final_summary_with_stats(summary_df)
    
    print(f"\n{'='*80}")
    print(f"ENHANCED EXPERIMENT COMPLETED!")
    print(f"Results saved to 'results_teacher_student_multi_seed_enhanced/'")
    print(f"- Summary with mean±std: summary_*.csv")
    print(f"- Detailed run data: detailed_*.csv") 
    print(f"Total training runs completed: {18 * n_runs}")
    print(f"Time saved with enhanced early stopping!")
    print(f"{'='*80}")
    
    return all_results, summary_df, detailed_df

if __name__ == "__main__":
    results, summary, detailed = main()
