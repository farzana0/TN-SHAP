#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Baseline Sweep Experiment on Pre-trained Diabetes Models

Loads pre-trained teacher and student models from:
experiments/UCI/out_local_student_singlegrid/diabetes_seed2711_K89_m10/

Runs baseline comparisons with budget sweep: 50,100,200,500,1000,1500,2000,10000
Compares against TNShap (computed once) measuring runtime, MSE, and cosine similarity.
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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load existing models and data loading functions
import sys

try:
    from src.tntree_model import BinaryTensorTree
except ImportError:
    BinaryTensorTree = None

try:
    from src.feature_mapped_tn import FeatureMappedTN
except ImportError:
    FeatureMappedTN = None

class MLPRegressor(nn.Module):
    """MLP Teacher model."""
    def __init__(self, d_in: int, hidden=(256,256,128), pdrop=0.0):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(pdrop)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

def load_diabetes_data(seed: int = 2711):
    """Load and preprocess diabetes dataset with same split as original."""
    ds = load_diabetes()
    X = ds.data.astype(float)
    y = ds.target.astype(float)
    
    # Same preprocessing as original
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed)
    
    sx = StandardScaler().fit(Xtr2)
    sy = StandardScaler().fit(ytr2.reshape(-1,1))
    
    Xte_scaled = sx.transform(Xte).astype(np.float32)
    
    return Xte_scaled, sx, sy

def load_teacher_model(model_path: str, d_hint: int) -> nn.Module:
    """Load teacher model."""
    obj = torch.load(model_path, map_location="cpu")
    if isinstance(obj, nn.Module):
        model = obj
    else:
        sd = obj.get("state_dict", obj)
        model = MLPRegressor(d_in=d_hint)
        model.load_state_dict(sd, strict=False)
    return model.eval()

def load_student_model(model_path: str, d_hint: int) -> nn.Module:
    """Load student TN model."""
    if BinaryTensorTree is None or FeatureMappedTN is None:
        raise RuntimeError("TN models not available")
    
    obj = torch.load(model_path, map_location="cpu")
    
    if isinstance(obj, nn.Module):
        return obj.eval()
    
    sd = obj.get("state_dict", obj)
    
    def _is_wrapper_state_dict(sd):
        return any(k.startswith("feature_map.") for k in sd.keys())
    
    def _strip_prefix(sd, prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    
    def _infer_rank_from_tn_state(sd):
        for k, v in sd.items():
            if k.startswith(("cores.", "tn.cores.")) and isinstance(v, torch.Tensor):
                shp = tuple(v.shape)
                if len(shp) in (2, 3):
                    return int(shp[0])
        return 16
    
    if _is_wrapper_state_dict(sd):
        # FeatureMappedTN
        tn_sd = _strip_prefix(sd, "tn.")
        fmap_sd = _strip_prefix(sd, "feature_map.")
        r = _infer_rank_from_tn_state(sd)
        
        tn_core = BinaryTensorTree(
            [2]*d_hint, ranks=r, out_dim=1,
            assume_bias_when_matrix=True, device="cpu"
        )
        tn_core.load_state_dict(tn_sd, strict=False)
        
        model = FeatureMappedTN(
            tn=tn_core, d_in=d_hint, fmap_hidden=32, fmap_act="relu"
        )
        model.feature_map.load_state_dict(fmap_sd, strict=False)
        return model.eval()
    
    else:
        # Bare TN
        r = _infer_rank_from_tn_state(sd)
        tn = BinaryTensorTree(
            [2]*d_hint, ranks=r, out_dim=1,
            assume_bias_when_matrix=True, device="cpu"
        )
        tn.load_state_dict(sd, strict=False)
        return tn.eval()

class TNShapCalculator:
    """Calculate Shapley values using TNShap (interpolation-based) method."""
    
    def __init__(self, model, t_nodes: np.ndarray, n_features: int):
        self.model = model
        self.t_nodes = t_nodes
        self.n_features = n_features
        
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model."""
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
    
    def shapley_values_tnshap(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute Shapley values using TNShap interpolation method."""
        start_time = time.perf_counter()
        
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
        
        runtime = time.perf_counter() - start_time
        return phi, runtime
    
    def shapley_interactions_tnshap(self, x: np.ndarray, order: int) -> Tuple[Dict[Tuple[int, ...], float], float]:
        """Compute Shapley interactions using TNShap method."""
        start_time = time.perf_counter()
        
        interactions = {}
        
        if order == 1:
            shap_vals, _ = self.shapley_values_tnshap(x)
            for i in range(self.n_features):
                interactions[(i,)] = shap_vals[i]
            runtime = time.perf_counter() - start_time
            return interactions, runtime
        
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
        
        runtime = time.perf_counter() - start_time
        return interactions, runtime

# Baseline methods
def kernel_shap_baseline(model, x: np.ndarray, budget: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """KernelSHAP baseline (order 1 only)."""
    try:
        import shap
        
        def predict_np(X):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                return model.forward(X_tensor).detach().cpu().numpy()
        
        np.random.seed(seed)
        d = len(x)
        explainer = shap.KernelExplainer(predict_np, np.zeros((1, d)))
        
        start_time = time.perf_counter()
        shap_values = explainer.shap_values(x.reshape(1, -1), nsamples=budget)[0]
        runtime = time.perf_counter() - start_time
        
        return np.asarray(shap_values, float), runtime
        
    except Exception as e:
        print(f"KernelSHAP failed: {e}")
        return np.full(len(x), np.nan), np.nan

def shapiq_baseline(model, x: np.ndarray, order: int, budget: int, 
                   approximator: str, index_name: str, seed: int = 42) -> Tuple[np.ndarray, float]:
    """SHAPIQ baseline for any order."""
    try:
        import shapiq
        
        def predict_np(X):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                return model.forward(X_tensor).detach().cpu().numpy()
        
        class NpModel:
            def __init__(self, fn):
                self.fn = fn
            def predict(self, X):
                return self.fn(np.asarray(X, np.float32))
        
        # Generate background data
        np.random.seed(seed)
        X_bg = np.random.normal(0, 1, (100, len(x)))
        
        explainer = shapiq.TabularExplainer(
            model=NpModel(predict_np),
            data=X_bg,
            approximator=approximator,
            index=index_name,
            max_order=order,
            random_state=seed
        )
        
        start_time = time.perf_counter()
        iv = explainer.explain(x.reshape(1, -1), budget=budget)
        runtime = time.perf_counter() - start_time
        
        if order == 1:
            if index_name == "SV":
                vals = np.asarray(iv.get_values(), float).ravel()
                return vals, runtime
            tens = iv.get_n_order_values(1)
            return np.asarray(tens, float).ravel(), runtime
        
        tens = iv.get_n_order_values(order)
        d = tens.shape[0]
        result = np.asarray([tens[tuple(T)] for T in itertools.combinations(range(d), order)], float)
        return result, runtime
        
    except Exception as e:
        print(f"SHAPIQ {approximator}-{index_name} failed: {e}")
        n_interactions = math.comb(len(x), order) if order <= len(x) else 0
        return np.full(n_interactions, np.nan), np.nan

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    
    # Handle NaN values
    valid_mask = ~(np.isnan(a) | np.isnan(b))
    if not valid_mask.any():
        return np.nan
    
    a_valid = a[valid_mask]
    b_valid = b[valid_mask]
    
    norm_a = np.linalg.norm(a_valid)
    norm_b = np.linalg.norm(b_valid)
    
    if norm_a < 1e-12 or norm_b < 1e-12:
        return np.nan
    
    return float(np.dot(a_valid, b_valid) / (norm_a * norm_b))

def run_baseline_sweep_experiment(model_dir: str):
    """Run baseline sweep experiment on pre-trained diabetes models."""
    
    budgets = [50, 100, 200, 500, 1000, 1500, 2000, 10000]
    
    print(f"Loading models from: {model_dir}")
    
    # Load manifest
    with open(os.path.join(model_dir, "manifest.json"), 'r') as f:
        manifest = json.load(f)
    
    n_features = manifest['dim']
    print(f"Features: {n_features}")
    
    # Load models
    teacher_path = os.path.join(model_dir, "teacher.pt")
    student_path = os.path.join(model_dir, "tn.pt")
    t_nodes_path = os.path.join(model_dir, "t_nodes_shared.npy")
    
    teacher = load_teacher_model(teacher_path, n_features)
    student = load_student_model(student_path, n_features)
    t_nodes = np.load(t_nodes_path)
    
    print(f"Loaded teacher and student models")
    print(f"Student training R²: {manifest['tn_info']['final_r2']:.6f}")
    print(f"Interpolation nodes: {len(t_nodes)}")
    
    # Load test data (use same seed as original)
    X_test, sx, sy = load_diabetes_data(seed=manifest['seed'])
    
    # Use subset of test points (same as original K=89)
    np.random.seed(manifest['seed'])
    n_test = min(10, len(X_test))  # Limit for faster execution
    test_indices = np.random.choice(len(X_test), size=n_test, replace=False)
    test_points = X_test[test_indices]
    
    print(f"Using {len(test_points)} test points")
    
    # Initialize TNShap calculators
    teacher_calc = TNShapCalculator(teacher, t_nodes, n_features)
    student_calc = TNShapCalculator(student, t_nodes, n_features)
    
    all_results = []
    
    # Process each test point
    for point_idx, x_test in enumerate(test_points):
        print(f"\nTest point {point_idx + 1}/{len(test_points)}")
        
        # Compute TNShap once (ground truth from teacher, prediction from student)
        tnshap_results = {}
        
        for order in range(1, 4):
            print(f"  Computing TNShap order {order}...")
            teacher_interactions, teacher_time = teacher_calc.shapley_interactions_tnshap(x_test, order)
            student_interactions, student_time = student_calc.shapley_interactions_tnshap(x_test, order)
            
            teacher_vals = [teacher_interactions[subset] for subset in teacher_interactions.keys()]
            student_vals = [student_interactions[subset] for subset in teacher_interactions.keys()]
            
            tnshap_results[order] = {
                'teacher_vals': teacher_vals,
                'student_vals': student_vals,
                'teacher_time': teacher_time,
                'student_time': student_time,
                'mse': np.mean((np.array(teacher_vals) - np.array(student_vals)) ** 2),
                'cosine': cosine_similarity(teacher_vals, student_vals)
            }
            print(f"    TNShap teacher time: {teacher_time:.3f}s, student time: {student_time:.3f}s")
        
        # Test baselines at different budgets
        for budget in budgets:
            print(f"  Budget {budget}")
            
            # KernelSHAP (order 1 only)
            print(f"    KernelSHAP...")
            kernel_vals, kernel_time = kernel_shap_baseline(teacher, x_test, budget, seed=manifest['seed'])
            if not np.isnan(kernel_vals).all():
                kernel_mse = np.mean((np.array(tnshap_results[1]['teacher_vals']) - kernel_vals) ** 2)
                kernel_cosine = cosine_similarity(tnshap_results[1]['teacher_vals'], kernel_vals)
            else:
                kernel_mse = kernel_cosine = np.nan
            
            result_row = {
                'point_idx': point_idx,
                'budget': budget,
                'method': 'KernelSHAP',
                'order': 1,
                'runtime_s': kernel_time,
                'mse_vs_tnshap': kernel_mse,
                'cosine_vs_tnshap': kernel_cosine,
                'tnshap_teacher_time': tnshap_results[1]['teacher_time'],
                'tnshap_student_time': tnshap_results[1]['student_time'],
                'tnshap_mse': tnshap_results[1]['mse'],
                'tnshap_cosine': tnshap_results[1]['cosine']
            }
            all_results.append(result_row)
            print(f"      Runtime: {kernel_time:.3f}s, MSE: {kernel_mse:.6f}, Cosine: {kernel_cosine:.6f}")
            
            # SHAPIQ variants
            shapiq_methods = [
                ('SHAPIQ_regression_SII', 'regression', 'SII'),
                ('SHAPIQ_regression_FSII', 'regression', 'FSII'),
                ('SHAPIQ_permutation_SII', 'permutation', 'SII'),
                ('SHAPIQ_montecarlo_SII', 'montecarlo', 'SII')
            ]
            
            for method_name, approximator, index_name in shapiq_methods:
                for order in range(1, 4):
                    print(f"    {method_name} order {order}...")
                    shapiq_vals, shapiq_time = shapiq_baseline(
                        teacher, x_test, order, budget, approximator, index_name, seed=manifest['seed']
                    )
                    
                    if not np.isnan(shapiq_vals).all():
                        shapiq_mse = np.mean((np.array(tnshap_results[order]['teacher_vals']) - shapiq_vals) ** 2)
                        shapiq_cosine = cosine_similarity(tnshap_results[order]['teacher_vals'], shapiq_vals)
                    else:
                        shapiq_mse = shapiq_cosine = np.nan
                    
                    result_row = {
                        'point_idx': point_idx,
                        'budget': budget,
                        'method': method_name,
                        'order': order,
                        'runtime_s': shapiq_time,
                        'mse_vs_tnshap': shapiq_mse,
                        'cosine_vs_tnshap': shapiq_cosine,
                        'tnshap_teacher_time': tnshap_results[order]['teacher_time'],
                        'tnshap_student_time': tnshap_results[order]['student_time'],
                        'tnshap_mse': tnshap_results[order]['mse'],
                        'tnshap_cosine': tnshap_results[order]['cosine']
                    }
                    all_results.append(result_row)
                    print(f"      Runtime: {shapiq_time:.3f}s, MSE: {shapiq_mse:.6f}, Cosine: {shapiq_cosine:.6f}")
    
    return all_results, manifest

def main():
    """Run the baseline sweep experiment on diabetes dataset."""
    
    model_dir = "experiments/UCI/out_local_student_singlegrid/diabetes_seed2711_K89_m10"
    
    print(f"{'='*80}")
    print("BASELINE SWEEP EXPERIMENT - DIABETES DATASET")
    print(f"{'='*80}")
    
    # Run experiment
    results, manifest = run_baseline_sweep_experiment(model_dir)
    
    # Save results
    output_dir = "results_diabetes_baseline_sweep"
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # Add metadata
    df['dataset'] = 'diabetes'
    df['seed'] = manifest['seed']
    df['student_rank'] = manifest['rank']
    df['n_features'] = manifest['dim']
    df['student_train_r2'] = manifest['tn_info']['final_r2']
    
    # Save detailed results
    detailed_path = os.path.join(output_dir, "detailed_diabetes_baseline_sweep.csv")
    df.to_csv(detailed_path, index=False)
    print(f"\nSaved detailed results to {detailed_path}")
    
    # Create summary by method and budget
    summary_cols = ['method', 'order', 'budget']
    summary = df.groupby(summary_cols).agg({
        'runtime_s': ['mean', 'std', 'count'],
        'mse_vs_tnshap': ['mean', 'std'],
        'cosine_vs_tnshap': ['mean', 'std'],
        'tnshap_teacher_time': ['mean'],
        'tnshap_student_time': ['mean'],
        'tnshap_mse': ['mean'],
        'tnshap_cosine': ['mean']
    }).reset_index()
    
    # Flatten column names
    summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns]
    
    summary_path = os.path.join(output_dir, "summary_diabetes_baseline_sweep.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary results to {summary_path}")
    
    # Print key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    
    print(f"Dataset: diabetes (seed={manifest['seed']})")
    print(f"Features: {manifest['dim']}")
    print(f"Student rank: {manifest['rank']}")
    print(f"Student training R²: {manifest['tn_info']['final_r2']:.6f}")
    
    # Show runtime comparison
    print(f"\nRuntime Comparison (averaged across test points):")
    print(f"{'Method':<25} {'Order':<5} {'Budget':<8} {'Runtime(s)':<12} {'Cosine':<8}")
    print("-" * 65)
    
    for method in ['KernelSHAP', 'SHAPIQ_regression_SII']:
        method_data = df[df['method'] == method]
        if not method_data.empty:
            for order in sorted(method_data['order'].unique()):
                order_data = method_data[method_data['order'] == order]
                for budget in sorted(order_data['budget'].unique()):
                    budget_data = order_data[order_data['budget'] == budget]
                    if not budget_data.empty:
                        avg_runtime = budget_data['runtime_s'].mean()
                        avg_cosine = budget_data['cosine_vs_tnshap'].mean()
                        print(f"{method:<25} {order:<5} {budget:<8} {avg_runtime:<12.3f} {avg_cosine:<8.3f}")
    
    # TNShap reference
    tnshap_times = {}
    for order in [1, 2, 3]:
        order_data = df[df['order'] == order]
        if not order_data.empty:
            tnshap_times[order] = order_data['tnshap_teacher_time'].iloc[0]
    
    print(f"\nTNShap Reference Times:")
    for order, time_val in tnshap_times.items():
        print(f"  Order {order}: {time_val:.3f}s")
    
    return df

if __name__ == "__main__":
    results = main()
    print("\nDiabetes baseline sweep experiment completed!")
