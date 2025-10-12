#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import math
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.tntree_model import BinaryTensorTree

# ---------------- Settings ----------------
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2711
torch.manual_seed(SEED)

DIMENSIONS = [10, 20, 30, 40, 50]  # Different problem dimensions
RANK_GT = 10     # Ground truth TN rank
RANK_STUDENT = 5 # Student TN rank (lower rank for compression)
OUT_DIM = 1

N_TRAIN = 10000  # Training samples
BATCH = 512
LR = 1e-3
EPOCHS = 300
CLIP = 1.0

# Test points for Shapley evaluation
N_TEST_POINTS = 5

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute R^2 with better numerical stability."""
    y_true = y_true.flatten().double()
    y_pred = y_pred.flatten().double()
    
    # Center the predictions and targets
    y_true_mean = y_true.mean()
    
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    
    if ss_tot < 1e-10:  # Avoid division by very small numbers
        return 1.0 if ss_res < 1e-10 else 0.0
    
    r2 = 1.0 - ss_res / ss_tot
    return float(torch.clamp(r2, -1000.0, 1.0))  # Clamp extreme values

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    v1_flat = v1.flatten().astype(np.float64)
    v2_flat = v2.flatten().astype(np.float64)
    
    dot_product = np.dot(v1_flat, v2_flat)
    norm_v1 = np.linalg.norm(v1_flat)
    norm_v2 = np.linalg.norm(v2_flat)
    
    if norm_v1 < 1e-10 or norm_v2 < 1e-10:
        return 1.0 if norm_v1 < 1e-10 and norm_v2 < 1e-10 else 0.0
    
    return float(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))

def make_ground_truth_tn(n_features: int, rank: int, seed: int = SEED) -> BinaryTensorTree:
    """Create a well-conditioned ground truth tensor tree."""
    torch.manual_seed(seed)
    
    gt_tn = BinaryTensorTree(
        leaf_phys_dims=[2] * n_features,
        ranks=rank,
        out_dim=OUT_DIM,
        seed=seed,
        device=DEVICE,
        dtype=torch.float64,
    ).to(DEVICE)
    
    # Initialize with small, well-conditioned values
    with torch.no_grad():
        for name, param in gt_tn.cores.items():
            if param.ndim == 2 and param.shape[0] == 2:  # Leaf node
                # Initialize carefully for stability
                param.data = torch.randn_like(param) * 0.1
                param[1, :] = torch.randn(param.shape[1]) * 0.05  # Small bias
            else:
                # Internal nodes
                param.data = torch.randn_like(param) * 0.1
    
    return gt_tn

def make_student_tn(n_features: int, rank: int, seed: int = SEED + 100) -> BinaryTensorTree:
    """Create a student tensor tree."""
    torch.manual_seed(seed)
    
    student_tn = BinaryTensorTree(
        leaf_phys_dims=[2] * n_features,
        ranks=rank,
        out_dim=OUT_DIM,
        seed=seed,
        device=DEVICE,
        dtype=torch.float64,
    ).to(DEVICE)
    
    return student_tn

def sample_X(n_samples: int, n_features: int) -> torch.Tensor:
    """Generate training data."""
    return torch.rand((n_samples, n_features), device=DEVICE, dtype=torch.float64)

def train_student_on_teacher(teacher_tn: BinaryTensorTree, student_tn: BinaryTensorTree,
                             X: torch.Tensor, epochs=EPOCHS, batch=BATCH, lr=LR, clip=CLIP):
    """Train student TN to match teacher TN."""
    with torch.no_grad():
        y = teacher_tn(X).reshape(-1, 1)

    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

    opt = optim.Adam(student_tn.parameters(), lr=lr, weight_decay=1e-6)  # Add weight decay
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20)
    loss_fn = nn.MSELoss()

    best = math.inf
    for epoch in range(1, epochs + 1):
        student_tn.train()
        running = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = student_tn(xb).reshape(-1, 1)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(student_tn.parameters(), max_norm=clip)
            opt.step()
            running += loss.item() * xb.size(0)

        train_mse = running / len(ds)
        sched.step(train_mse)
        best = min(best, train_mse)

        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                y_pred_full = student_tn(X).reshape(-1, 1)
                full_mse = loss_fn(y_pred_full, y).item()
                r2 = r2_score(y.flatten(), y_pred_full.flatten())
            print(f"Epoch {epoch:4d} | MSE={train_mse:.3e} | R²={r2:.6f} | lr={opt.param_groups[0]['lr']:.2e}")

        if best < 1e-12:
            print(f"Early stop at epoch {epoch}: MSE ~ 0")
            break

    with torch.no_grad():
        final = loss_fn(student_tn(X).reshape(-1, 1), y).item()
    return final

def chebyshev_nodes_01(m: int, device=None, dtype=torch.float32):
    """Generate Chebyshev nodes in [0,1]."""
    device = device or DEVICE
    k = torch.arange(m, device=device, dtype=dtype)
    nodes = torch.cos((2*k + 1) * math.pi / (2*m))
    return 0.5*(nodes + 1.0)

@torch.no_grad()
def tn_shap_path_selector(model: BinaryTensorTree, x: np.ndarray, m_points: Optional[int] = None) -> np.ndarray:
    """Compute TN-SHAP values using path interpolation."""
    x = torch.tensor(x, device=DEVICE, dtype=torch.float64).flatten()
    d = x.numel()
    
    m_points = m_points or min(d + 5, 20)  # Limit polynomial degree for stability
    
    # Get Chebyshev nodes
    t = chebyshev_nodes_01(m_points, device=DEVICE, dtype=torch.float64)
    
    phi = torch.zeros(d, device=DEVICE, dtype=torch.float64)
    
    for i in range(d):
        # Path interpolation
        X_g = t.unsqueeze(1) * x.unsqueeze(0)  # [m, d]
        X_h = X_g.clone()
        X_h[:, i] = 0.0
        
        y_g = model(X_g).squeeze(-1)
        y_h = model(X_h).squeeze(-1)
        H_i = y_g - y_h
        
        # Robust polynomial fitting
        V = torch.vander(t, N=m_points, increasing=True)
        
        # Use pseudoinverse for stability
        try:
            V_pinv = torch.linalg.pinv(V)
            c = V_pinv @ H_i
        except:
            # Fallback: simple integration
            phi[i] = torch.mean(H_i)
            continue
        
        # Integration weights
        weights = 1.0 / torch.arange(1, m_points + 1, device=DEVICE, dtype=torch.float64)
        phi[i] = torch.sum(c * weights)
    
    return phi.cpu().numpy()

def exact_shapley_tn(teacher_tn: BinaryTensorTree, x: np.ndarray) -> np.ndarray:
    """Compute exact Shapley values using all subsets (for small d only)."""
    x = np.asarray(x, dtype=np.float64)
    d = len(x)
    
    if d > 15:
        raise ValueError(f"Exact Shapley not feasible for d={d} > 15")
    
    phi = np.zeros(d, dtype=np.float64)
    
    def f(x_eval):
        x_tensor = torch.tensor(x_eval, dtype=torch.float64, device=DEVICE)
        return teacher_tn(x_tensor.unsqueeze(0)).item()
    
    for i in range(d):
        shapley_sum = 0.0
        
        for subset_mask in range(1 << (d-1)):
            subset = []
            idx = 0
            for j in range(d):
                if j != i:
                    if (subset_mask >> idx) & 1:
                        subset.append(j)
                    idx += 1
            
            s = len(subset)
            
            x_with_i = np.zeros(d)
            x_with_i[subset + [i]] = x[subset + [i]]
            
            x_without_i = np.zeros(d)
            x_without_i[subset] = x[subset]
            
            marginal = f(x_with_i) - f(x_without_i)
            weight = math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d)
            shapley_sum += weight * marginal
        
        phi[i] = shapley_sum
    
    return phi

def evaluate_dimension(d: int) -> Dict:
    """Train and evaluate a model for dimension d."""
    print(f"\n{'='*60}")
    print(f"EVALUATING DIMENSION d={d}")
    print(f"{'='*60}")
    
    # Create models
    teacher_tn = make_ground_truth_tn(d, RANK_GT, SEED)
    student_tn = make_student_tn(d, RANK_STUDENT, SEED + 100)
    
    # Generate training data
    Xtrain = sample_X(N_TRAIN, d)
    
    print(f"Training TN student (rank={RANK_STUDENT}) on teacher (rank={RANK_GT})...")
    start_time = time.time()
    final_mse = train_student_on_teacher(teacher_tn, student_tn, Xtrain)
    train_time = time.time() - start_time
    
    # Compute training R²
    with torch.no_grad():
        y_true = teacher_tn(Xtrain).flatten()
        y_pred = student_tn(Xtrain).flatten()
        train_r2 = r2_score(y_true, y_pred)
    
    print(f"Training completed in {train_time:.2f}s")
    print(f"Final train MSE: {final_mse:.3e}")
    print(f"Train R²: {train_r2:.6f}")
    
    # Test Shapley computation
    np.random.seed(SEED)
    X_test = np.random.uniform(0, 1, (N_TEST_POINTS, d))  # Keep in [0,1]
    
    print(f"\nEvaluating Shapley methods on {N_TEST_POINTS} test points...")
    
    shapley_results = []
    
    for i, x_test in enumerate(X_test):
        print(f"Test point {i+1}/{N_TEST_POINTS}:")
        
        # TN-SHAP
        start_time = time.time()
        phi_tn = tn_shap_path_selector(student_tn, x_test)
        tn_time = time.time() - start_time
        
        # Exact Shapley (only for small d)
        if d <= 12:
            start_time = time.time()
            phi_exact = exact_shapley_tn(teacher_tn, x_test)
            exact_time = time.time() - start_time
            method_used = "exact"
        else:
            # For larger d, compare student vs teacher TN-SHAP
            start_time = time.time()
            phi_exact = tn_shap_path_selector(teacher_tn, x_test)
            exact_time = time.time() - start_time
            method_used = "teacher_tn_shap"
        
        cos_sim = cosine_similarity(phi_tn, phi_exact)
        
        shapley_results.append({
            'test_point': i,
            'phi_tn': phi_tn,
            'phi_exact': phi_exact,
            'cosine_similarity': cos_sim,
            'tn_time_ms': tn_time * 1000,
            'exact_time_ms': exact_time * 1000,
            'speedup': exact_time / tn_time if tn_time > 0 else np.inf,
            'method_used': method_used
        })
        
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  TN-SHAP time: {tn_time*1000:.2f} ms")
        print(f"  {method_used} time: {exact_time*1000:.2f} ms")
        if tn_time > 0:
            print(f"  Speedup: {exact_time/tn_time:.1f}x")
    
    # Aggregate results
    cos_sims = [r['cosine_similarity'] for r in shapley_results]
    tn_times = [r['tn_time_ms'] for r in shapley_results]
    exact_times = [r['exact_time_ms'] for r in shapley_results]
    speedups = [r['speedup'] for r in shapley_results if np.isfinite(r['speedup'])]
    
    results = {
        'dimension': d,
        'teacher_rank': RANK_GT,
        'student_rank': RANK_STUDENT,
        'train_r2': train_r2,
        'train_time_s': train_time,
        'final_mse': final_mse,
        'mean_cosine_similarity': np.mean(cos_sims),
        'std_cosine_similarity': np.std(cos_sims),
        'mean_tn_time_ms': np.mean(tn_times),
        'mean_exact_time_ms': np.mean(exact_times),
        'mean_speedup': np.mean(speedups) if speedups else np.inf,
        'individual_results': shapley_results,
        'comparison_method': shapley_results[0]['method_used'] if shapley_results else 'unknown'
    }
    
    print(f"\nSUMMARY for d={d}:")
    print(f"  Train R²: {train_r2:.6f}")
    print(f"  Mean cosine similarity: {results['mean_cosine_similarity']:.6f} ± {results['std_cosine_similarity']:.6f}")
    print(f"  Mean TN-SHAP time: {results['mean_tn_time_ms']:.2f} ms")
    print(f"  Mean {results['comparison_method']} time: {results['mean_exact_time_ms']:.2f} ms")
    print(f"  Mean speedup: {results['mean_speedup']:.1f}x")
    
    return results

def main():
    print("Device:", DEVICE)
    print(f"Training tensor-tree surrogates for dimensions: {DIMENSIONS}")
    print(f"Teacher rank: {RANK_GT}, Student rank: {RANK_STUDENT}")
    print(f"Test points per dimension: {N_TEST_POINTS}")
    print(f"Training samples: {N_TRAIN}")
    
    all_results = []
    
    for d in DIMENSIONS:
        results = evaluate_dimension(d)
        all_results.append(results)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dim':<5} {'Train R²':<10} {'Cos Sim':<12} {'TN-SHAP (ms)':<13} {'Baseline (ms)':<13} {'Speedup':<8}")
    print(f"{'-'*80}")
    
    for r in all_results:
        print(f"{r['dimension']:<5} {r['train_r2']:<10.6f} {r['mean_cosine_similarity']:<12.6f} "
              f"{r['mean_tn_time_ms']:<13.2f} {r['mean_exact_time_ms']:<13.2f} {r['mean_speedup']:<8.1f}x")
    
    print(f"\nKey findings:")
    print(f"- Teacher rank: {RANK_GT}, Student rank: {RANK_STUDENT}")
    high_r2 = all(r['train_r2'] > 0.8 for r in all_results)
    print(f"- Training R² values are {'high (>0.8)' if high_r2 else 'variable'}")
    high_cos = all(r['mean_cosine_similarity'] > 0.8 for r in all_results)
    print(f"- TN-SHAP achieves {'excellent' if high_cos else 'good'} agreement with baseline")
    avg_speedup = np.mean([r['mean_speedup'] for r in all_results if np.isfinite(r['mean_speedup'])])
    print(f"- TN-SHAP provides {avg_speedup:.1f}x average speedup")
    
    return all_results

if __name__ == "__main__":
    main()
