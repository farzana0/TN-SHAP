#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Teacher-Student Rank Sweep (TNShap, FIXED)
- Correct thin-diagonal selectors for order-1 and higher orders
- Degree-d polynomial fits with d+1 Chebyshev nodes (stable)
- Aggregate R^2 over many test points for stability
- Enhanced early stopping preserved (mild)
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
from src.tntree_model import make_balanced_binary_tensor_tree  # assumes available


# ----------------------------
# Teachers
# ----------------------------

class TensorTreeTeacher:
    """Teacher using tensor tree with specified rank."""
    def __init__(self, n_features: int, ranks: int, seed: int = 42):
        self.n_features = n_features
        self.ranks = ranks
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)
        phys_dims = [2] * n_features  # binary selectors (presence channels)
        self.teacher_tree = make_balanced_binary_tensor_tree(
            leaf_phys_dims=phys_dims,
            ranks=ranks,
            out_dim=1,
            assume_bias_when_matrix=True,
            seed=seed
        )
        # Chebyshev nodes in [0,1]; keep a long list, we will slice to d+1
        self.t_nodes = self._chebyshev_nodes(n_features + 16)

    @staticmethod
    def _chebyshev_nodes(m: int) -> np.ndarray:
        i = np.arange(1, m + 1)
        nodes = 0.5 * (1 + np.cos((2 * i - 1) * np.pi / (2 * m)))
        return nodes[::-1]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y = self.teacher_tree.forward(X_tensor).view(-1)
        return y.detach().cpu().numpy()


class GenericMultilinearTeacher:
    """Teacher using sparse multilinear function up to max_order."""
    def __init__(self, n_features: int, max_order: int = 3, sparsity: float = 0.3, seed: int = 42):
        self.n_features = n_features
        self.max_order = max_order
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.coefficients = {}
        for order in range(max_order + 1):
            for subset in itertools.combinations(range(n_features), order):
                if order == 0 or rng.random() < sparsity:
                    self.coefficients[subset] = rng.normal(0.0, 1.2)

        self.t_nodes = self._chebyshev_nodes(n_features + 16)

    @staticmethod
    def _chebyshev_nodes(m: int) -> np.ndarray:
        i = np.arange(1, m + 1)
        nodes = 0.5 * (1 + np.cos((2 * i - 1) * np.pi / (2 * m)))
        return nodes[::-1]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        y = np.zeros(n, dtype=float)
        for subset, coeff in self.coefficients.items():
            if len(subset) == 0:
                y += coeff
            else:
                term = np.ones(n, dtype=float)
                for idx in subset:
                    term *= X[:, idx]
                y += coeff * term
        return y


# ----------------------------
# TNShap Calculator (FIXED)
# ----------------------------

class TNShapCalculator:
    """
    TNShap calculator with fixed thin-diagonal paths and degree-d fits.
    """
    def __init__(self, model, t_nodes: np.ndarray, n_features: int):
        self.model = model
        self.t_nodes = t_nodes
        self.n_features = n_features

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, 'evaluate'):
            return self.model.evaluate(X)
        # PyTorch module
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_pred = self.model.forward(X_tensor).view(-1)
        return y_pred.detach().cpu().numpy()

    # ----- Polynomial fitting utilities -----

    def _degree_d_vandermonde(self, d: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (t, V_pinv) for degree-d polynomial fit using d+1 Chebyshev nodes in [0,1].
        V has columns [t^0, t^1, ..., t^d] (increasing=True).
        """
        t = self.t_nodes[:d+1]  # length d+1
        V = np.vander(t, N=d+1, increasing=True)
        V_pinv = np.linalg.pinv(V)  # numerically stable
        return t, V_pinv

    # ----- Order-1 Shapley values (thin-diagonal include/exclude) -----

    def shapley_values_tnshap(self, x: np.ndarray) -> np.ndarray:
        d = self.n_features
        t, V_pinv = self._degree_d_vandermonde(d)
        phi = np.zeros(d, dtype=float)

        # Shapley weights for terms that include i: k = 1..d
        from math import comb
        w = np.zeros(d+1, dtype=float)
        for k in range(1, d+1):
            w[k] = 1.0 / (k * comb(d-1, k-1))

        for i in range(d):
            # Build evaluations g_inc(t) and g_exc(t)
            X_inc = np.tile(x, (d+1, 1))
            X_exc = np.tile(x, (d+1, 1))
            comp = [j for j in range(d) if j != i]

            # sweep complements by t (same t for all complements)
            X_inc[:, comp] *= t[:, None]
            X_exc[:, comp] *= t[:, None]
            # feature i: included vs excluded
            # included: keep x_i as-is; excluded: zero it out
            X_exc[:, i] = 0.0

            f_inc = self._predict(X_inc)
            f_exc = self._predict(X_exc)
            diff = f_inc - f_exc  # isolates i-containing terms

            c = V_pinv @ diff  # coefficients c_0..c_d
            phi[i] = float(np.dot(c, w))

        return phi

    # ----- Higher-order Shapley interactions (subset path + inclusion-exclusion) -----

    def _poly_coeffs_for_subset(self, x: np.ndarray, subset: Tuple[int, ...]) -> np.ndarray:
        """
        g_S(t) = f( S_i(1) for i in S, S_j(t) for j not in S )
        Fit degree-d polynomial and return coeffs c_0..c_d.
        """
        d = self.n_features
        t, V_pinv = self._degree_d_vandermonde(d)

        X = np.tile(x, (d+1, 1))
        comp = [j for j in range(d) if j not in subset]
        # sweep complements by t; members of S remain fully on
        X[:, comp] *= t[:, None]

        vals = self._predict(X)  # shape (d+1,)
        c = V_pinv @ vals
        return c

    def shapley_interactions_tnshap(self, x: np.ndarray, order: int) -> Dict[Tuple[int, ...], float]:
        """
        Return dict mapping subset -> SII value for given order.
        Order=1 returns per-feature Shapley values (equivalent to shapley_values_tnshap).
        For r>=2, use inclusion-exclusion over subset polynomials; take the coefficient at degree r.
        """
        d = self.n_features
        if order == 1:
            phi = self.shapley_values_tnshap(x)
            return {(i,): float(phi[i]) for i in range(d)}

        interactions = {}
        for S in itertools.combinations(range(d), order):
            total = 0.0
            # Inclusion-exclusion over all T subset of S
            for k in range(order + 1):
                for T in itertools.combinations(S, k):
                    coeffs_T = self._poly_coeffs_for_subset(x, T)
                    c_k = coeffs_T[order] if order < len(coeffs_T) else 0.0
                    sign = (-1) ** (order - k)
                    total += sign * c_k
            interactions[S] = float(total)
        return interactions


# ----------------------------
# Training (student)
# ----------------------------

def train_student_tensor_tree_enhanced(teacher, n_features: int, rank: int,
                                       n_samples: int = 10000, epochs: int = 1500, seed: int = 42):
    """
    Train a student tensor tree with mild early stopping and live heartbeats.
    Shows loss, R², teacher rank, and student rank for clear monitoring.
    """
    print(f"\n🧩 Training student tensor tree:")
    print(f"   → Teacher rank = {getattr(teacher, 'ranks', 'N/A')}, Student rank = {rank}, Seed = {seed}")
    print(f"   → Samples = {n_samples}, Epochs = {epochs}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = np.random.normal(0, 1, (n_samples, n_features))
    y_train = teacher.evaluate(X_train)

    student = make_balanced_binary_tensor_tree(
        leaf_phys_dims=[2] * n_features,
        ranks=rank,
        out_dim=1,
        assume_bias_when_matrix=True,
        seed=seed
    )

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1)

    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=75, factor=0.7, verbose=False)

    best_loss = float('inf')
    patience = 0
    max_patience = 200
    loss_threshold = 1e-8
    zero_loss_threshold = 1e-10
    r2_threshold = 0.9999
    perfect_r2_threshold = 0.99999
    improvement_threshold = 1e-7

    # start timer
    start_time = time.time()

    for epoch in range(epochs):
        opt.zero_grad()
        y_pred = student.forward(X_tensor).view(-1)
        loss = nn.MSELoss()(y_pred, y_tensor)
        loss.backward()
        opt.step()
        sched.step(loss)

        cur = loss.item()
        if cur < best_loss - improvement_threshold:
            best_loss = cur
            patience = 0
        else:
            patience += 1

        # Compute R² occasionally (to see improvement)
        if epoch % 25 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                r2 = r2_score(y_tensor.cpu().numpy(), y_pred.detach().cpu().numpy())
            elapsed = time.time() - start_time
            print(f"      Epoch {epoch:4d} | Loss = {cur:.5e} | R² = {r2:.6f} | Patience = {patience:3d} | Time = {elapsed/60:.1f} min")

        # Early stopping checks every 100 epochs
        if epoch % 100 == 0:
            with torch.no_grad():
                r2 = r2_score(y_tensor.cpu().numpy(), y_pred.detach().cpu().numpy())
            if cur < loss_threshold:
                print(f"      ⏹ Early stop (loss < {loss_threshold:.1e}) at epoch {epoch}")
                break
            if cur < zero_loss_threshold:
                print(f"      ⏹ Early stop (zero loss) at epoch {epoch}")
                break
            if r2 > perfect_r2_threshold or r2 > r2_threshold:
                print(f"      ✅ Early stop (R² ≈ {r2:.5f}) at epoch {epoch}")
                break

        if patience > max_patience or torch.isnan(loss):
            print(f"      ⚠️ Early stop (patience exceeded or NaN loss) at epoch {epoch}")
            break

    total_time = time.time() - start_time
    final_r2 = r2_score(y_tensor.cpu().numpy(), y_pred.detach().cpu().numpy())
    print(f"   ✅ Final training R² = {final_r2:.6f} | Best loss = {best_loss:.2e} | Total time = {total_time/60:.1f} min\n")
    return student, final_r2


# ----------------------------
# Experiment runner (aggregated R^2)
# ----------------------------

def run_single_experiment_aggregated(teacher_name: str, teacher,
                                     student_rank: int, n_features: int,
                                     test_points: np.ndarray, run_seed: int,
                                     n_runs: int) -> Dict:
    """
    Train one student; compute aggregated R^2 across all test points
    for orders 1,2,3 to stabilize R^2.
    """
    student, train_r2 = train_student_tensor_tree_enhanced(
        teacher, n_features, student_rank,
        n_samples=10000, epochs=1500, seed=run_seed
    )

    tcalc = TNShapCalculator(teacher, teacher.t_nodes, n_features)
    scalc = TNShapCalculator(student, teacher.t_nodes, n_features)

    # Aggregate across all points
    o1_teacher, o1_student = [], []
    o2_teacher, o2_student = [], []
    o3_teacher, o3_student = [], []

    for x in test_points:
        # Order-1
        t1 = tcalc.shapley_interactions_tnshap(x, 1)
        s1 = scalc.shapley_interactions_tnshap(x, 1)
        for i in range(n_features):
            o1_teacher.append(t1[(i,)])
            o1_student.append(s1[(i,)])

        # Order-2
        t2 = tcalc.shapley_interactions_tnshap(x, 2)
        s2 = scalc.shapley_interactions_tnshap(x, 2)
        for pair in itertools.combinations(range(n_features), 2):
            o2_teacher.append(t2[pair])
            o2_student.append(s2[pair])

        # Order-3
        t3 = tcalc.shapley_interactions_tnshap(x, 3)
        s3 = scalc.shapley_interactions_tnshap(x, 3)
        for tri in itertools.combinations(range(n_features), 3):
            o3_teacher.append(t3[tri])
            o3_student.append(s3[tri])

    # Compute R^2 across all stacked values
    order_1_r2 = r2_score(o1_teacher, o1_student) if len(o1_teacher) > 1 else 0.0
    order_2_r2 = r2_score(o2_teacher, o2_student) if len(o2_teacher) > 1 else 0.0
    order_3_r2 = r2_score(o3_teacher, o3_student) if len(o3_teacher) > 1 else 0.0

    return {
        'teacher_name': teacher_name,
        'student_rank': student_rank,
        'run_seed': run_seed,
        'train_r2': float(train_r2),
        'order_1_r2': float(order_1_r2),
        'order_2_r2': float(order_2_r2),
        'order_3_r2': float(order_3_r2),
    }


def run_teacher_student_experiment_multiple_seeds(teacher_name: str, teacher,
                                                  student_ranks: List[int],
                                                  n_features: int = 4,
                                                  n_test_points: int = 128,
                                                  n_runs: int = 10):
    """
    Run for one teacher type across ranks and seeds (aggregated R^2).
    """
    print(f"\n=== TEACHER: {teacher_name} ===")
    # Fixed test points
    rng = np.random.default_rng(4242)
    test_points = rng.normal(0, 1, size=(n_test_points, n_features))

    # Generate seeds
    rng = np.random.default_rng(12345)
    run_seeds = rng.integers(1000, 9999, size=n_runs).tolist()
    print(f"Seeds: {run_seeds}")

    results = {}
    for rnk in student_ranks:
        per_runs = []
        for sidx, seed in enumerate(run_seeds):
            res = run_single_experiment_aggregated(
                teacher_name, teacher, rnk,
                n_features, test_points, seed, n_runs
            )
            per_runs.append(res)
        results[rnk] = {'run_seeds': run_seeds, 'detailed_results': per_runs}
    return results


# ----------------------------
# Summaries & saving
# ----------------------------

def save_results_with_stats(all_results: Dict, outdir: str, n_runs: int = 10):
    os.makedirs(outdir, exist_ok=True)
    detailed_rows = []
    summary_rows = []

    for teacher_name, ranks_dict in all_results.items():
        for rank, payload in ranks_dict.items():
            runs = payload['detailed_results']
            detailed_rows.extend(runs)
            df = pd.DataFrame(runs)

            # Per-run already aggregated; compute mean/std across runs
            summary_rows.append({
                'teacher_name': teacher_name,
                'student_rank': rank,
                'n_runs': n_runs,
                'run_seeds': str(payload['run_seeds']),
                'train_r2_mean': df['train_r2'].mean(),
                'train_r2_std': df['train_r2'].std(ddof=1),
                'order_1_r2_mean': df['order_1_r2'].mean(),
                'order_1_r2_std': df['order_1_r2'].std(ddof=1),
                'order_2_r2_mean': df['order_2_r2'].mean(),
                'order_2_r2_std': df['order_2_r2'].std(ddof=1),
                'order_3_r2_mean': df['order_3_r2'].mean(),
                'order_3_r2_std': df['order_3_r2'].std(ddof=1),
            })

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = pd.DataFrame(summary_rows)

    ts = time.strftime("%Y%m%d_%H%M%S")
    p_summary = os.path.join(outdir, f"summary_{ts}.csv")
    p_detailed = os.path.join(outdir, f"detailed_{ts}.csv")
    summary_df.to_csv(p_summary, index=False)
    detailed_df.to_csv(p_detailed, index=False)
    print(f"Saved summary: {p_summary}")
    print(f"Saved detailed: {p_detailed}")
    return summary_df, detailed_df


def print_final_summary_with_stats(summary_df: pd.DataFrame, n_runs: int = 10):
    print("\n" + "=" * 96)
    print(f"FINAL SUMMARY  (mean ± std over {n_runs} runs)")
    print("=" * 96)
    for teacher in summary_df['teacher_name'].unique():
        df_t = summary_df[summary_df['teacher_name'] == teacher].sort_values('student_rank')
        print(f"\n{teacher}:")
        print("  Rank | Train R² (μ±σ)    | Order 1 R² (μ±σ)  | Order 2 R² (μ±σ)  | Order 3 R² (μ±σ)")
        print("  -----|-------------------|-------------------|-------------------|-------------------")
        for _, row in df_t.iterrows():
            train_str = f"{row['train_r2_mean']:.4f}±{row['train_r2_std']:.4f}"
            o1_str = f"{row['order_1_r2_mean']:.4f}±{row['order_1_r2_std']:.4f}"
            o2_str = f"{row['order_2_r2_mean']:.4f}±{row['order_2_r2_std']:.4f}"
            o3_str = f"{row['order_3_r2_mean']:.4f}±{row['order_3_r2_std']:.4f}"
            print(f"  {int(row['student_rank']):4d} | {train_str:17s} | {o1_str:17s} | {o2_str:17s} | {o3_str:17s}")


# ----------------------------
# Main
# ----------------------------

def main():
    n_features = 4
    student_ranks = [2, 4, 6, 8, 10, 16]
    n_runs = 10
    n_test_points = 128
    outdir = "results_teacher_student_multi_seed_fixed"

    print("Starting TNShap rank sweep (FIXED paths, degree-d fits, aggregated R^2)")
    print(f"Features: {n_features}, Ranks: {student_ranks}, Runs: {n_runs}, Test points: {n_test_points}")

    # Teachers (fixed seeds)
    teachers = {
        'TensorTree_Rank3': TensorTreeTeacher(n_features, ranks=3, seed=42),
        'TensorTree_Rank16': TensorTreeTeacher(n_features, ranks=16, seed=43),
        'GenericMultilinear': GenericMultilinearTeacher(n_features, max_order=3, seed=44),
    }

    all_results = {}
    for name, tch in teachers.items():
        res = run_teacher_student_experiment_multiple_seeds(
            name, tch, student_ranks,
            n_features=n_features,
            n_test_points=n_test_points,
            n_runs=n_runs
        )
        all_results[name] = res

    summary_df, detailed_df = save_results_with_stats(all_results, outdir, n_runs=n_runs)
    print_final_summary_with_stats(summary_df, n_runs=n_runs)
    return all_results, summary_df, detailed_df


if __name__ == "__main__":
    results, summary, detailed = main()
