# src/extra_experiments/exp1_sparse_poly/run.py

import time
import sys
import numpy as np
import torch

from .data import generate_data
from .evaluate import accuracy_at_k
from .config import N_EXPLAIN

from .masked_dataset import chebyshev_nodes_01, build_masked_dataset_k1
from .tn_surrogate import train_tn_on_masked
from .tnshap_path import tnshap_order1_path


def run():
    print("Generating data + teacher...", flush=True)
    X_train, y_train, X_test, y_test, S, f = generate_data()
    d = X_train.shape[1]

    # 1) Choose targets for explanation - REDUCED TO 10 for faster local surrogate
    K = 10
    Xtargets = X_test[:K]
    print(f"[Targets] Using {K} explanation points for local surrogate", flush=True)

    # 2) Chebyshev grid - 50 nodes per feature for good resolution
    m = 50
    t_nodes = chebyshev_nodes_01(m)
    print(f"[Grid] m={m}, Chebyshev nodes on [0,1]", flush=True)

    # 3) Masked dataset along paths (k=1 only)
    masked_X = build_masked_dataset_k1(Xtargets, t_nodes)
    print(f"[FewEval] masked_X shape: {masked_X.shape}", flush=True)

    # 4) Query teacher (ground-truth polynomial f)
    y_teacher = f(masked_X).astype(np.float32)
    print("[Teacher y] mean =", float(y_teacher.mean()), "std =", float(y_teacher.std()), flush=True)


    # 5) Train TN surrogate on masked datapoints
    print("[TN] training surrogate on masked Chebyshev dataset...", flush=True)
    print(f"[TN] Dataset size: {masked_X.shape[0]} samples (10 points × 50 nodes × {d} features)", flush=True)
    tn_model, tn_info = train_tn_on_masked(
        masked_X,
        y_teacher,
        ranks=48,           # Good capacity for 50 features
        fmap_out_dim=10,    # Rich feature representation
        seed=0,
        max_epochs=500,     # Allow time to converge
        lr=0.01,            # Conservative LR: 1e-2 (was 0.08, too high for deep tree)
        test_split=0.05,    # 5% test split for honest evaluation
        fmap_hidden=80,     # MLP hidden size
        print_every=25,     # Frequent progress updates
        patience=100,       # Allow convergence time
        target_r2=0.9,      # TARGET R² = 0.9
        batch_size=2048,    # Larger batches for stable gradients (was 128)
    )
    
    # Report TN training time
    tn_train_time = tn_info.get('elapsed', 0)
    print(f"\n{'='*70}", flush=True)
    print(f"TN TRAINING SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Training time: {tn_train_time:.2f}s ({tn_train_time/60:.2f} min)", flush=True)
    print(f"Train R²:      {tn_info.get('final_r2_train', 0):.4f}", flush=True)
    print(f"Test R²:       {tn_info.get('final_r2_test', 0):.4f}", flush=True)
    print(f"Epochs:        {tn_info.get('epochs', 0)}", flush=True)
    print(f"Dataset size:  {masked_X.shape[0]} samples", flush=True)
    print(f"{'='*70}\n", flush=True)

    # 6) Evaluate TN-Shap (path) on TN surrogate
    print(f"{'='*70}", flush=True)
    print(f"SHAPLEY VALUE EVALUATION ({K} points)", flush=True)
    print(f"{'='*70}", flush=True)
    
    recalls = []
    times = []
    eval_times = []
    solve_times = []

    shapley_start = time.time()
    for i in range(K):
        x_np = Xtargets[i]  # (d,)

        t0 = time.time()
        phi, timing = tnshap_order1_path(
            tn_model,
            x_np,
            m=m,
            t_nodes=t_nodes,  # use SAME grid
        )
        dt = time.time() - t0

        r = accuracy_at_k(phi, S, k=len(S))
        recalls.append(r)
        times.append(dt)
        eval_times.append(timing['t_eval_s'])
        solve_times.append(timing['t_solve_s'])

        print(
            f"[Point {i:2d}] recall={r:.3f}, total={dt:.3f}s "
            f"(eval={timing['t_eval_s']:.4f}s, solve={timing['t_solve_s']:.4f}s)",
            flush=True
        )

    shapley_total = time.time() - shapley_start

    print(f"\n{'='*70}", flush=True)
    print(f"FINAL RESULTS (TN-SHAP on Sparse Polynomial)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\nTIMING:", flush=True)
    print(f"  TN training time:        {tn_train_time:.2f}s ({tn_train_time/60:.2f} min)", flush=True)
    print(f"  Shapley eval total:      {shapley_total:.2f}s ({shapley_total/60:.2f} min)", flush=True)
    print(f"  Shapley per point (avg): {np.mean(times):.3f}s ± {np.std(times):.3f}s", flush=True)
    print(f"    - Model eval time:     {np.mean(eval_times):.4f}s ± {np.std(eval_times):.4f}s", flush=True)
    print(f"    - Path solver time:    {np.mean(solve_times):.4f}s ± {np.std(solve_times):.4f}s", flush=True)
    print(f"\nACCURACY:", flush=True)
    print(f"  Recall@{len(S)} (mean):     {np.mean(recalls):.4f} ± {np.std(recalls):.4f}", flush=True)
    print(f"  Recall@{len(S)} (min/max):  {np.min(recalls):.4f} / {np.max(recalls):.4f}", flush=True)
    print(f"\nMODEL QUALITY:", flush=True)
    print(f"  TN Train R²:             {tn_info.get('final_r2_train', 0):.4f}", flush=True)
    print(f"  TN Test R²:              {tn_info.get('final_r2_test', 0):.4f}", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    run()
