#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# run_d_sweep.py
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Local import
from src.tntree_model import BinaryTensorTree

# ---------------- Defaults / Config ----------------
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 2711
torch.manual_seed(SEED)
np.random.seed(SEED)

# Choose which model provides the "reference" Shapley (for cosine comparisons):
#   "teacher" -> validate algorithm vs the ground-truth function
#   "student" -> validate implementation internally on the student surrogate
COMPARE_AGAINST = "teacher"

# Teacher/Student ranks
RANK_GT = 8
RANK_STUDENT = 5

# Training hyperparams
N_TRAIN = 10000
BATCH = 128
LR = 5e-3
EPOCHS = 200
CLIP = 0.5
EARLY_STOP_R2 = 0.97

# Eval
N_TEST_POINTS = 5  # per dimension
MAX_EXACT_D = 15   # use exact enumeration up to this d; else use path integral approx


# ---------------- Utils ----------------
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if y_true.numel() != y_pred.numel():
        return float("nan")
    ss_res = torch.sum((y_true - y_pred) ** 2)
    y_mean = y_true.mean()
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def safe_cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = np.asarray(v1, float).ravel()
    v2 = np.asarray(v2, float).ravel()
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 and n2 < 1e-12:
        return 1.0
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


def chebyshev_nodes_01(m: int, device=None, dtype=torch.float64):
    device = device or DEVICE
    k = torch.arange(m, device=device, dtype=dtype)
    nodes = torch.cos((2 * k + 1) * math.pi / (2 * m))
    return 0.5 * (nodes + 1.0)


# ---------------- TN builders ----------------
def make_ground_truth_tn(n_features: int, rank: int, seed: int = SEED) -> BinaryTensorTree:
    torch.manual_seed(seed)
    effective_rank = min(rank, n_features + 2)
    gt_tn = BinaryTensorTree(
        leaf_phys_dims=[2] * n_features,
        ranks=effective_rank,
        out_dim=1,
        seed=seed,
        device=DEVICE,
        dtype=torch.float64,
    ).to(DEVICE)

    with torch.no_grad():
        init_scale = 0.1
        # initialize cores
        for _, param in gt_tn.cores.items():
            if param.ndim == 2 and param.shape[0] == 2:
                param.data.normal_(0, init_scale)
                param[1, :] = torch.clamp(param[1, :], -0.5, 0.5)
            else:
                param.data.normal_(0, init_scale * 0.5)

        # rescale outputs to moderate std
        test_X = torch.rand(1000, n_features, device=DEVICE, dtype=torch.float64)
        test_y = gt_tn(test_X)
        if test_y.std() > 1e-8:
            target_std = 0.5
            scale = float(target_std / test_y.std().item())
            for p in gt_tn.parameters():
                p.data *= scale
    return gt_tn


def make_student_tn(n_features: int, rank: int, seed: int = SEED + 100) -> BinaryTensorTree:
    torch.manual_seed(seed)
    st = BinaryTensorTree(
        leaf_phys_dims=[2] * n_features,
        ranks=rank,
        out_dim=1,
        seed=seed,
        device=DEVICE,
        dtype=torch.float64,
    ).to(DEVICE)
    with torch.no_grad():
        for p in st.parameters():
            p.data.normal_(0, 0.01)
    return st


# ---------------- Exact / Approx Shapley ----------------
def exact_shapley_tn(model: BinaryTensorTree, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    d = len(x)
    if d > 20:
        raise ValueError("Exact Shapley infeasible for d > 20")

    def f(x_eval: np.ndarray) -> float:
        xx = torch.tensor(x_eval, dtype=torch.float64, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            return float(model(xx).squeeze().item())

    phi = np.zeros(d, float)
    all_idx = list(range(d))
    for i in range(d):
        ssum = 0.0
        other = [j for j in all_idx if j != i]
        n_other = len(other)
        for mask in range(1 << n_other):
            subset = [other[j] for j in range(n_other) if (mask >> j) & 1]
            s = len(subset)

            x_with_i = np.zeros(d, float)
            x_with_i[subset + [i]] = x[subset + [i]]

            x_without_i = np.zeros(d, float)
            x_without_i[subset] = x[subset]

            marginal = f(x_with_i) - f(x_without_i)
            weight = math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d)
            ssum += weight * marginal
        phi[i] = ssum
    return phi


@torch.no_grad()
def efficient_shapley_tn(model: BinaryTensorTree, x: np.ndarray, n_pts: int = 128) -> np.ndarray:
    x_t = torch.tensor(np.asarray(x, float), dtype=torch.float64, device=DEVICE).flatten()
    d = x_t.numel()
    phi = torch.zeros(d, dtype=torch.float64, device=DEVICE)

    t_values = torch.linspace(0, 1, n_pts, device=DEVICE, dtype=torch.float64)
    for i in range(d):
        ssum = 0.0
        for t in t_values:
            x_with_i = torch.zeros_like(x_t)
            x_with_i[: i + 1] = t * x_t[: i + 1]

            x_without_i = torch.zeros_like(x_t)
            x_without_i[: i] = t * x_t[: i]

            ssum += model(x_with_i.unsqueeze(0)).item() - model(x_without_i.unsqueeze(0)).item()
        phi[i] = ssum / n_pts

    return phi.cpu().numpy()


# ---------------- TN‑SHAP (original/optimized) ----------------
@torch.no_grad()
def tn_shap_path_selector(model: BinaryTensorTree, x: np.ndarray, m_points: Optional[int] = None) -> np.ndarray:
    x_t = torch.tensor(x, device=DEVICE, dtype=torch.float64).flatten()
    d = x_t.numel()
    m_points = m_points or d
    t = chebyshev_nodes_01(m_points, device=DEVICE, dtype=torch.float64)

    phi = torch.zeros(d, device=DEVICE, dtype=torch.float64)
    V = torch.vander(t, N=m_points, increasing=True)

    for i in range(d):
        X_g = t.unsqueeze(1) * x_t.unsqueeze(0)
        X_h = X_g.clone()
        X_h[:, i] = 0.0

        y_g = model(X_g).squeeze(-1)
        y_h = model(X_h).squeeze(-1)
        H_i = y_g - y_h

        try:
            c = torch.linalg.solve(V, H_i.unsqueeze(1)).squeeze(1)
        except RuntimeError:
            c = torch.linalg.lstsq(V, H_i.unsqueeze(1)).solution.squeeze(1)

        weights = 1.0 / torch.arange(1, m_points + 1, device=DEVICE, dtype=torch.float64)
        phi[i] = torch.sum(c * weights)

    return phi.cpu().numpy()


@torch.no_grad()
def tn_shap_optimized(model: BinaryTensorTree, x: np.ndarray, m_points: Optional[int] = None) -> np.ndarray:
    x_t = torch.tensor(x, device=DEVICE, dtype=torch.float64).flatten()
    d = x_t.numel()
    m_points = m_points or min(d + 5, 20)

    t = chebyshev_nodes_01(m_points, device=DEVICE, dtype=torch.float64)
    X_g = t.unsqueeze(1) * x_t.unsqueeze(0)  # [m, d]
    y_g = model(X_g).squeeze(-1)            # [m]

    V = torch.vander(t, N=m_points, increasing=True)
    V_pinv = torch.linalg.pinv(V)
    weights = 1.0 / torch.arange(1, m_points + 1, device=DEVICE, dtype=torch.float64)

    phi = torch.zeros(d, device=DEVICE, dtype=torch.float64)
    for i in range(d):
        X_h = X_g.clone()
        X_h[:, i] = 0.0
        y_h = model(X_h).squeeze(-1)
        H_i = y_g - y_h
        c = V_pinv @ H_i
        phi[i] = torch.sum(c * weights)

    return phi.cpu().numpy()


# ---------------- Data ----------------
def sample_X(n_samples: int, n_features: int) -> torch.Tensor:
    torch.manual_seed(SEED)
    xs: List[torch.Tensor] = []
    if n_features <= 4:
        n_grid = max(2, int(round(n_samples ** (1.0 / n_features))))
        grid = torch.linspace(0.1, 0.9, n_grid, device=DEVICE, dtype=torch.float64)
        meshes = torch.meshgrid(*([grid] * n_features), indexing="ij")
        X_grid = torch.stack([m.reshape(-1) for m in meshes], dim=1)
        if X_grid.size(0) > n_samples // 2:
            idx = torch.randperm(X_grid.size(0), device=DEVICE)[: n_samples // 2]
            X_grid = X_grid[idx]
        xs.append(X_grid)
    n_so_far = sum(x.size(0) for x in xs) if xs else 0
    n_random = max(0, n_samples - n_so_far)
    if n_random > 0:
        xs.append(torch.rand(n_random, n_features, device=DEVICE, dtype=torch.float64))
    X = torch.cat(xs, dim=0) if xs else torch.rand(n_samples, n_features, device=DEVICE, dtype=torch.float64)
    if X.size(0) > n_samples:
        X = X[torch.randperm(X.size(0), device=DEVICE)[:n_samples]]
    elif X.size(0) < n_samples:
        pad = torch.rand(n_samples - X.size(0), n_features, device=DEVICE, dtype=torch.float64)
        X = torch.cat([X, pad], dim=0)
    return X[torch.randperm(X.size(0), device=DEVICE)]


def gen_test_points(n_points: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    centers = rng.uniform(-0.5, 0.5, size=(n_points, d))
    noise = rng.normal(0, 0.2, size=(n_points, d))
    pts = np.clip(centers + noise, -1.0, 1.0)
    return pts


# ---------------- Training ----------------
def train_student(teacher: BinaryTensorTree, student: BinaryTensorTree, X: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        y = teacher(X).reshape(-1, 1)
        print(f"Target stats: mean={y.mean().item():.3f}, std={y.std().item():.3f}, "
              f"range=[{y.min().item():.3f}, {y.max().item():.3f}]")

    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

    opt = optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15, verbose=True)
    loss_fn = nn.MSELoss()

    best_r2 = -1e9
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        student.train()
        running = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = student(xb).reshape(-1, 1)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), CLIP)
            opt.step()
            running += loss.item() * xb.size(0)
        train_mse = running / len(ds)
        sched.step(train_mse)

        if epoch % 10 == 0 or epoch == 1:
            student.eval()
            with torch.no_grad():
                y_pred = student(X).reshape(-1, 1)
                full_mse = loss_fn(y_pred, y).item()
                r2 = r2_score(y.flatten(), y_pred.flatten())
            print(f"Epoch {epoch:4d} | batch-MSE={train_mse:.3e} | full-MSE={full_mse:.3e} | "
                  f"R^2={r2:.6f} | lr={opt.param_groups[0]['lr']:.2e}")
            if r2 > best_r2:
                best_r2 = r2
                patience = 0
            else:
                patience += 1
            if r2 > EARLY_STOP_R2 or patience >= 30:
                print(f"Early stop at epoch {epoch} (R^2={r2:.5f}, patience={patience})")
                break
        if train_mse < 1e-12:
            print(f"Early stop at epoch {epoch}: MSE ~ 0")
            break

    student.eval()
    with torch.no_grad():
        y_pred = student(X).reshape(-1, 1)
        final_mse = loss_fn(y_pred, y).item()
        final_r2 = r2_score(y.flatten(), y_pred.flatten())
    return {"final_mse": final_mse, "train_r2": final_r2}


# ---------------- Evaluation per dimension ----------------
def evaluate_dimension(d: int) -> Dict:
    print("\n" + "=" * 60)
    print(f"Evaluating dimension d={d}")
    print("=" * 60)

    teacher = make_ground_truth_tn(d, RANK_GT, SEED)
    student = make_student_tn(d, RANK_STUDENT, SEED + 100)

    # Train
    Xtrain = sample_X(N_TRAIN, d)
    t0 = time.time()
    train_stats = train_student(teacher, student, Xtrain)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s | R^2={train_stats['train_r2']:.5f}")

    # Reference model for fair cosine (avoid student-vs-teacher mismatch)
    if COMPARE_AGAINST == "teacher":
        ref_model = teacher
    elif COMPARE_AGAINST == "student":
        ref_model = student
    else:
        raise ValueError("COMPARE_AGAINST must be 'teacher' or 'student'")

    # Test points
    X_test = gen_test_points(N_TEST_POINTS, d)

    results = []
    for i, x_test in enumerate(X_test):
        print(f"Test point {i+1}/{N_TEST_POINTS}")
        x_eval = (np.asarray(x_test, float) + 1.0) / 2.0  # map [-1,1] -> [0,1]

        # Reference Shapley
        if d <= MAX_EXACT_D:
            t_ref = time.time()
            phi_exact = exact_shapley_tn(ref_model, x_eval)
            exact_time = time.time() - t_ref
            ref_name = "exact"
        else:
            t_ref = time.time()
            phi_exact = efficient_shapley_tn(ref_model, x_eval, n_pts=128)
            exact_time = time.time() - t_ref
            ref_name = "efficient"

        # TN‑SHAP on SAME ref model
        t0 = time.time()
        phi_tn = tn_shap_path_selector(ref_model, x_eval)
        tn_time = time.time() - t0

        t0 = time.time()
        phi_tn_opt = tn_shap_optimized(ref_model, x_eval)
        tn_opt_time = time.time() - t0

        cos_tn = safe_cosine_sim(phi_tn, phi_exact)
        cos_tn_opt = safe_cosine_sim(phi_tn_opt, phi_exact)

        # Optional diagnostic: student TN‑SHAP vs φ_exact(ref)
        t0 = time.time()
        phi_tn_student = tn_shap_optimized(student, x_eval)
        tn_student_time = time.time() - t0
        cos_tn_student = safe_cosine_sim(phi_tn_student, phi_exact)

        print(f"  TN‑SHAP orig: {tn_time*1000:.2f} ms, cos={cos_tn:.6f}")
        print(f"  TN‑SHAP opt : {tn_opt_time*1000:.2f} ms, cos={cos_tn_opt:.6f}")
        print(f"  TN‑SHAP stud: {tn_student_time*1000:.2f} ms, cos={cos_tn_student:.6f}")
        print(f"  {ref_name.capitalize()} ref: {exact_time*1000:.2f} ms")
        print(f"  Speedups vs ref: orig={exact_time/tn_time:.1f}x, opt={exact_time/tn_opt_time:.1f}x")

        results.append({
            "test_idx": i,
            "phi_exact": phi_exact.tolist(),
            "phi_tn": phi_tn.tolist(),
            "phi_tn_opt": phi_tn_opt.tolist(),
            "phi_tn_student": phi_tn_student.tolist(),
            "tn_time_ms": tn_time * 1000.0,
            "tn_opt_time_ms": tn_opt_time * 1000.0,
            "tn_student_time_ms": tn_student_time * 1000.0,
            "exact_time_ms": exact_time * 1000.0,
            "cosine_tn": cos_tn,
            "cosine_tn_opt": cos_tn_opt,
            "cosine_tn_student": cos_tn_student,
            "ref_kind": ref_name,
        })

    # Aggregate
    tn_times = [r["tn_time_ms"] for r in results]
    tn_opt_times = [r["tn_opt_time_ms"] for r in results]
    exact_times = [r["exact_time_ms"] for r in results]
    tn_speedups = [et / (tt / 1000.0) for et, tt in zip(exact_times, tn_times)]
    tn_opt_speedups = [et / (tt / 1000.0) for et, tt in zip(exact_times, tn_opt_times)]

    out = {
        "dimension": d,
        "teacher_rank": RANK_GT,
        "student_rank": RANK_STUDENT,
        "train_r2": train_stats["train_r2"],
        "train_time_s": train_time,
        "mean_tn_time_ms": float(np.mean(tn_times)),
        "mean_tn_opt_time_ms": float(np.mean(tn_opt_times)),
        "mean_exact_time_ms": float(np.mean(exact_times)),
        "mean_tn_speedup": float(np.mean(tn_speedups)),
        "mean_tn_opt_speedup": float(np.mean(tn_opt_speedups)),
        "mean_cosine_tn": float(np.mean([r["cosine_tn"] for r in results])),
        "mean_cosine_tn_opt": float(np.mean([r["cosine_tn_opt"] for r in results])),
        "mean_cosine_tn_student": float(np.mean([r["cosine_tn_student"] for r in results])),
        "ref_kind": results[0]["ref_kind"] if results else "unknown",
        "individual_results": results,
    }
    return out


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Run TN-SHAP dimension sweep.")
    parser.add_argument("--dims", type=str, default="10,20,30,40,50",
                        help="Comma-separated list of dimensions, e.g., 10,20,30")
    parser.add_argument("--outdir", type=str, default="out_d_sweep",
                        help="Output directory for CSV/JSONL")
    parser.add_argument("--compare", type=str, choices=["teacher", "student"], default="teacher",
                        help="Compare TN-SHAP against 'teacher' or 'student' reference")
    args = parser.parse_args()

    global COMPARE_AGAINST
    COMPARE_AGAINST = args.compare

    dims = [int(x.strip()) for x in args.dims.split(",") if x.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Run sweep
    all_rows: List[Dict] = []
    jsonl_path = outdir / "per_test_details.jsonl"
    csv_path = outdir / "summary.csv"

    with open(jsonl_path, "w", encoding="utf-8") as fjson:
        for d in dims:
            res = evaluate_dimension(d)
            # write details line-by-line
            fjson.write(json.dumps(res) + "\n")

            # row for CSV
            row = {
                "dimension": d,
                "train_r2": res["train_r2"],
                "mean_tn_time_ms": res["mean_tn_time_ms"],
                "mean_tn_opt_time_ms": res["mean_tn_opt_time_ms"],
                "mean_exact_time_ms": res["mean_exact_time_ms"],
                "mean_tn_speedup_x": res["mean_tn_speedup"],
                "mean_tn_opt_speedup_x": res["mean_tn_opt_speedup"],
                "mean_cosine_tn": res["mean_cosine_tn"],
                "mean_cosine_tn_opt": res["mean_cosine_tn_opt"],
                "mean_cosine_tn_student": res["mean_cosine_tn_student"],
                "ref_kind": res["ref_kind"],
            }
            all_rows.append(row)

    # Write CSV
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Pretty print summary
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY")
    print("=" * 100)
    hdr = f"{'Dim':<6}{'R^2':>10}{'TN(ms)':>12}{'TNopt(ms)':>12}{'Exact(ms)':>12}{'SpdUp':>10}{'SpdUp(opt)':>12}{'cos(TN)':>10}{'cos(TNo)':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        print(f"{r['dimension']:<6}{r['train_r2']:>10.4f}{r['mean_tn_time_ms']:>12.2f}{r['mean_tn_opt_time_ms']:>12.2f}"
              f"{r['mean_exact_time_ms']:>12.2f}{r['mean_tn_speedup_x']:>10.1f}{r['mean_tn_opt_speedup_x']:>12.1f}"
              f"{r['mean_cosine_tn']:>10.3f}{r['mean_cosine_tn_opt']:>10.3f}")

    print(f"\nWrote per-test details to: {jsonl_path}")
    print(f"Wrote summary CSV to     : {csv_path}")


if __name__ == "__main__":
    main()
