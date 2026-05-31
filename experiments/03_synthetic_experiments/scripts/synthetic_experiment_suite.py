#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# -*- coding: utf-8 -*-

"""
Multilinear-only Shapley experiments (orders 1–3) using your BinaryTensorTree.

Experiments:
  1) GT = low-rank TN (easy)         -> student TN (same rank)
  2) GT = higher-rank TN (harder)    -> student TN (same rank as #1, underfitting)
  3) GT = generic multilinear (<=3)  -> student TN

Computes Shapley interaction indices (SII) at a single anchor x0 via:
  - GT: exact enumeration (true v(S))
  - Student TN: selector-based batched coalition eval (v_hat(S))

Compares TN vs GT SII with R² for orders 1, 2, 3.
Saves: models, v-caches, SII per order as JSON, summary.

Requires: tntree_model.py in Python path.
"""

import os
import json
import math
import time
import itertools as it
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from src.tntree_model import make_balanced_binary_tensor_tree


# -------------------------- utils: reproducibility -------------------------- #

def set_seed(seed: int = 123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------- utils: data generation -------------------------- #

def sample_zero_mean(n: int, d: int) -> np.ndarray:
    """Draw n samples in R^d, zero-mean Gaussian."""
    return np.random.randn(n, d).astype(np.float32)


# --------------------------- GT: tensor-tree models -------------------------- #

def make_gt_tn(d: int, rank: int, seed: int = 11):
    """
    Build a ground-truth tensor tree (multilinear) with given rank.
    Leaf phys dim = 2 so each leaf is [value, bias].
    """
    tn = make_balanced_binary_tensor_tree(
        leaf_phys_dims=[2] * d,
        ranks=rank,
        out_dim=1,
        assume_bias_when_matrix=True,  # convenient for training/eval on matrix inputs
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return tn


# --------------------- GT: generic multilinear coefficients ------------------ #

def make_generic_multilinear_coeffs(d: int, max_order: int = 3, seed: int = 7) -> Dict[Tuple[int, ...], float]:
    """
    Make a generic multilinear function f(x) = sum_{|S|<=max_order} c_S * prod_{i in S} x_i
    with a few large interactions to help identifiability.
    """
    rng = np.random.default_rng(seed)
    coeffs: Dict[Tuple[int, ...], float] = {}

    # constant term zero for zero-baseline shapley
    coeffs[tuple()] = 0.0

    # linear terms (moderate)
    for i in range(d):
        coeffs[(i,)] = float(1.5 + 0.3 * rng.standard_normal())

    # some strong pairs
    strong_pairs = [(0, 1), (2, 3)]
    for (i, j) in it.combinations(range(d), 2):
        if (i, j) in strong_pairs:
            coeffs[(i, j)] = float(6.0 + rng.standard_normal())
        else:
            coeffs[(i, j)] = float(0.4 * rng.standard_normal())

    # a few strong triplets, rest small
    chosen_trips = [(0, 1, 2), (4, 5, 6)]
    for S in it.combinations(range(d), 3):
        if S in chosen_trips:
            coeffs[S] = float(8.0 + 0.8 * rng.standard_normal())
        else:
            coeffs[S] = float(0.25 * rng.standard_normal())

    # remove beyond max_order if asked
    for k in range(max_order + 1, d + 1):
        for S in it.combinations(range(d), k):
            coeffs[S] = 0.0

    return coeffs


def eval_generic_multilinear(coeffs: Dict[Tuple[int, ...], float], x: np.ndarray) -> float:
    """
    Evaluate f(x) from coeffs dict. Assumes coeffs[( )] is constant term.
    """
    total = 0.0
    d = x.shape[0]
    for k in range(0, d + 1):
        for S in it.combinations(range(d), k):
            c = coeffs.get(S, 0.0)
            if c == 0.0:
                continue
            prod = 1.0
            for i in S:
                prod *= x[i]
            total += c * prod
    return float(total)


# --------------------------- student TN: training ---------------------------- #

def train_student_tn(
    X: np.ndarray,
    y: np.ndarray,
    rank: int,
    seed: int = 13,
    lr: float = 5e-3,
    epochs: int = 1200,
    patience: int = 150,
    log_every: int = 80,
) -> Tuple[object, Dict[str, float]]:
    """
    Train a BinaryTensorTree on (X,y) using matrix input (assume_bias_when_matrix=True).
    Returns (model, logs).
    """
    d = X.shape[1]
    device = torch.device("cpu")

    model = make_balanced_binary_tensor_tree(
        leaf_phys_dims=[2] * d,
        ranks=rank,
        out_dim=1,
        assume_bias_when_matrix=True,
        seed=seed,
        device=device,
        dtype=torch.float32,
    )

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_r2 = -1e9
    best_state = None
    bad = 0
    logs = {}

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        pred = model(Xt).squeeze()
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred_np = pred.detach().cpu().numpy()
            r2 = r2_score(yt.detach().cpu().numpy(), pred_np)

        if r2 > best_r2 + 1e-6:
            best_r2 = r2
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch == 1 or epoch % log_every == 0:
            print(f"[epoch {epoch:4d}] train MSE={loss.item():.6f}  R2={r2:.4f}  (best R2={best_r2:.4f})")

        if bad >= patience:
            break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    logs["train_r2"] = float(best_r2)
    return model, logs


# ---------------------- coalition batching (selectors) ----------------------- #

def all_masks(d: int) -> np.ndarray:
    """
    Return [2^d, d] boolean mask matrix; row s has mask for subset S (bit order: 0..d-1).
    """
    B = 1 << d
    m = np.zeros((B, d), dtype=np.float32)
    for s in range(B):
        for i in range(d):
            if (s >> i) & 1:
                m[s, i] = 1.0
    return m


def tn_v_cache_all_coalitions(tn, x: np.ndarray) -> Dict[int, float]:
    """
    Evaluate TN at x_S for all coalitions S via selector inputs, in a single forward pass.
    Returns dict: mask_int -> value.
    """
    d = x.shape[0]
    masks = all_masks(d)  # [B,d]
    B = masks.shape[0]

    # Per-leaf inputs: [B, 2] with [value_or_zero, 1]
    per_leaf = []
    for i in range(d):
        v = (masks[:, i] * x[i]).astype(np.float32)  # [B]
        leaf = np.stack([v, np.ones(B, dtype=np.float32)], axis=1)  # [B,2]
        per_leaf.append(torch.tensor(leaf, dtype=torch.float32))

    with torch.no_grad():
        y = tn(per_leaf).detach().cpu().numpy().reshape(-1)

    return {s: float(y[s]) for s in range(B)}


def generic_v_cache_all_coalitions(coeffs: Dict[Tuple[int, ...], float], x: np.ndarray) -> Dict[int, float]:
    """
    Evaluate generic multilinear f(x_S) for all coalitions S by masking x and calling eval.
    """
    d = x.shape[0]
    B = 1 << d
    out = {}
    for s in range(B):
        xS = np.zeros_like(x)
        for i in range(d):
            if (s >> i) & 1:
                xS[i] = x[i]
        out[s] = float(eval_generic_multilinear(coeffs, xS))
    return out


# ------------------------ Shapley interactions from v ------------------------ #

def shapley_order_k_from_cache(v_cache: Dict[int, float], d: int, k: int) -> Dict[Tuple[int, ...], float]:
    """
    General Shapley interaction index (SII) for order k from cached coalition values v(T).
    SII(S) = sum_{T subset N\S} w(|T|) * Δ_S v(T),
      where Δ_S v(T) = sum_{L subset S} (-1)^{k - |L|} v(T ∪ L),
      and w(t) = t! * (d-k-t)! / (d-k+1)!.
    Returns dict mapping tuple S (sorted) -> value.
    """
    assert 1 <= k <= d
    N_mask = (1 << d) - 1
    fact = [math.factorial(i) for i in range(d + 1)]

    def weight(t: int) -> float:
        return fact[t] * fact[d - k - t] / fact[d - k + 1]

    results: Dict[Tuple[int, ...], float] = {}

    for S_tuple in it.combinations(range(d), k):
        S_mask = 0
        for i in S_tuple:
            S_mask |= (1 << i)

        rem = N_mask ^ S_mask
        total = 0.0

        # iterate T subset of N \ S
        T = rem
        seen = set()
        while True:
            if T in seen:
                break
            seen.add(T)

            t = int(bin(T).count("1"))

            # Δ_S v(T) = sum_{L subset S} (-1)^{k - |L|} v(T ∪ L)
            delta = 0.0
            L_mask = S_mask
            seenL = set()
            while True:
                if L_mask in seenL:
                    break
                seenL.add(L_mask)

                l = int(bin(L_mask).count("1"))
                sign = (-1.0) ** (k - l)
                TL = T | L_mask
                delta += sign * v_cache[TL]

                if L_mask == 0:
                    break
                L_mask = (L_mask - 1) & S_mask  # next submask of S

            total += weight(t) * delta

            if T == 0:
                break
            T = (T - 1) & rem  # next submask of rem

        results[S_tuple] = float(total)

    return results


def shapley_all_orders_1to3(v_cache: Dict[int, float], d: int) -> Dict[int, Dict[Tuple[int, ...], float]]:
    """
    Convenience: compute order 1,2,3 Shapley interactions.
    """
    out: Dict[int, Dict[Tuple[int, ...], float]] = {}
    for k in (1, 2, 3):
        out[k] = shapley_order_k_from_cache(v_cache, d, k)
    return out


def r2_per_order(gt: Dict[int, Dict[Tuple[int, ...], float]],
                 est: Dict[int, Dict[Tuple[int, ...], float]]) -> Dict[str, float]:
    out = {}
    for k in (1, 2, 3):
        keys = sorted(gt[k].keys())
        y = np.array([gt[k][S] for S in keys], dtype=np.float64)
        yhat = np.array([est[k][S] for S in keys], dtype=np.float64)
        out[f"order_{k}"] = float(r2_score(y, yhat))
    return out


# ------------------------------ experiment core ------------------------------ #

def run_exp_gt_tn(
    name: str,
    d: int,
    rank_gt: int,
    rank_student: int,
    n_train: int = 20000,
    seed: int = 1234,
    outdir: str = "out_tn_vs_tn",
):
    """
    GT is a tensor-tree (rank_gt). Train student TN (rank_student). Compute SII up to 3.
    """
    print(f"\n=== EXP: {name} ===")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # Build GT TN and freeze (random initialized GT is fine; it defines a multilinear function)
    tn_gt = make_gt_tn(d, rank_gt, seed=seed + 1)

    # Training data
    X = sample_zero_mean(n_train, d)
    with torch.no_grad():
        y = tn_gt(torch.tensor(X)).squeeze().detach().cpu().numpy()

    # Train student
    print(f"[train] student rank={rank_student}")
    tn_student, logs = train_student_tn(X, y, rank=rank_student, seed=seed + 2)

    # Save models
    torch.save(tn_gt.state_dict(), os.path.join(outdir, f"{name}_gt_tn.pt"))
    torch.save(tn_student.state_dict(), os.path.join(outdir, f"{name}_student_tn.pt"))

    # Anchor x0 for SII
    x0 = sample_zero_mean(1, d)[0].astype(np.float32)
    print(f"[{name}] SII anchor x0 = {np.array2string(x0, precision=4, floatmode='fixed')}")

    # v-caches via selectors (GT and student)
    v_gt = tn_v_cache_all_coalitions(tn_gt, x0)
    v_st = tn_v_cache_all_coalitions(tn_student, x0)

    # Shapley interactions up to order 3
    sii_gt = shapley_all_orders_1to3(v_gt, d)
    sii_st = shapley_all_orders_1to3(v_st, d)

    r2s = r2_per_order(sii_gt, sii_st)
    print(f"R2 per order (student TN vs GT TN): {r2s}")

    # Save artifacts
    with open(os.path.join(outdir, f"{name}_x0.json"), "w") as f:
        json.dump({"x0": x0.tolist()}, f, indent=2)

    for k in (1, 2, 3):
        with open(os.path.join(outdir, f"{name}_sii_gt_order{k}.json"), "w") as f:
            json.dump({str(k_): v for k_, v in sii_gt[k].items()}, f, indent=2)
        with open(os.path.join(outdir, f"{name}_sii_student_order{k}.json"), "w") as f:
            json.dump({str(k_): v for k_, v in sii_st[k].items()}, f, indent=2)

    with open(os.path.join(outdir, f"{name}_summary.json"), "w") as f:
        json.dump({"train_r2": logs["train_r2"], "r2_per_order": r2s}, f, indent=2)


def run_exp_gt_generic(
    name: str,
    d: int,
    rank_student: int,
    n_train: int = 20000,
    seed: int = 777,
    outdir: str = "out_generic",
):
    """
    GT is a generic multilinear function up to order 3.
    Train student TN (rank_student). Compute SII up to order 3.
    """
    print(f"\n=== EXP: {name} ===")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # Build coeffs
    coeffs = make_generic_multilinear_coeffs(d=d, max_order=3, seed=seed + 1)

    # Training data
    X = sample_zero_mean(n_train, d)
    y = np.array([eval_generic_multilinear(coeffs, x) for x in X], dtype=np.float32)

    # Train student
    print(f"[train] student rank={rank_student}")
    tn_student, logs = train_student_tn(X, y, rank=rank_student, seed=seed + 2)

    # Save student & coeffs
    torch.save(tn_student.state_dict(), os.path.join(outdir, f"{name}_student_tn.pt"))
    with open(os.path.join(outdir, f"{name}_coeffs.json"), "w") as f:
        json.dump({str(k): float(v) for k, v in coeffs.items()}, f, indent=2)

    # Anchor x0
    x0 = sample_zero_mean(1, d)[0].astype(np.float32)
    print(f"[{name}] SII anchor x0 = {np.array2string(x0, precision=4, floatmode='fixed')}")

    # v-caches
    v_gt = generic_v_cache_all_coalitions(coeffs, x0)
    v_st = tn_v_cache_all_coalitions(tn_student, x0)

    # SIIs up to 3
    sii_gt = shapley_all_orders_1to3(v_gt, d)
    sii_st = shapley_all_orders_1to3(v_st, d)

    r2s = r2_per_order(sii_gt, sii_st)
    print(f"R2 per order (student TN vs GT generic): {r2s}")

    # Save artifacts
    with open(os.path.join(outdir, f"{name}_x0.json"), "w") as f:
        json.dump({"x0": x0.tolist()}, f, indent=2)

    for k in (1, 2, 3):
        with open(os.path.join(outdir, f"{name}_sii_gt_order{k}.json"), "w") as f:
            json.dump({str(k_): v for k_, v in sii_gt[k].items()}, f, indent=2)
        with open(os.path.join(outdir, f"{name}_sii_student_order{k}.json"), "w") as f:
            json.dump({str(k_): v for k_, v in sii_st[k].items()}, f, indent=2)

    with open(os.path.join(outdir, f"{name}_summary.json"), "w") as f:
        json.dump({"train_r2": logs["train_r2"], "r2_per_order": r2s}, f, indent=2)


# ----------------------------------- main ----------------------------------- #

def main():
    t0 = time.time()
    d = 8

    # 1) Easy: GT low-rank TN, student same rank
    run_exp_gt_tn(
        name="tn_gt_rank3_student_rank3",
        d=d,
        rank_gt=3,
        rank_student=3,
        n_train=20000,
        seed=101,
        outdir="out_tn_easy",
    )

    # 2) Harder: GT higher rank, student keeps same (underfitting)
    run_exp_gt_tn(
        name="tn_gt_rank16_student_rank3",
        d=d,
        rank_gt=16,
        rank_student=3,  # same as in (1)
        n_train=30000,
        seed=202,
        outdir="out_tn_hard",
    )

    # 3) Generic multilinear (<=3)
    run_exp_gt_generic(
        name="generic_order_leq3_student_rank12",
        d=d,
        rank_student=12,
        n_train=30000,
        seed=303,
        outdir="out_generic",
    )

    print(f"\nDone in {(time.time()-t0):.1f}s")


if __name__ == "__main__":
    main()
