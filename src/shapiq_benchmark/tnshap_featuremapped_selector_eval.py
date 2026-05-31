#!/usr/bin/env python3
"""
TN-SHAP via Gi(t; x) using *selectors in feature space* for FeatureMappedTN.

We assume:
  - You trained a FeatureMappedTN with fmap_out_dim = 1 (+ 1 bias channel),
    using `train_featuremapped_tn_shapiq_benchmark.py`.
  - The checkpoint contains hyperparams: rank, fmap_out_dim, fmap_hidden, etc.

This script:
  1) Loads the FeatureMappedTN checkpoint.
  2) Rebuilds the model with selector_mode='per_feature_scalar'.
  3) Implements Gi(t; x) via selectors (thin diagonal in feature-map space):
       - j ≠ i: selector = t
       - j = i, "on":  selector = 1
       - j = i, "off": selector = 0
     where feature map ψ satisfies ψ(0) = 0, so selectors=0 corresponds to baseline.
  4) Uses Vandermonde interpolation over t to recover coefficients m_s^(i),
     then Shapley_i(x) ≈ Σ_s α_s m_s^(i).

You can either:
  - import tnshap_gi_selector_surrogate() and call it from your code, or
  - use the CLI here to test on random coalitions or custom x.
"""

import argparse
import math
import os
from typing import Optional

import numpy as np
import torch

from feature_mapped_tn import make_feature_mapped_tn, FeatureMappedTN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Small utilities
# -----------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shapley_weights(D: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """
    Shapley weights α_s for s = 0,...,D-1:

        α_s = s!(D-s-1)! / D!
    """
    if device is None:
        device = DEVICE
    alphas = torch.empty(D, dtype=dtype, device=device)
    D_fact = math.factorial(D)
    for s in range(D):
        num = math.factorial(s) * math.factorial(D - s - 1)
        alphas[s] = num / D_fact
    return alphas


def chebyshev_nodes_unit_interval(n: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """
    Chebyshev nodes mapped to [0,1]:

        t_k = 0.5 * (1 - cos((2k+1)/(2n) * pi)),  k = 0,...,n-1
    """
    if device is None:
        device = DEVICE
    k = torch.arange(n, device=device, dtype=dtype)
    t = 0.5 * (1.0 - torch.cos((2.0 * k + 1.0) * math.pi / (2.0 * n)))
    return t


def build_vandermonde(t_nodes: torch.Tensor, degree_max: int) -> torch.Tensor:
    """
    V[l, r] = t_nodes[l] ** r,  r = 0,...,degree_max.

    t_nodes: [L] (float64)
    returns: [L, degree_max + 1]
    """
    t = t_nodes.to(dtype=torch.float64)
    exps = torch.arange(0, degree_max + 1, dtype=torch.float64, device=t.device)
    V = t.unsqueeze(1) ** exps.unsqueeze(0)  # [L, degree_max+1]
    return V


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    var = torch.var(y_true)
    if var < 1e-12:
        return 1.0 if torch.allclose(y_true, y_pred) else 0.0
    return float(1.0 - torch.mean((y_true - y_pred) ** 2) / (var + 1e-12))


# -----------------------
# Load FeatureMappedTN with selectors enabled
# -----------------------

def load_featuremapped_tn_with_selectors(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> FeatureMappedTN:
    """
    Load a FeatureMappedTN checkpoint and rebuild the model with
    selector_mode='per_feature_scalar' to enable thin-diagonal selectors
    in feature-map space.

    Assumes checkpoint dict was created by train_featuremapped_tn_shapiq_benchmark.py.
    """
    if device is None:
        device = DEVICE

    ckpt = torch.load(ckpt_path, map_location=device)

    n_players = ckpt["n_players"]
    rank = ckpt["rank"]
    fmap_out_dim = ckpt["fmap_out_dim"]
    fmap_hidden = ckpt["fmap_hidden"]
    use_polynomial_features = ckpt["use_polynomial_features"]

    # Rebuild with selector_mode='per_feature_scalar'
    model = make_feature_mapped_tn(
        d_in=n_players,
        fmap_out_dim=fmap_out_dim,
        ranks=rank,
        out_dim=1,
        fmap_hidden=fmap_hidden,
        fmap_act="relu",
        use_polynomial_features=use_polynomial_features,
        use_log_scale=False,
        selector_mode="per_feature_scalar",  # <<< enable selectors
        seed=None,
        device=device,
        dtype=torch.float32,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# -----------------------
# Gi(t; x) in *feature space* via selectors
# -----------------------

def eval_Gi_selector_surrogate(
    model: FeatureMappedTN,
    x: torch.Tensor,            # [D] or [1, D] with entries in {0,1}
    i: int,
    t_nodes: torch.Tensor,      # [L]
) -> torch.Tensor:
    """
    Compute G_i(t_ℓ; x) using selectors in feature space:

        G_i(t; x) = f_on(t; x) - f_off(t; x),

    where:
      - For all j ≠ i, selector_j(t) = t
      - For j = i:
          * on-path:  selector_i(t) = 1
          * off-path: selector_i(t) = 0

    With ψ(0) = 0, selectors=0 corresponds to baseline,
    and selectors=1 corresponds to "fully on" feature.

    Args:
        model: FeatureMappedTN with selector_mode='per_feature_scalar'
        x:     [D] or [1,D], on the same device as model
        i:     feature index (0 ≤ i < D)
        t_nodes: [L], interpolation nodes in [0,1]

    Returns:
        h: [L] tensor with Gi(t_ℓ; x).
    """
    device = x.device
    t_nodes = t_nodes.to(device=device, dtype=torch.float32)

    if x.ndim == 1:
        x = x.unsqueeze(0)  # [1, D]
    assert x.ndim == 2
    B, D = x.shape
    assert B == 1, "Gi eval currently assumes a single point x; call in a loop otherwise."
    assert 0 <= i < D

    L = t_nodes.shape[0]

    # Repeat x for each t node: [L, D]
    x_batch = x.repeat(L, 1)

    # Base selectors: j ≠ i → t, same t across all features for a given row
    t_col = t_nodes.view(L, 1)          # [L,1]
    selectors_base = t_col.repeat(1, D) # [L, D]

    # ON path: feature i selector=1, others=t
    selectors_on = selectors_base.clone()
    selectors_on[:, i] = 1.0

    # OFF path: feature i selector=0, others=t
    selectors_off = selectors_base.clone()
    selectors_off[:, i] = 0.0

    with torch.no_grad():
        y_on = model(x_batch, selectors_on).squeeze(-1)   # [L]
        y_off = model(x_batch, selectors_off).squeeze(-1) # [L]

    h = y_on - y_off
    return h  # [L]


# -----------------------
# TN-SHAP via Gi(t; x) + Vandermonde
# -----------------------

def tnshap_gi_selector_surrogate(
    model: FeatureMappedTN,
    x: torch.Tensor,                # [D]
    max_degree: int,
    t_nodes: torch.Tensor,          # [L], L ≥ max_degree+1
) -> torch.Tensor:
    """
    TN-SHAP approximation for FeatureMappedTN via Gi(t; x) with selectors.

    Steps:
      1) For each feature i:
           - Evaluate Gi(t_ℓ; x) on t_ℓ (ℓ=0..L-1).
           - Restrict to first max_degree+1 nodes for polynomial degree ≤ max_degree.
      2) Solve Vandermonde system V m^(i) ≈ h_i for m^(i) coefficients.
      3) Approximate Shapley_i(x) ≈ Σ_{s=0}^{max_degree} α_s m_s^(i),
         where α_s are Shapley weights for total dimension D.

    Args:
        model: FeatureMappedTN with selector_mode='per_feature_scalar'.
        x: [D] input point (e.g., coalition in {0,1}^D).
        max_degree: maximum degree of interpolation in t.
        t_nodes: interpolation nodes in [0,1], len(t_nodes) ≥ max_degree+1.

    Returns:
        phi: [D] tensor of approximate Shapley values under this surrogate.
    """
    device = x.device
    x = x.to(device=device, dtype=torch.float32)

    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor [D]")

    D = x.shape[0]
    assert t_nodes.shape[0] >= max_degree + 1, "Need at least max_degree+1 t-nodes."

    # Use first max_degree+1 t nodes
    t_sub = t_nodes[: max_degree + 1].to(device=device, dtype=torch.float64)
    V = build_vandermonde(t_sub, degree_max=max_degree)  # [L, max_degree+1]
    alphas = shapley_weights(D, device=device, dtype=torch.float64)  # [D]

    phi = torch.zeros(D, device=device, dtype=torch.float32)

    for i in range(D):
        # 1) Collect Gi(t; x) on these nodes (float32 → float64 for interpolation)
        h = eval_Gi_selector_surrogate(model, x, i, t_sub.to(device=device, dtype=torch.float32))  # [L]
        h64 = h.to(torch.float64)

        # 2) Solve V m^(i) ≈ h via lstsq
        m_i, *_ = torch.linalg.lstsq(V, h64.unsqueeze(-1))  # [max_degree+1, 1]
        m_i = m_i.squeeze(-1)  # [max_degree+1]

        # 3) Shapley_i ≈ Σ_{s=0}^{max_degree} α_s m_s^(i)
        phi_i = torch.sum(alphas[: max_degree + 1] * m_i)
        phi[i] = phi_i.to(torch.float32)

    return phi  # [D]


# -----------------------
# Simple CLI for testing
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="TN-SHAP via Gi(t; x) using selectors in feature space for FeatureMappedTN."
    )
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to FeatureMappedTN checkpoint (from train_featuremapped_tn_shapiq_benchmark.py).")
    p.add_argument("--max-degree", type=int, default=None,
                   help="Max polynomial degree in t. Default: D-1 (full degree).")
    p.add_argument("--n_t_nodes", type=int, default=None,
                   help="Number of Chebyshev nodes in [0,1]. Default: max_degree+1.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for sampling a test coalition.")
    p.add_argument("--coalition", type=str, default=None,
                   help="Optional explicit coalition as a bitstring, e.g. '101001'. Overrides random.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1) Load model with selectors enabled
    model = load_featuremapped_tn_with_selectors(args.ckpt, device=DEVICE)
    D = model.d_in  # number of features

    print(f"[INFO] Loaded FeatureMappedTN with selectors from {args.ckpt}")
    print(f"[INFO] d_in = {D}")

    # 2) Choose x (coalition) ∈ {0,1}^D
    if args.coalition is not None:
        bits = args.coalition.strip()
        if len(bits) != D:
            raise ValueError(f"coalition bitstring length ({len(bits)}) != d_in ({D})")
        x_np = np.array([1.0 if c == "1" else 0.0 for c in bits], dtype=np.float32)
    else:
        # random coalition with ~Bernoulli(0.5)
        x_np = (np.random.rand(D) < 0.5).astype(np.float32)

    x = torch.from_numpy(x_np).to(DEVICE)  # [D]

    print(f"[INFO] Using coalition x (first 20): {x.cpu().numpy()[:20]}")

    # 3) t-nodes and degree
    if args.max_degree is None:
        max_degree = D - 1
    else:
        max_degree = int(args.max_degree)
    if args.n_t_nodes is None:
        n_t_nodes = max_degree + 1
    else:
        n_t_nodes = int(args.n_t_nodes)
        if n_t_nodes < max_degree + 1:
            raise ValueError("n_t_nodes must be at least max_degree+1.")

    t_nodes = chebyshev_nodes_unit_interval(n_t_nodes, device=DEVICE, dtype=torch.float64)

    print(f"[INFO] max_degree = {max_degree}, n_t_nodes = {n_t_nodes}")

    # 4) Compute TN-SHAP via Gi(t; x) with selectors
    phi = tnshap_gi_selector_surrogate(
        model=model,
        x=x,
        max_degree=max_degree,
        t_nodes=t_nodes,
    )

    print("[RESULT] TN-SHAP φ(x) via selectors (first 20):")
    print(phi.detach().cpu().numpy()[:20])


if __name__ == "__main__":
    main()
