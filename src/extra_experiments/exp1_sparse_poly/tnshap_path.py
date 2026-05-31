# src/extra_experiments/exp1_sparse_poly/tnshap_path.py

import math
import time
from itertools import combinations
from typing import Optional, Tuple

import numpy as np
import torch


# Simple device helper (use model / x device if possible)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def tn_selector_k1_sharedgrid(
    model: torch.nn.Module,
    x: np.ndarray,
    t_nodes: np.ndarray,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, dict]:
    """
    TN-Shap style selector-path estimator for *first-order* interactions (k=1),
    using a single grid t_nodes on [0, 1].

    This is a stripped-down version of tn_selector_any_k_sharedgrid from your
    main TN-SHAP code, specialized to k=1.
    """
    device = device or DEVICE
    x = np.asarray(x, np.float32).ravel()
    d = x.size
    k = 1

    # Grid + Vandermonde inverse
    t = torch.tensor(np.asarray(t_nodes, np.float32), device=device)   # [m]
    m = int(t.numel())
    V = torch.vander(t, N=m, increasing=True)  # [m, m]

    _sync(); t0 = time.perf_counter()
    Vinv = torch.linalg.inv(V)                # [m, m]
    _sync(); t_solve = time.perf_counter() - t0

    x_t = torch.tensor(x, dtype=torch.float32, device=device)  # [d]

    # Global path (S = empty)
    Xg = t.unsqueeze(1) * x_t.unsqueeze(0)    # [m, d]

    _sync(); t0 = time.perf_counter()
    yg = model(Xg).squeeze(-1)               # [m]
    _sync(); t_eval = time.perf_counter() - t0

    c_empty = (Vinv @ yg.unsqueeze(1)).squeeze(1)  # [m]
    coeffs = {(): c_empty.detach().cpu().numpy().astype(np.float64)}

    # Masked paths for all singletons S = {i}
    subs = [(i,) for i in range(d)]
    if subs:
        Xh = Xg.repeat(len(subs), 1)         # [(d*m), d]
        for r, S in enumerate(subs):
            i = S[0]
            Xh[r*m:(r+1)*m, i] = 0.0        # mask feature i along the path

        _sync(); t0 = time.perf_counter()
        yh = model(Xh).squeeze(-1).view(len(subs), m)  # [d, m]
        _sync(); t_eval += time.perf_counter() - t0

        _sync(); t0 = time.perf_counter()
        cS = (yh @ Vinv.T)                            # [d, m]
        _sync(); t_solve += time.perf_counter() - t0

        for S, c in zip(subs, cS):
            coeffs[S] = c.detach().cpu().numpy().astype(np.float64)

    # For k=1: all T are just singletons {i}
    all_T = [(i,) for i in range(d)]
    phi = np.zeros(len(all_T), dtype=np.float64)

    # weights[r] = 1 / C(r, 1) = 1 / r for r >= 1
    weights = np.zeros(m, dtype=np.float64)
    for r in range(k, m):
        weights[r] = 1.0 / max(r, 1)

    for idx_T, T in enumerate(all_T):
        # Inclusion–exclusion over S ⊆ T (just S = ∅ and S = T)
        # c^T = (+1)*c^(∅) + (-1)*c^(T)
        cT = np.zeros(m, dtype=np.float64)
        cT += coeffs[()]                      # S = empty, sign = (+1)
        cT -= coeffs[T]                       # S = T,     sign = (-1)

        # Integrate k=1 contribution from polynomial coefficients
        phi[idx_T] = float(np.dot(cT[k:], weights[k:]))

    timing = {
        "t_eval_s": float(t_eval),
        "t_solve_s": float(t_solve),
        "t_total_s": float(t_eval + t_solve),
    }
    return phi, timing


# in tnshap_path.py

from .masked_dataset import chebyshev_nodes_01

def tnshap_order1_path(
    model: torch.nn.Module,
    x: np.ndarray,
    m: int = 32,
    *,
    t_nodes: np.ndarray = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, dict]:
    """
    If t_nodes is given, use it; otherwise build Chebyshev grid with size m.
    """
    if t_nodes is None:
        t_nodes = chebyshev_nodes_01(m)
    phi, timing = tn_selector_k1_sharedgrid(model, x, t_nodes, device=device)
    return phi, timing

