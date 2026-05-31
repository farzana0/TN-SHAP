#!/usr/bin/env python3
"""
Compare TN-SHAP (FeatureMappedTN with selectors) to exact Shapley values
on a shapiq benchmark Game (e.g., AdultCensusDataValuation).

Pipeline:
  1) Load Game via shapiq.benchmark.load_games_from_configuration.
  2) Compute exact Shapley values φ_exact for the cooperative game v(S).
  3) Load FeatureMappedTN checkpoint (trained to approximate v(S)).
  4) Compute TN-SHAP φ_TN at x = 1^n via Gi(t; x) using selectors in feature space.
  5) Report correlation, MSE, and top-k support overlap between φ_exact and φ_TN.
"""

import argparse
import math
import os
from typing import Optional

import numpy as np
import torch

from shapiq.benchmark import load_games_from_configuration

from feature_mapped_tn import make_feature_mapped_tn, FeatureMappedTN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Small utilities (copied / adapted)
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
# Exact Shapley for shapiq Game
# -----------------------

def exact_shapley_from_game(game) -> np.ndarray:
    """
    Compute exact Shapley values for a shapiq Game by enumerating all coalitions.

    Game:
      - game.n_players = n
      - game(coalitions) accepts an array of shape [B, n] with bool/0-1 entries.

    Formula:
      φ_i = Σ_{S ⊆ N\{i}} [ |S|!(n-|S|-1)! / n! ] * [v(S ∪ {i}) - v(S)]
    """
    n = game.n_players
    num_coalitions = 1 << n  # 2^n

    # Enumerate all coalitions as bitmasks 0..2^n-1
    masks = np.arange(num_coalitions, dtype=np.int64)

    # Convert masks -> coalition matrix [2^n, n] with entries in {0,1}
    bits = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)

    # Evaluate v(S) for all S at once (precomputed Game makes this cheap)
    v_all = np.asarray(game(bits), dtype=np.float64).reshape(-1)

    N_fact = math.factorial(n)
    phi = np.zeros(n, dtype=np.float64)

    # For each player i, sum over subsets S not containing i
    for i in range(n):
        contrib = 0.0
        for mask in range(num_coalitions):
            if mask & (1 << i):  # skip subsets that already contain i
                continue
            s = mask.bit_count()
            weight = math.factorial(s) * math.factorial(n - s - 1) / N_fact
            v_S = v_all[mask]
            v_Si = v_all[mask | (1 << i)]
            contrib += weight * (v_Si - v_S)
        phi[i] = contrib

    return phi  # [n]


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
    x: torch.Tensor,            # [D] or [1, D]
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
    selectors=1 corresponds to "fully on".
    """
    device = x.device
    t_nodes = t_nodes.to(device=device, dtype=torch.float32)

    if x.ndim == 1:
        x = x.unsqueeze(0)  # [1, D]
    assert x.ndim == 2
    B, D = x.shape
    assert B == 1
    assert 0 <= i < D

    L = t_nodes.shape[0]

    # Repeat x for each t node: [L, D]
    x_batch = x.repeat(L, 1)

    # Base selectors: j ≠ i → t, same t across all features for a given row
    t_col = t_nodes.view(L, 1)             # [L,1]
    selectors_base = t_col.repeat(1, D)    # [L,D]

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


def tnshap_gi_selector_surrogate(
    model: FeatureMappedTN,
    x: torch.Tensor,                # [D]
    max_degree: int,
    t_nodes: torch.Tensor,          # [L], L ≥ max_degree+1
) -> torch.Tensor:
    """
    TN-SHAP approximation for FeatureMappedTN via Gi(t; x) with selectors.
    Returns φ(x) ∈ R^D.
    """
    device = x.device
    x = x.to(device=device, dtype=torch.float32)
    if x.ndim != 1:
        raise ValueError("x must be 1D [D]")

    D = x.shape[0]
    assert t_nodes.shape[0] >= max_degree + 1

    t_sub = t_nodes[: max_degree + 1].to(device=device, dtype=torch.float64)
    V = build_vandermonde(t_sub, degree_max=max_degree)  # [L, max_degree+1]
    alphas = shapley_weights(D, device=device, dtype=torch.float64)

    phi = torch.zeros(D, device=device, dtype=torch.float32)

    for i in range(D):
        h = eval_Gi_selector_surrogate(model, x, i, t_sub.to(device=device, dtype=torch.float32))  # [L]
        h64 = h.to(torch.float64)
        m_i, *_ = torch.linalg.lstsq(V, h64.unsqueeze(-1))
        m_i = m_i.squeeze(-1)  # [max_degree+1]
        phi_i = torch.sum(alphas[: max_degree + 1] * m_i)
        phi[i] = phi_i.to(torch.float32)

    return phi  # [D]


# -----------------------
# CLI & main
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare TN-SHAP (FeatureMappedTN + selectors) to exact Shapley "
                    "for a shapiq benchmark Game (AdultCensusDataValuation or SentimentAnalysisLocalXAI)."
    )
    p.add_argument(
        "--benchmark",
        type=str,
        choices=["adult_sv", "sentiment_si"],
        required=True,
        help=(
            "Which shapiq benchmark to use:\n"
            "  adult_sv     -> AdultCensusDataValuation (data valuation Shapley)\n"
            "  sentiment_si -> SentimentAnalysisLocalXAI (local explanation). "
            "For now, exact Shapley here is still computed as a cooperative game over players."
        ),
    )
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to FeatureMappedTN checkpoint (from train_featuremapped_tn_shapiq_benchmark.py).")
    p.add_argument("--max-degree", type=int, default=None,
                   help="Max polynomial degree in t (default: D-1).")
    p.add_argument("--n_t_nodes", type=int, default=None,
                   help="Number of Chebyshev nodes (default: max_degree+1).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ---------------- load Game ----------------
    if args.benchmark == "adult_sv":
        game_identifier = "AdultCensusDataValuation"
        config_id = 1
        n_player_id = 0
    elif args.benchmark == "sentiment_si":
        game_identifier = "SentimentAnalysisLocalXAI"
        config_id = 1
        n_player_id = 0
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    games = load_games_from_configuration(
        game_class=game_identifier,
        n_player_id=n_player_id,
        config_id=config_id,
        n_games=1,
    )
    games = list(games)
    if len(games) == 0:
        raise RuntimeError("No games loaded from configuration.")
    game = games[0]
    n_players = game.n_players

    print(f"[INFO] Loaded Game: {game}")
    print(f"[INFO] n_players = {n_players}")

    # ---------------- exact Shapley ----------------
    print("[INFO] Computing exact Shapley values by full enumeration...")
    phi_exact = exact_shapley_from_game(game)  # [n_players]
    print("[INFO] φ_exact (first 20):", phi_exact[:20])

    # ---------------- load surrogate ----------------
    print(f"[INFO] Loading FeatureMappedTN with selectors from {args.ckpt}")
    model = load_featuremapped_tn_with_selectors(args.ckpt, device=DEVICE)
    D = model.d_in
    assert D == n_players, f"Model d_in={D} but game.n_players={n_players}"

    # Evaluate TN-SHAP at x = 1^D (full coalition)
    x = torch.ones(D, device=DEVICE, dtype=torch.float32)

    # ---------------- t-nodes & degree ----------------
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

    # ---------------- TN-SHAP via selectors ----------------
    print("[INFO] Computing TN-SHAP via Gi(t; x=1^D) with selectors...")
    phi_tn = tnshap_gi_selector_surrogate(
        model=model,
        x=x,
        max_degree=max_degree,
        t_nodes=t_nodes,
    )  # [D]
    phi_tn_np = phi_tn.detach().cpu().numpy()
    print("[INFO] φ_TN (first 20):", phi_tn_np[:20])

    # ---------------- comparison ----------------
    # Align shapes
    phi_exact_t = torch.from_numpy(phi_exact).to(DEVICE, dtype=torch.float32)
    phi_tn_t    = torch.from_numpy(phi_tn_np).to(DEVICE, dtype=torch.float32)

    # Correlation
    if torch.std(phi_exact_t) < 1e-8 or torch.std(phi_tn_t) < 1e-8:
        corr = float("nan")
    else:
        corr = float(torch.corrcoef(torch.stack([phi_exact_t, phi_tn_t]))[0, 1].item())

    # R², MSE
    mse = torch.mean((phi_exact_t - phi_tn_t) ** 2).item()
    r2 = r2_score(phi_exact_t, phi_tn_t)

    # Top-k support overlap
    # k = number of players with |φ_exact| above median (or just k = n)
    k = n_players  # or choose something else (e.g., k = 5)
    top_exact = torch.topk(phi_exact_t.abs(), k).indices.tolist()
    top_tn    = torch.topk(phi_tn_t.abs(),    k).indices.tolist()
    overlap = len(set(top_exact) & set(top_tn)) / k

    print("==== Comparison TN-SHAP vs Exact Shapley ====")
    print(f"Correlation: {corr:.4f}")
    print(f"R²:          {r2:.4f}")
    print(f"MSE:         {mse:.4e}")
    print(f"Top-{k} overlap: {overlap:.4f}")
    print(f"Top-{k} indices exact: {top_exact}")
    print(f"Top-{k} indices TN:    {top_tn}")


if __name__ == "__main__":
    main()
