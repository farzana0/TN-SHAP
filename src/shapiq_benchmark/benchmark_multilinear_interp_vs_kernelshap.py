#!/usr/bin/env python3
"""
Benchmark: Multilinear-extension-style Shapley via Gi(t; x) interpolation
vs. a simple permutation-based KernelSHAP-style estimator,
using only the GROUND-TRUTH game (no TN training, no MLP).

What this script does:

1) Load a shapiq benchmark Game (AdultCensusDataValuation or SentimentAnalysisLocalXAI).
2) Compute exact Shapley values φ_exact via full 2^n enumeration.
3) For N_POINTS random coalitions x ∈ {0,1}^D:
     - Define paths in INPUT space:
         x_on_i(t)[j]  = x[j]      if j == i, else t * x[j]
         x_off_i(t)[j] = 0         if j == i, else t * x[j]
     - For each feature i:
           G_i(t; x) = v(x_on_i(t)) - v(x_off_i(t))
       Evaluate G_i at Chebyshev nodes t_ℓ, fit a polynomial in t via Vandermonde
       and derive an approximate Shapley:
           Shapley_i(x) ≈ Σ_s α_s m_s^(i)
       where α_s are the standard Shapley weights and m_s^(i) are polynomial coefficients.
     - Compare φ_interp(x) to φ_exact:
           * Pearson correlation
           * top-k (default k=3) overlap on |φ|.
   Aggregate mean ± std over the N_POINTS coalitions.
   Also track how many game queries this interpolation used.

4) With approximately the SAME game-query budget, run a simple
   permutation-based "KernelSHAP-style" estimator multiple times:
     - φ_KS (global Shapley estimate, independent of x).
     - Compare φ_KS to φ_exact:
           * correlation
           * top-k overlap
       Report mean ± std over n_ks_runs.

⚠ NOTE:
   This script assumes the game can be evaluated on continuous inputs X ∈ [0,1]^D.
   If your shapiq Game only supports boolean coalitions, you'll need to replace
   'game(X)' with your own continuous function (teacher) that implements v(x).
"""

import argparse
import math
import time
from typing import Tuple

import numpy as np
import torch
from shapiq.benchmark import load_games_from_configuration


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Utilities
# -----------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_random_coalitions(
    n_players: int,
    n_coalitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample random coalitions as {0,1}^n.

    Returns:
        coalitions: np.ndarray [n_coalitions, n_players], dtype=bool
    """
    coalitions = rng.integers(
        low=0,
        high=2,
        size=(n_coalitions, n_players),
        dtype=np.int8,
    )
    return coalitions.astype(bool)


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


# -----------------------
# Exact Shapley for shapiq Game
# -----------------------

def exact_shapley_from_game(game) -> np.ndarray:
    """
    Compute exact Shapley values for a shapiq Game by enumerating all coalitions.

    Game:
      - game.n_players = n
      - game(coalitions) accepts array [B, n] with bool/0-1 entries.

    Formula:
      φ_i = Σ_{S ⊆ N\{i}} [ |S|!(n-|S|-1)! / n! ] * [v(S ∪ {i}) - v(S)]
    """
    n = game.n_players
    num_coalitions = 1 << n  # 2^n

    masks = np.arange(num_coalitions, dtype=np.int64)
    # coalition matrix [2^n, n] with entries in {0,1}
    bits = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)

    # Evaluate v(S) for all S
    v_all = np.asarray(game(bits), dtype=np.float64).reshape(-1)

    N_fact = math.factorial(n)
    phi = np.zeros(n, dtype=np.float64)

    for i in range(n):
        contrib = 0.0
        for mask in range(num_coalitions):
            if mask & (1 << i):  # S must not contain i
                continue
            s = mask.bit_count()
            weight = math.factorial(s) * math.factorial(n - s - 1) / N_fact
            v_S = v_all[mask]
            v_Si = v_all[mask | (1 << i)]
            contrib += weight * (v_Si - v_S)
        phi[i] = contrib

    return phi  # [n]


# -----------------------
# Gi(t; x) and interpolation on the GROUND-TRUTH game
# -----------------------

def eval_Gi_game_interp(
    game,
    x: torch.Tensor,        # [D] in {0,1}
    i: int,
    t_nodes: torch.Tensor,  # [L]
) -> Tuple[torch.Tensor, int]:
    """
    Compute G_i(t_ℓ; x) for the GROUND-TRUTH game using INPUT-space paths:

        x_on_i(t)[j]  = x[j]      if j == i, else t * x[j]
        x_off_i(t)[j] = 0         if j == i, else t * x[j]

    Assumes:
      - The game can be evaluated on X ∈ [0,1]^D (continuous inputs),
        or you replace 'game(X_np)' with your continuous teacher v(x).

    Returns:
        h: [L] tensor with G_i(t_ℓ; x)
        n_queries: number of game evaluations used (= 2 * L)
    """
    device = x.device
    if x.ndim != 1:
        raise ValueError("x must be 1D [D].")

    D = x.shape[0]
    assert 0 <= i < D

    t_nodes = t_nodes.to(device=device, dtype=torch.float32)  # [L]
    L = t_nodes.shape[0]

    # base: t * x for all j
    base = t_nodes.view(L, 1) * x.view(1, D)  # [L, D] in [0,1]

    X_on = base.clone()
    X_on[:, i] = x[i]  # original 0/1 value

    X_off = base.clone()
    X_off[:, i] = 0.0  # baseline

    # Convert to numpy for game evaluation
    X_on_np = X_on.detach().cpu().numpy()
    X_off_np = X_off.detach().cpu().numpy()

    # GAME CALLS HERE – must support continuous X, or be replaced by your own v(x)
    v_on = np.asarray(game(X_on_np), dtype=np.float64).reshape(-1)   # [L]
    v_off = np.asarray(game(X_off_np), dtype=np.float64).reshape(-1) # [L]

    h = torch.from_numpy(v_on - v_off).to(device=device, dtype=torch.float32)
    n_queries = 2 * L
    return h, n_queries


def tnshap_gi_game_interp(
    game,
    x: torch.Tensor,                # [D] in {0,1}
    max_degree: int,
    t_nodes: torch.Tensor,          # [L], L ≥ max_degree+1
) -> Tuple[torch.Tensor, int]:
    """
    TN-SHAP-style approximation for the GROUND-TRUTH game via Gi(t; x),
    assuming a (hypothetical) multilinear extension in input space.

    Returns:
        phi: [D] tensor of approximate Shapley values at coalition x.
        total_queries: total number of game evaluations used for this x.
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
    total_queries = 0

    for i in range(D):
        h, n_q = eval_Gi_game_interp(game, x, i, t_sub.to(device=device, dtype=torch.float32))
        total_queries += n_q
        h64 = h.to(torch.float64)
        m_i, *_ = torch.linalg.lstsq(V, h64.unsqueeze(-1))
        m_i = m_i.squeeze(-1)  # [max_degree+1]
        phi_i = torch.sum(alphas[: max_degree + 1] * m_i)
        phi[i] = phi_i.to(torch.float32)

    return phi, total_queries


# -----------------------
# KernelSHAP-style permutation estimator
# -----------------------

class CountingGameWrapper:
    """
    Wraps a Game to count the number of evaluations (queries).
    """

    def __init__(self, game):
        self.game = game
        self.eval_count = 0
        self.n_players = game.n_players

    def __call__(self, coalitions):
        coalitions = np.asarray(coalitions, dtype=bool)
        self.eval_count += coalitions.shape[0]
        return self.game(coalitions)


def kernelshap_permutation_sampling(
    game,
    n_players: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """
    Simple permutation-sampling Shapley estimator:

    For each permutation π of players:
      - start from empty set S = ∅ with value v(S).
      - for each player j in permutation order:
            φ_j += v(S ∪ {j}) - v(S)
            S ← S ∪ {j}

    Returns:
        phi_est: np.ndarray [n_players]
        n_queries: total game evaluations used.
    """
    wrapped = CountingGameWrapper(game)
    phi = np.zeros(n_players, dtype=np.float64)

    for _ in range(n_permutations):
        perm = rng.permutation(n_players)
        # evaluate v(∅)
        empty_bits = np.zeros((1, n_players), dtype=bool)
        v_S = float(wrapped(empty_bits)[0])
        mask = 0

        for j in perm:
            mask_with = mask | (1 << j)
            bits_with = ((np.array([[mask_with]], dtype=np.int64) >> np.arange(n_players)) & 1).astype(bool)
            v_S_with = float(wrapped(bits_with)[0])
            phi[j] += v_S_with - v_S
            mask = mask_with
            v_S = v_S_with

    phi /= float(n_permutations)
    return phi, wrapped.eval_count


# -----------------------
# CLI & main
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark multilinear-extension-style Shapley via Gi(t; x) interpolation "
                    "vs permutation-based KernelSHAP-style estimator on a shapiq Game."
    )
    p.add_argument(
        "--benchmark",
        type=str,
        choices=["adult_sv", "sentiment_si"],
        required=True,
        help=(
            "Which shapiq benchmark to use:\n"
            "  adult_sv     -> AdultCensusDataValuation\n"
            "  sentiment_si -> SentimentAnalysisLocalXAI"
        ),
    )
    p.add_argument("--n_points", type=int, default=100,
                   help="Number of random coalitions x to evaluate interpolation-based TN-SHAP on.")
    p.add_argument("--topk", type=int, default=3,
                   help="k for top-k support overlap on |φ|.")
    p.add_argument("--max_degree", type=int, default=None,
                   help="Max polynomial degree in t. Default: D-1.")
    p.add_argument("--n_t_nodes", type=int, default=None,
                   help="Number of Chebyshev nodes in [0,1]. Default: max_degree+1.")
    p.add_argument(
        "--n_ks_runs",
        type=int,
        default=5,
        help="Number of independent KernelSHAP runs (for std).",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1) Load Game
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

    # 2) Exact Shapley
    print("[INFO] Computing exact Shapley values (full enumeration)...")
    phi_exact = exact_shapley_from_game(game)  # [n_players]
    phi_exact_t = torch.from_numpy(phi_exact).to(DEVICE, dtype=torch.float32)
    print("[INFO] φ_exact (first 10):", phi_exact[:10])

    # 3) TN-SHAP-style interpolation settings
    D = n_players
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

    # 4) Evaluate interpolation-based TN-SHAP on random coalitions
    rng = np.random.default_rng(args.seed)
    k = min(args.topk, n_players)
    print(f"[INFO] Sampling {args.n_points} random coalitions and evaluating interpolation-based TN-SHAP...")

    corrs_interp = []
    overlaps_interp = []
    queries_interp_list = []

    for idx in range(args.n_points):
        x_np = (rng.random(D) < 0.5).astype(np.float32)  # random coalition {0,1}
        x = torch.from_numpy(x_np).to(DEVICE)

        phi_interp, n_q = tnshap_gi_game_interp(
            game=game,
            x=x,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )

        queries_interp_list.append(n_q)

        if torch.std(phi_exact_t) < 1e-8 or torch.std(phi_interp) < 1e-8:
            corr = float("nan")
        else:
            corr = float(torch.corrcoef(torch.stack([phi_exact_t, phi_interp]))[0, 1].item())

        top_exact = torch.topk(phi_exact_t.abs(), k).indices.tolist()
        top_interp = torch.topk(phi_interp.abs(), k).indices.tolist()
        overlap = len(set(top_exact) & set(top_interp)) / float(k)

        corrs_interp.append(corr)
        overlaps_interp.append(overlap)

    corrs_interp_np = np.array(corrs_interp, dtype=float)
    overlaps_interp_np = np.array(overlaps_interp, dtype=float)
    queries_interp_np = np.array(queries_interp_list, dtype=float)

    if np.all(np.isnan(corrs_interp_np)):
        mean_corr_interp = float("nan")
        std_corr_interp = float("nan")
    else:
        mean_corr_interp = float(np.nanmean(corrs_interp_np))
        std_corr_interp = float(np.nanstd(corrs_interp_np))

    mean_overlap_interp = float(np.mean(overlaps_interp_np))
    std_overlap_interp = float(np.std(overlaps_interp_np, ddof=1)) if len(overlaps_interp_np) > 1 else 0.0
    mean_queries_interp = float(np.mean(queries_interp_np))
    total_queries_interp = float(np.sum(queries_interp_np))

    print("==== Interpolation-based TN-SHAP on ground-truth game ====")
    print(f"corr(φ_interp(x), φ_exact): mean = {mean_corr_interp:.4f}, std = {std_corr_interp:.4f}")
    print(f"top-{k} overlap:            mean = {mean_overlap_interp:.4f}, std = {std_overlap_interp:.4f}")
    print(f"Average game queries per x: {mean_queries_interp:.1f}")
    print(f"Total game queries over {args.n_points} points: {total_queries_interp:.1f}")

    # 5) KernelSHAP-style estimator with similar budget
    approx_queries_per_perm = n_players + 1  # empty set + one per prefix
    # use TOTAL interpolation budget as KernelSHAP query budget
    kernel_budget = int(total_queries_interp)
    n_permutations = max(1, kernel_budget // approx_queries_per_perm)
    print(f"[INFO] KernelSHAP budget ≈ {kernel_budget}, approx queries/perm = {approx_queries_per_perm}")
    print(f"[INFO] Derived n_permutations ≈ {n_permutations}")

    rng_ks = np.random.default_rng(args.seed + 123)
    corr_ks_runs = []
    overlap_ks_runs = []
    queries_ks_runs = []

    for run in range(args.n_ks_runs):
        phi_ks, n_queries_ks = kernelshap_permutation_sampling(
            game=game,
            n_players=n_players,
            n_permutations=n_permutations,
            rng=rng_ks,
        )
        phi_ks_t = torch.from_numpy(phi_ks).to(DEVICE, dtype=torch.float32)

        if torch.std(phi_exact_t) < 1e-8 or torch.std(phi_ks_t) < 1e-8:
            corr_ks = float("nan")
        else:
            corr_ks = float(torch.corrcoef(torch.stack([phi_exact_t, phi_ks_t]))[0, 1].item())

        top_exact_ks = torch.topk(phi_exact_t.abs(), k).indices.tolist()
        top_ks = torch.topk(phi_ks_t.abs(), k).indices.tolist()
        overlap_ks = len(set(top_exact_ks) & set(top_ks)) / float(k)

        corr_ks_runs.append(corr_ks)
        overlap_ks_runs.append(overlap_ks)
        queries_ks_runs.append(n_queries_ks)

    corr_ks_np = np.array(corr_ks_runs, dtype=float)
    overlap_ks_np = np.array(overlap_ks_runs, dtype=float)
    queries_ks_np = np.array(queries_ks_runs, dtype=float)

    mean_corr_ks = float(np.nanmean(corr_ks_np))
    std_corr_ks = float(np.nanstd(corr_ks_np))
    mean_overlap_ks = float(np.mean(overlap_ks_np))
    std_overlap_ks = float(np.std(overlap_ks_np, ddof=1)) if len(overlap_ks_np) > 1 else 0.0
    mean_queries_ks = float(np.mean(queries_ks_np))

    print("==== KernelSHAP-style permutation estimator (global) ====")
    print(f"corr(φ_KS, φ_exact): mean = {mean_corr_ks:.4f}, std = {std_corr_ks:.4f}")
    print(f"top-{k} overlap:     mean = {mean_overlap_ks:.4f}, std = {std_overlap_ks:.4f}")
    print(f"Number of game queries used (avg over runs): {mean_queries_ks:.1f}")
    print(f"Queries per permutation: ~{n_players + 1} (empty set + one for each prefix).")


if __name__ == "__main__":
    main()
