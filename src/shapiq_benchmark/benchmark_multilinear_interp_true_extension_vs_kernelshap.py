#!/usr/bin/env python3
"""
Benchmark: Shapley via true multilinear extension + Gi(t; x)
vs. a simple permutation-based KernelSHAP-style estimator,
using ONLY the discrete game values v(S).

No TN training, no MLP, no continuous game calls.

Steps:

1) Load a shapiq Game (AdultCensusDataValuation / SentimentAnalysisLocalXAI).
2) Enumerate all coalitions S ⊆ N, get v(S) and store as v_all[mask].
3) Define the multilinear extension:

       F(z) = sum_S v(S) * prod_{i in S} z_i * prod_{i not in S} (1 - z_i),

   and implement it for batched z.

4) For N_POINTS random coalitions x ∈ {0,1}^D:
     - For each feature i, define paths in INPUT space:
         x_on_i(t)[j]  = x[j]      if j == i, else t * x[j]
         x_off_i(t)[j] = 0         if j == i, else t * x[j]
       Compute G_i(t; x) = F(x_on_i(t)) - F(x_off_i(t)) on Chebyshev nodes,
       fit polynomial in t via Vandermonde, and obtain Shapley_i(x)
       from the coefficients using α_s weights.
     - Compare φ_interp(x) to φ_exact:
           * correlation
           * top-k (default k=3) overlap on |φ|.
   Aggregate mean ± std.

5) Use the same enumerated v(S) to compute exact Shapley φ_exact.
6) Run a permutation-based KernelSHAP-style estimator with a comparable
   query budget and compare to φ_exact.
"""

import argparse
import math
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
    V = t.unsqueeze(1) ** exps.unsqueeze(0)  # [L, max_degree+1]
    return V


# -----------------------
# Enumerate v(S) and exact Shapley
# -----------------------

def enumerate_game_values(game) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enumerate v(S) for all S ⊆ N.

    Returns:
        v_all: np.ndarray [2^n] with v(S) in index = bitmask
        bits:  np.ndarray [2^n, n], boolean coalition matrix.
    """
    n = game.n_players
    num_coalitions = 1 << n

    masks = np.arange(num_coalitions, dtype=np.int64)
    bits = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)

    v_all = np.asarray(game(bits), dtype=np.float64).reshape(-1)  # [2^n]
    return v_all, bits


def exact_shapley_from_v_all(v_all: np.ndarray, n_players: int) -> np.ndarray:
    """
    Compute exact Shapley values from precomputed v_all[mask].

    φ_i = Σ_{S ⊆ N\{i}} [ |S|!(n-|S|-1)! / n! ] * [v(S ∪ {i}) - v(S)]
    """
    n = n_players
    num_coalitions = 1 << n
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

    return phi


# -----------------------
# True multilinear extension from v_all
# -----------------------

def multilinear_extension_from_v_all(
    v_all: np.ndarray,
    z: torch.Tensor,        # [L, D] or [D]
) -> torch.Tensor:
    """
    Evaluate the multilinear extension

        F(z) = Σ_S v(S) ∏_{i∈S} z_i ∏_{i∉S} (1 - z_i)

    using precomputed v_all[mask] (mask encodes S).

    Args:
        v_all: np.ndarray [2^D]
        z: torch.Tensor [L, D] or [D]

    Returns:
        F(z): torch.Tensor [L] (or scalar if input was 1D)
    """
    v_all_t = torch.from_numpy(v_all).to(z.device, dtype=torch.float64)  # [2^D]

    if z.ndim == 1:
        z = z.unsqueeze(0)  # [1, D]
        squeeze_back = True
    else:
        squeeze_back = False

    L, D = z.shape
    num_coalitions = 1 << D

    # precompute masks: [2^D, D] booleans
    masks = torch.arange(num_coalitions, device=z.device, dtype=torch.long)
    bits = ((masks.unsqueeze(1) >> torch.arange(D, device=z.device)) & 1).bool()  # [2^D, D]

    # For each coalition S (row in bits), we need:
    #   weight_l(S) = ∏_i z_l[i]     if bits[S,i] = 1
    #                          (1-z_l[i]) if bits[S,i] = 0
    #
    # We can do this as:
    #   term_pos = z.unsqueeze(0)      # [1, L, D] -> [2^D, L, D]
    #   term_neg = (1 - z).unsqueeze(0)
    #   choose term_pos/term_neg based on bits, then product over D.
    z64 = z.to(dtype=torch.float64)
    term_pos = z64.unsqueeze(0).expand(num_coalitions, L, D)        # [2^D, L, D]
    term_neg = (1.0 - z64).unsqueeze(0).expand(num_coalitions, L, D)

    bits_expanded = bits.unsqueeze(1).expand(num_coalitions, L, D)  # [2^D, L, D]
    terms = torch.where(bits_expanded, term_pos, term_neg)         # [2^D, L, D]

    weights = terms.prod(dim=2)  # [2^D, L]
    # Now F(z_l) = Σ_S v_all[S] * weights[S, l]
    F_vals = (v_all_t.unsqueeze(1) * weights).sum(dim=0)  # [L]

    if squeeze_back:
        return F_vals[0]
    return F_vals


# -----------------------
# Gi(t; x) and Shapley via multilinear extension
# -----------------------

def eval_Gi_multilinear(
    v_all: np.ndarray,
    x: torch.Tensor,        # [D] in {0,1}
    i: int,
    t_nodes: torch.Tensor,  # [L]
) -> torch.Tensor:
    """
    Compute G_i(t_ℓ; x) using the multilinear extension:

        x_on_i(t)[j]  = x[j]      if j == i, else t * x[j]
        x_off_i(t)[j] = 0         if j == i, else t * x[j]

      G_i(t; x) = F(x_on_i(t)) - F(x_off_i(t)),

    where F is the multilinear extension built from v_all.
    No extra game evaluations are used here.
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
    X_on[:, i] = x[i]

    X_off = base.clone()
    X_off[:, i] = 0.0

    F_on = multilinear_extension_from_v_all(v_all, X_on)   # [L]
    F_off = multilinear_extension_from_v_all(v_all, X_off) # [L]

    h = (F_on - F_off).to(dtype=torch.float32)
    return h  # [L]


def tnshap_gi_multilinear(
    v_all: np.ndarray,
    x: torch.Tensor,                # [D] in {0,1}
    max_degree: int,
    t_nodes: torch.Tensor,          # [L], L ≥ max_degree+1
) -> torch.Tensor:
    """
    TN-SHAP-style approximation via Gi(t; x) using the TRUE multilinear extension.

    Returns:
        phi: [D] tensor of approximate Shapley values at coalition x.
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
        h = eval_Gi_multilinear(v_all, x, i, t_sub.to(device=device, dtype=torch.float32))  # [L]
        h64 = h.to(torch.float64)
        m_i, *_ = torch.linalg.lstsq(V, h64.unsqueeze(-1))
        m_i = m_i.squeeze(-1)  # [max_degree+1]
        phi_i = torch.sum(alphas[: max_degree + 1] * m_i)
        phi[i] = phi_i.to(torch.float32)

    return phi  # [D]


# -----------------------
# KernelSHAP-style estimator (same as before)
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
        description="Benchmark Shapley via true multilinear extension + Gi(t; x) "
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

    # 2) Enumerate v(S) once
    print("[INFO] Enumerating v(S) for all coalitions...")
    v_all, bits = enumerate_game_values(game)
    print(f"[INFO] Enumerated {v_all.shape[0]} coalitions.")

    # 3) Exact Shapley from v_all
    print("[INFO] Computing exact Shapley values from v_all...")
    phi_exact = exact_shapley_from_v_all(v_all, n_players)
    phi_exact_t = torch.from_numpy(phi_exact).to(DEVICE, dtype=torch.float32)
    print("[INFO] φ_exact (first 10):", phi_exact[:10])

    # 4) Multilinear TN-SHAP-style interpolation settings
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

    # 5) Evaluate interpolation-based TN-SHAP on random coalitions
    rng = np.random.default_rng(args.seed)
    k = min(args.topk, n_players)
    print(f"[INFO] Sampling {args.n_points} random coalitions and evaluating interpolation-based TN-SHAP...")

    corrs_interp = []
    overlaps_interp = []

    for idx in range(args.n_points):
        x_np = (rng.random(D) < 0.5).astype(np.float32)  # random coalition {0,1}
        x = torch.from_numpy(x_np).to(DEVICE)

        phi_interp = tnshap_gi_multilinear(
            v_all=v_all,
            x=x,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )

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

    mean_corr_interp = float(np.nanmean(corrs_interp_np))
    std_corr_interp = float(np.nanstd(corrs_interp_np))
    mean_overlap_interp = float(np.mean(overlaps_interp_np))
    std_overlap_interp = float(np.std(overlaps_interp_np, ddof=1)) if len(overlaps_interp_np) > 1 else 0.0

    print("==== Interpolation-based TN-SHAP via TRUE multilinear extension ====")
    print(f"corr(φ_interp(x), φ_exact): mean = {mean_corr_interp:.4f}, std = {std_corr_interp:.4f}")
    print(f"top-{k} overlap:            mean = {mean_overlap_interp:.4f}, std = {std_overlap_interp:.4f}")
    print("[INFO] No additional game queries beyond the initial 2^n enumeration are used.")

    # 6) KernelSHAP-style estimator with a comparable query budget
    #    Here, to be fair, we might simply pick a fixed number of permutations.
    approx_queries_per_perm = n_players + 1
    n_permutations = max(1, (1 << n_players) // approx_queries_per_perm)
    print(f"[INFO] For KernelSHAP, using n_permutations ≈ {n_permutations} "
          f"(~2^n / (n+1) queries).")

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
