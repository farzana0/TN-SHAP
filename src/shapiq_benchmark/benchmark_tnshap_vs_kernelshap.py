#!/usr/bin/env python3
"""
Benchmark TN-SHAP (FeatureMappedTN + selectors in feature space)
vs. a simple permutation-sampling "KernelSHAP-style" estimator
on a shapiq benchmark Game (e.g. AdultCensusDataValuation).

What this script does:

1) Load a shapiq Game (AdultCensusDataValuation or SentimentAnalysisLocalXAI).
2) Compute exact Shapley values φ_exact by full enumeration over 2^n coalitions.
3) Load a trained FeatureMappedTN checkpoint (from train_featuremapped_tn_shapiq_benchmark.py),
   rebuilt with selector_mode='per_feature_scalar' so we can do path selectors in feature space.
4) Sample N_POINTS random coalitions x ∈ {0,1}^n.
   For each x:
     - Compute TN-SHAP φ_TN(x) via Gi(t; x) with selectors in feature space.
     - Compare φ_TN(x) to φ_exact:
          * correlation
          * top-k overlap on |φ| (support accuracy)
   Aggregate mean ± std over the N_POINTS coalitions.
5) Compute a permutation-sampling Shapley estimator φ_KS (KernelSHAP-style),
   with n_permutations derived from a query budget (tn_budget) or provided explicitly.
   We repeat this KS estimator ks_repeats times with different permutations, and report:
          * mean ± std correlation vs φ_exact
          * mean ± std top-k overlap
          * mean ± std query count
"""

import argparse
import math
from typing import Optional, Tuple

import numpy as np
import torch

from shapiq.benchmark import load_games_from_configuration
from feature_mapped_tn import make_feature_mapped_tn, FeatureMappedTN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Utilities
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
# Load FeatureMappedTN with selectors enabled
# -----------------------

def load_featuremapped_tn_with_selectors(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> FeatureMappedTN:
    """
    Load a FeatureMappedTN checkpoint and rebuild the model with
    selector_mode='per_feature_scalar' so we can apply thin-diagonal selectors
    in feature-map space.
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
# Gi(t; x) via selectors in feature space
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

    With ψ(0) = 0, selectors=0 corresponds to baseline, selectors=1 is "fully on".
    """
    device = x.device
    t_nodes = t_nodes.to(device=device, dtype=torch.float32)

    if x.ndim == 1:
        x = x.unsqueeze(0)  # [1, D]
    B, D = x.shape
    assert B == 1
    assert 0 <= i < D

    L = t_nodes.shape[0]

    # Repeat x for each t node: [L, D]
    x_batch = x.repeat(L, 1)

    # Base selectors: j ≠ i → t, same t across all features for each row
    t_col = t_nodes.view(L, 1)            # [L,1]
    selectors_base = t_col.repeat(1, D)   # [L,D]

    # ON path: feature i selector=1, others=t
    selectors_on = selectors_base.clone()
    selectors_on[:, i] = 1.0

    # OFF path: feature i selector=0, others=t
    selectors_off = selectors_base.clone()
    selectors_off[:, i] = 0.0

    with torch.no_grad():
        y_on = model(x_batch, selectors_on).squeeze(-1)    # [L]
        y_off = model(x_batch, selectors_off).squeeze(-1)  # [L]

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
        h = eval_Gi_selector_surrogate(
            model, x, i, t_sub.to(device=device, dtype=torch.float32)
        )  # [L]
        h64 = h.to(torch.float64)
        m_i, *_ = torch.linalg.lstsq(V, h64.unsqueeze(-1))
        m_i = m_i.squeeze(-1)  # [max_degree+1]
        phi_i = torch.sum(alphas[: max_degree + 1] * m_i)
        phi[i] = phi_i.to(torch.float32)

    return phi  # [D]


# -----------------------
# Simple permutation-sampling "KernelSHAP-style" estimator
# -----------------------

class CountingGameWrapper:
    """
    Wraps a shapiq Game to count the number of evaluations (queries).
    """

    def __init__(self, game):
        self.game = game
        self.eval_count = 0
        # expose basic attributes used by some code if needed
        self.n_players = game.n_players
        self.normalize = getattr(game, "normalize", False)
        self.normalization_value = getattr(game, "normalization_value", 0.0)

    def __call__(self, coalitions):
        coalitions = np.asarray(coalitions, dtype=bool)
        self.eval_count += coalitions.shape[0]
        return self.game(coalitions)


def kernelshap_permutation_sampling(
    game, n_players: int, n_permutations: int, rng: np.random.Generator
) -> Tuple[np.ndarray, int]:
    """
    Very simple permutation-sampling Shapley estimator (similar spirit to
    permutation-based KernelSHAP / PermutationSamplingSV).

    For each permutation π of players:
      - start from empty set S = ∅ with value v(S).
      - for each player j in permutation order:
            φ_j += v(S ∪ {j}) - v(S)
            S ← S ∪ {j}

    We approximate φ by averaging contributions over n_permutations.

    Returns:
        phi_est: np.ndarray [n_players]
        n_queries: total number of game evaluations used.
    """
    wrapped = CountingGameWrapper(game)
    phi = np.zeros(n_players, dtype=np.float64)

    for _ in range(n_permutations):
        perm = rng.permutation(n_players)
        # start from empty coalition
        mask = 0
        # evaluate v(∅)
        empty_bits = np.zeros((1, n_players), dtype=bool)
        v_S = float(wrapped(empty_bits)[0])
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
        description="Benchmark TN-SHAP (FeatureMappedTN + selectors) vs permutation-based "
                    "KernelSHAP-style estimator on a shapiq Game."
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
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to FeatureMappedTN checkpoint (from train_featuremapped_tn_shapiq_benchmark.py).")
    p.add_argument("--n_points", type=int, default=100,
                   help="Number of random coalitions x to evaluate TN-SHAP on.")
    p.add_argument("--topk", type=int, default=5,
                   help="k for top-k support overlap on |φ|.")
    p.add_argument("--max-degree", type=int, default=None,
                   help="Max polynomial degree in t. Default: D-1.")
    p.add_argument("--n_t_nodes", type=int, default=None,
                   help="Number of Chebyshev nodes in [0,1]. Default: max_degree+1.")
    p.add_argument(
        "--n_permutations",
        type=int,
        default=None,
        help=(
            "Number of permutations for permutation-sampling kernelshap-style estimator. "
            "If None and --tn_budget is given, it will be derived from tn_budget."
        ),
    )
    p.add_argument(
        "--tn_budget",
        type=int,
        default=None,
        help=(
            "Approximate total number of model evaluations used by TN-SHAP (i.e., "
            "TN training query budget). Used to match KernelSHAP's budget."
        ),
    )
    p.add_argument(
        "--ks_repeats",
        type=int,
        default=5,
        help="Number of independent KernelSHAP-style runs to estimate mean/std.",
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
    print("[INFO] φ_exact (first 10):", phi_exact[:10])

    # 3) Load TN surrogate with selectors
    print(f"[INFO] Loading FeatureMappedTN with selectors from {args.ckpt}")
    model = load_featuremapped_tn_with_selectors(args.ckpt, device=DEVICE)
    D = model.d_in
    assert D == n_players, f"Model d_in={D} but game.n_players={n_players}"
    print("[INFO] FeatureMappedTN loaded.")

    # 4) t-nodes and degree
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

    # 5) Benchmark TN-SHAP over N_POINTS random coalitions
    rng = np.random.default_rng(args.seed)
    phi_exact_t = torch.from_numpy(phi_exact).to(DEVICE, dtype=torch.float32)

    corrs_tn = []
    overlaps_tn = []

    k = min(args.topk, n_players)

    print(f"[INFO] Sampling {args.n_points} random coalitions and evaluating TN-SHAP...")

    for idx in range(args.n_points):
        # sample random coalition x ∈ {0,1}^D
        x_np = (rng.random(D) < 0.5).astype(np.float32)
        x = torch.from_numpy(x_np).to(DEVICE)

        phi_tn = tnshap_gi_selector_surrogate(
            model=model,
            x=x,
            max_degree=max_degree,
            t_nodes=t_nodes,
        )  # [D]

        phi_tn_t = phi_tn.to(DEVICE, dtype=torch.float32)

        # correlation
        if torch.std(phi_exact_t) < 1e-8 or torch.std(phi_tn_t) < 1e-8:
            corr = float("nan")
        else:
            corr = float(torch.corrcoef(torch.stack([phi_exact_t, phi_tn_t]))[0, 1].item())

        # top-k overlap on |φ|
        top_exact = torch.topk(phi_exact_t.abs(), k).indices.tolist()
        top_tn    = torch.topk(phi_tn_t.abs(),    k).indices.tolist()
        overlap = len(set(top_exact) & set(top_tn)) / float(k)

        corrs_tn.append(corr)
        overlaps_tn.append(overlap)

    # ---- aggregate TN-SHAP metrics (nan-safe using numpy) ----
    corrs_np = np.array(corrs_tn, dtype=float)
    overlaps_tn_t = torch.tensor(overlaps_tn, device=DEVICE)

    if np.all(np.isnan(corrs_np)):
        mean_corr_tn = float("nan")
        std_corr_tn = float("nan")
    else:
        mean_corr_tn = float(np.nanmean(corrs_np))
        std_corr_tn = float(np.nanstd(corrs_np))

    mean_overlap_tn = overlaps_tn_t.mean().item()
    std_overlap_tn = overlaps_tn_t.std(unbiased=True).item() if len(overlaps_tn_t) > 1 else 0.0

    print("==== TN-SHAP (FeatureMappedTN + selectors) over random coalitions ====")
    print(f"corr(φ_TN(x), φ_exact): mean = {mean_corr_tn:.4f}, std = {std_corr_tn:.4f}")
    print(f"top-{k} overlap:        mean = {mean_overlap_tn:.4f}, std = {std_overlap_tn:.4f}")
    print("[INFO] TN-SHAP evaluation uses 0 additional game queries at test time "
          "(all game queries were used in training the surrogate).")

    # 6) KernelSHAP-style permutation-sampling estimator (global)
    #    Match query budget to TN-SHAP if tn_budget is provided.
    if args.tn_budget is not None:
        approx_queries_per_perm = n_players + 1  # empty set + one per prefix
        n_permutations = max(1, args.tn_budget // approx_queries_per_perm)
        print("[INFO] Using tn_budget={} to set KernelSHAP permutations.".format(args.tn_budget))
        print(f"[INFO] Approx. queries per permutation: {approx_queries_per_perm}")
        print(f"[INFO] Derived n_permutations ≈ {n_permutations}")
    else:
        if args.n_permutations is None:
            raise ValueError(
                "Either --tn_budget or --n_permutations must be provided for KernelSHAP."
            )
        n_permutations = args.n_permutations
        print(f"[INFO] Using explicit n_permutations = {n_permutations} for KernelSHAP.")

    # ---- Repeat KernelSHAP-style estimator several times to get mean/std ----
    print("[INFO] Running permutation-based KernelSHAP-style estimator "
          f"for ks_repeats={args.ks_repeats} runs...")

    ks_corrs = []
    ks_overlaps = []
    ks_queries = []

    for rep in range(args.ks_repeats):
        # separate RNG stream per repeat for independent permutations
        rng_rep = np.random.default_rng(args.seed + 1000 + rep)

        phi_ks, n_queries_ks = kernelshap_permutation_sampling(
            game=game,
            n_players=n_players,
            n_permutations=n_permutations,
            rng=rng_rep,
        )
        phi_ks_t = torch.from_numpy(phi_ks).to(DEVICE, dtype=torch.float32)

        # correlation & overlap vs exact
        if torch.std(phi_exact_t) < 1e-8 or torch.std(phi_ks_t) < 1e-8:
            corr_ks = float("nan")
        else:
            corr_ks = float(torch.corrcoef(torch.stack([phi_exact_t, phi_ks_t]))[0, 1].item())

        top_exact_ks = torch.topk(phi_exact_t.abs(), k).indices.tolist()
        top_ks       = torch.topk(phi_ks_t.abs(),    k).indices.tolist()
        overlap_ks = len(set(top_exact_ks) & set(top_ks)) / float(k)

        ks_corrs.append(corr_ks)
        ks_overlaps.append(overlap_ks)
        ks_queries.append(n_queries_ks)

    ks_corrs_np = np.array(ks_corrs, dtype=float)
    ks_overlaps_np = np.array(ks_overlaps, dtype=float)
    ks_queries_np = np.array(ks_queries, dtype=float)

    if np.all(np.isnan(ks_corrs_np)):
        mean_corr_ks = float("nan")
        std_corr_ks = float("nan")
    else:
        mean_corr_ks = float(np.nanmean(ks_corrs_np))
        std_corr_ks = float(np.nanstd(ks_corrs_np))

    mean_overlap_ks = float(np.mean(ks_overlaps_np))
    std_overlap_ks = float(np.std(ks_overlaps_np, ddof=1)) if args.ks_repeats > 1 else 0.0

    mean_queries_ks = float(np.mean(ks_queries_np))
    std_queries_ks = float(np.std(ks_queries_np, ddof=1)) if args.ks_repeats > 1 else 0.0

    print("==== KernelSHAP-style permutation estimator (global) ====")
    print(f"corr(φ_KS, φ_exact): mean = {mean_corr_ks:.4f}, std = {std_corr_ks:.4f}")
    print(f"top-{k} overlap:     mean = {mean_overlap_ks:.4f}, std = {std_overlap_ks:.4f}")
    print(f"Number of game queries used: mean = {mean_queries_ks:.1f}, std = {std_queries_ks:.1f}")
    print(f"Queries per permutation: ~{n_players + 1} (empty set + one for each prefix).")


if __name__ == "__main__":
    main()
