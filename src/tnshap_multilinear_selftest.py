#!/usr/bin/env python3
"""
Standalone multilinear self-test for TN-SHAP (Gi + Vandermonde).

Compares the approximate Shapley values from tnshap_gi_teacher() against
the exact Shapley values of a multilinear polynomial with baseline 0.
"""

import argparse
import math
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def build_vandermonde(t_nodes: torch.Tensor, degree_max: int) -> torch.Tensor:
    """
    V[l, r] = t_nodes[l] ** r  for r = 0,...,degree_max.
    """
    t = t_nodes.to(dtype=torch.float64)
    exps = torch.arange(0, degree_max + 1, dtype=torch.float64, device=t.device)
    V = t.unsqueeze(1) ** exps.unsqueeze(0)  # [L, degree_max+1]
    return V


def eval_Gi_teacher(eval_fn, x: torch.Tensor, i: int, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    Compute vector G_i(t_ℓ; x) using:
        x_on_i(t)[j]  = x[j]    if j == i, else t * x[j]
        x_off_i(t)[j] = 0       if j == i, else t * x[j]
    """
    x = x.to(DEVICE)
    t_nodes = t_nodes.to(x.device)
    base = t_nodes.unsqueeze(1) * x.unsqueeze(0)   # [L, D]
    X_on = base.clone()
    X_on[:, i] = x[i]
    X_off = base.clone()
    X_off[:, i] = 0.0
    with torch.no_grad():
        y_on = eval_fn(X_on)   # [L]
        y_off = eval_fn(X_off) # [L]
    return y_on - y_off


def tnshap_gi_teacher(eval_fn, x: torch.Tensor, max_degree: int, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    TN-SHAP approximation via Gi(t;x) + Vandermonde.
    """
    x = x.to(DEVICE)
    D = x.shape[0]
    assert t_nodes.shape[0] >= max_degree + 1, "Need at least max_degree+1 nodes."
    t_sub = t_nodes[: max_degree + 1]
    V = build_vandermonde(t_sub, degree_max=max_degree)  # [L, max_degree+1]
    alphas = shapley_weights(D, device=DEVICE, dtype=torch.float64)  # [D]
    phi = torch.zeros(D, device=DEVICE, dtype=torch.float32)
    for i in range(D):
        h = eval_Gi_teacher(eval_fn, x, i, t_sub)  # [L]
        h64 = h.to(torch.float64)
        m_i, *_ = torch.linalg.lstsq(V, h64.unsqueeze(-1))
        m_i = m_i.squeeze(-1)  # [max_degree+1]
        phi_i = torch.sum(alphas[: max_degree + 1] * m_i)
        phi[i] = phi_i.to(torch.float32)
    return phi


def make_multilinear_coeffs(
    D: int,
    min_degree: int,
    max_degree: int,
    seed: int = 0,
    include_constant: bool = True,
):
    """
    Create random multilinear coefficients for all monomials up to max_degree.
    Returns a dict mapping tuple of indices -> coefficient (float64).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    min_degree = max(1, min(min_degree, D))
    max_degree = max(min_degree, min(max_degree, D))
    coeffs = {}
    if include_constant:
        coeffs[()] = torch.empty((), dtype=torch.float64).uniform_(-0.5, 0.5, generator=rng).item()
    for k in range(min_degree, max_degree + 1):
        for comb in torch.combinations(torch.arange(D), r=k):
            key = tuple(int(i) for i in comb.tolist())
            coeffs[key] = torch.empty((), dtype=torch.float64).uniform_(-1.0, 1.0, generator=rng).item()
    return coeffs


def eval_multilinear(coeffs, x_batch: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a multilinear polynomial for a batch.
    x_batch: [N, D]
    Returns: [N] float64
    """
    x_batch = x_batch.to(dtype=torch.float64)
    y = torch.zeros(x_batch.shape[0], dtype=torch.float64, device=x_batch.device)
    for key, coeff in coeffs.items():
        if len(key) == 0:
            y = y + coeff
            continue
        idx = torch.tensor(key, device=x_batch.device)
        term = x_batch[:, idx].prod(dim=1)
        y = y + coeff * term
    return y


def exact_shapley_multilinear(coeffs, x: torch.Tensor) -> torch.Tensor:
    """
    Exact Shapley for multilinear game v(S)=f(x_S) with baseline 0.
    For each monomial over T, each player gets 1/|T| of that term.
    """
    x = x.to(dtype=torch.float64)
    D = x.shape[0]
    phi = torch.zeros(D, dtype=torch.float64, device=x.device)
    for key, coeff in coeffs.items():
        if len(key) == 0:
            continue
        T = list(key)
        term_val = coeff * x[T].prod()
        share = term_val / float(len(T))
        for i in T:
            phi[i] = phi[i] + share
    return phi


def run_self_test(
    D: int,
    term_degree: int,
    max_degree: int,
    seed: int,
    atol: float,
    rtol: float,
):
    coeffs = make_multilinear_coeffs(
        D,
        min_degree=term_degree,
        max_degree=term_degree,
        seed=seed,
    )
    x = torch.empty(D, dtype=torch.float64, device=DEVICE).uniform_(-1.0, 1.0)
    t_nodes = torch.linspace(0.0, 1.0, steps=max_degree + 1, dtype=torch.float64, device=DEVICE)

    def eval_fn(x_batch: torch.Tensor) -> torch.Tensor:
        return eval_multilinear(coeffs, x_batch)

    phi_tn = tnshap_gi_teacher(eval_fn, x, max_degree=max_degree, t_nodes=t_nodes)
    phi_exact = exact_shapley_multilinear(coeffs, x)

    max_abs = (phi_tn.to(torch.float64) - phi_exact).abs().max().item()
    ok = torch.allclose(phi_tn.to(torch.float64), phi_exact, rtol=rtol, atol=atol)
    print("==== Multilinear self-test (Gi + Vandermonde) ====")
    print(f"D={D}, term_degree={term_degree}, max_degree={max_degree}, seed={seed}")
    print(f"max_abs_err={max_abs:.3e}, atol={atol:.1e}, rtol={rtol:.1e}, pass={ok}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Validate tnshap_gi_teacher() on a random multilinear function."
    )
    parser.add_argument("--d", type=int, default=6, help="Number of features.")
    parser.add_argument("--term-degree", type=int, default=None,
                        help="Degree of multilinear monomials to generate. Default: D (full degree).")
    parser.add_argument("--max-degree", type=int, default=None,
                        help="Interpolation max degree. Default: term_degree - 1.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance.")
    args = parser.parse_args()
    term_degree = args.term_degree if args.term_degree is not None else args.d
    if term_degree < 1 or term_degree > args.d:
        raise ValueError("term_degree must be between 1 and D.")
    max_degree = args.max_degree if args.max_degree is not None else term_degree - 1
    if max_degree < 0:
        raise ValueError("max_degree must be >= 0.")
    run_self_test(args.d, term_degree, max_degree, args.seed, args.atol, args.rtol)


if __name__ == "__main__":
    main()

