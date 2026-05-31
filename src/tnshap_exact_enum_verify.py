#!/usr/bin/env python3
"""
Verify that selector-matrix Gi(t) + Vandermonde matches exact Shapley
enumeration on a multilinear function.
"""

import argparse
import math
from itertools import combinations

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shapley_weights(D: int, device=None, dtype=torch.float64) -> torch.Tensor:
    if device is None:
        device = DEVICE
    alphas = torch.empty(D, dtype=dtype, device=device)
    D_fact = math.factorial(D)
    for s in range(D):
        num = math.factorial(s) * math.factorial(D - s - 1)
        alphas[s] = num / D_fact
    return alphas


def build_vandermonde(t_nodes: torch.Tensor, degree_max: int) -> torch.Tensor:
    t = t_nodes.to(dtype=torch.float64)
    exps = torch.arange(0, degree_max + 1, dtype=torch.float64, device=t.device)
    return t.unsqueeze(1) ** exps.unsqueeze(0)


def make_full_degree_coeffs(D: int, seed: int = 0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    coeffs = {}
    # Full-degree terms only: all size-D monomials (single term).
    key = tuple(range(D))
    coeffs[key] = torch.empty((), dtype=torch.float64).uniform_(-1.0, 1.0, generator=rng).item()
    return coeffs


def make_multilinear_coeffs(
    D: int,
    min_degree: int,
    max_degree: int,
    seed: int = 0,
    include_constant: bool = True,
):
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
    x_batch = x_batch.to(dtype=torch.float64)
    y = torch.zeros(x_batch.shape[0], dtype=torch.float64, device=x_batch.device)
    for key, coeff in coeffs.items():
        if len(key) == 0:
            y = y + coeff
            continue
        idx = torch.tensor(key, device=x_batch.device)
        y = y + coeff * x_batch[:, idx].prod(dim=1)
    return y


def exact_shapley_enumeration(coeffs, x: torch.Tensor) -> torch.Tensor:
    x = x.to(dtype=torch.float64)
    D = x.shape[0]
    phi = torch.zeros(D, dtype=torch.float64, device=x.device)
    players = list(range(D))
    for i in range(D):
        others = [j for j in players if j != i]
        for s in range(0, D):
            for S in combinations(others, s):
                S = set(S)
                weight = math.factorial(s) * math.factorial(D - s - 1) / math.factorial(D)
                mask_on = torch.zeros(D, dtype=torch.float64, device=x.device)
                mask_off = torch.zeros(D, dtype=torch.float64, device=x.device)
                for j in S:
                    mask_on[j] = 1.0
                    mask_off[j] = 1.0
                mask_on[i] = 1.0
                f_on = eval_multilinear(coeffs, (x * mask_on).unsqueeze(0))[0]
                f_off = eval_multilinear(coeffs, (x * mask_off).unsqueeze(0))[0]
                phi[i] = phi[i] + weight * (f_on - f_off)
    return phi


def exact_shapley_multilinear(coeffs, x: torch.Tensor) -> torch.Tensor:
    """
    Closed-form Shapley for multilinear v(S)=f(x_S) with baseline 0.
    Each monomial over T is split evenly among players in T.
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


def tnshap_gi_vandermonde_selector(coeffs, x: torch.Tensor, max_degree: int, t_nodes: torch.Tensor):
    x = x.to(DEVICE)
    D = x.shape[0]
    t_sub = t_nodes[: max_degree + 1].to(device=DEVICE, dtype=torch.float64)
    V = build_vandermonde(t_sub, degree_max=max_degree)
    alphas = shapley_weights(D, device=DEVICE, dtype=torch.float64)
    phi = torch.zeros(D, device=DEVICE, dtype=torch.float64)

    for i in range(D):
        # Selector matrices: t for all j≠i, and 1/0 for i
        t_col = t_sub.view(-1, 1)  # [L,1]
        selectors_base = t_col.repeat(1, D)
        selectors_on = selectors_base.clone()
        selectors_on[:, i] = 1.0
        selectors_off = selectors_base.clone()
        selectors_off[:, i] = 0.0

        x_batch = x.unsqueeze(0).repeat(t_sub.shape[0], 1)
        y_on = eval_multilinear(coeffs, x_batch * selectors_on)
        y_off = eval_multilinear(coeffs, x_batch * selectors_off)
        h = (y_on - y_off).to(torch.float64)  # [L]

        m_i, *_ = torch.linalg.lstsq(V, h.unsqueeze(-1))
        m_i = m_i.squeeze(-1)
        phi[i] = torch.sum(alphas[: max_degree + 1] * m_i)

    return phi


def run_verify(D: int, seed: int, atol: float, rtol: float):
    """
    Full-degree test: compares selector+Vandermonde TN-SHAP against
    exact Shapley enumeration, and checks enumeration vs closed-form.
    """
    coeffs = make_full_degree_coeffs(D, seed=seed)
    x = torch.empty(D, dtype=torch.float64, device=DEVICE).uniform_(-1.0, 1.0)
    max_degree = D - 1
    t_nodes = torch.linspace(0.0, 1.0, steps=max_degree + 1, dtype=torch.float64, device=DEVICE)

    phi_enum = exact_shapley_enumeration(coeffs, x)
    phi_closed = exact_shapley_multilinear(coeffs, x)
    phi_tn = tnshap_gi_vandermonde_selector(coeffs, x, max_degree=max_degree, t_nodes=t_nodes)

    max_abs_tn = (phi_tn - phi_enum).abs().max().item()
    ok_tn = torch.allclose(phi_tn, phi_enum, rtol=rtol, atol=atol)
    max_abs_enum = (phi_enum - phi_closed).abs().max().item()
    ok_enum = torch.allclose(phi_enum, phi_closed, rtol=rtol, atol=atol)

    print("==== Exact Enumeration vs Selector+Vandermonde ====")
    print(f"D={D}, max_degree={max_degree}, seed={seed}")
    print(f"max_abs_err_tn={max_abs_tn:.3e}, pass_tn={ok_tn}")
    print(f"max_abs_err_enum={max_abs_enum:.3e}, pass_enum={ok_enum}")
    return ok_tn and ok_enum


def run_arbitrary_self_test(
    D: int,
    min_degree: int,
    max_degree: int,
    seed: int,
    atol: float,
    rtol: float,
):
    coeffs = make_multilinear_coeffs(
        D,
        min_degree=min_degree,
        max_degree=max_degree,
        seed=seed,
        include_constant=True,
    )
    x = torch.empty(D, dtype=torch.float64, device=DEVICE).uniform_(-1.0, 1.0)
    phi_enum = exact_shapley_enumeration(coeffs, x)
    phi_closed = exact_shapley_multilinear(coeffs, x)
    max_abs = (phi_enum - phi_closed).abs().max().item()
    ok = torch.allclose(phi_enum, phi_closed, rtol=rtol, atol=atol)
    print("==== Exact Enumeration Self-Test (Arbitrary Multilinear) ====")
    print(f"D={D}, min_degree={min_degree}, max_degree={max_degree}, seed={seed}")
    print(f"max_abs_err={max_abs:.3e}, atol={atol:.1e}, rtol={rtol:.1e}, pass={ok}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test-arbitrary", action="store_true",
                        help="Validate exact enumeration against closed-form Shapley "
                             "for a random multilinear function.")
    parser.add_argument("--min-degree", type=int, default=1,
                        help="Minimum monomial degree for arbitrary self-test.")
    parser.add_argument("--max-degree", type=int, default=None,
                        help="Maximum monomial degree for arbitrary self-test. Default: D.")
    parser.add_argument("--d", type=int, default=6, help="Number of features.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance.")
    args = parser.parse_args()
    if args.self_test_arbitrary:
        max_degree = args.max_degree if args.max_degree is not None else args.d
        run_arbitrary_self_test(
            args.d,
            min_degree=args.min_degree,
            max_degree=max_degree,
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
        )
    else:
        run_verify(args.d, args.seed, args.atol, args.rtol)


if __name__ == "__main__":
    main()

