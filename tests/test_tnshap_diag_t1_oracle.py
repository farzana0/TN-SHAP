#!/usr/bin/env python3
"""
This test distinguishes the scalar baseline-0 Gi path from the paper’s
augmented [x,1] diag(t,1) selector semantics.
"""

import math
from itertools import combinations

import torch


def _make_coeffs(D: int, K: int, seed: int):
    rng = torch.Generator()
    rng.manual_seed(seed)
    coeffs = {}
    for k in range(1, K + 1):
        for comb in combinations(range(D), k):
            coeffs[comb] = torch.empty((), dtype=torch.float64).uniform_(
                -1.0, 1.0, generator=rng
            ).item()
    return coeffs


def _eval_g_from_u(u: torch.Tensor, coeffs) -> torch.Tensor:
    out = torch.zeros((), dtype=torch.float64)
    for T, c_T in coeffs.items():
        idx = torch.tensor(T, dtype=torch.int64)
        out = out + c_T * u[idx].prod()
    return out


def _shapley_weights(D: int) -> torch.Tensor:
    alphas = torch.empty(D, dtype=torch.float64)
    D_fact = math.factorial(D)
    for s in range(D):
        alphas[s] = math.factorial(s) * math.factorial(D - s - 1) / D_fact
    return alphas


def _build_vandermonde(t_nodes: torch.Tensor, degree_max: int) -> torch.Tensor:
    exps = torch.arange(0, degree_max + 1, dtype=torch.float64)
    return t_nodes.unsqueeze(1) ** exps.unsqueeze(0)


def _exact_shapley_enum(D, v_table):
    phi = torch.zeros(D, dtype=torch.float64)
    D_fact = math.factorial(D)
    for i in range(D):
        for mask in range(1 << D):
            if (mask >> i) & 1:
                continue
            s = mask.bit_count()
            weight = math.factorial(s) * math.factorial(D - s - 1) / D_fact
            v_on = v_table[mask | (1 << i)]
            v_off = v_table[mask]
            phi[i] = phi[i] + weight * (v_on - v_off)
    return phi


def _agg_s_i(D, v_table):
    agg = torch.zeros((D, D), dtype=torch.float64)
    for i in range(D):
        for mask in range(1 << D):
            if (mask >> i) & 1:
                continue
            s = mask.bit_count()
            agg[i, s] += v_table[mask | (1 << i)] - v_table[mask]
    return agg


def _tnshap_gi_vandermonde(D, K, x, a, b, coeffs, t_nodes):
    max_degree = K - 1
    V = _build_vandermonde(t_nodes, degree_max=max_degree)
    alphas = _shapley_weights(D)
    m = torch.zeros((D, max_degree + 1), dtype=torch.float64)
    phi = torch.zeros(D, dtype=torch.float64)

    for i in range(D):
        h = torch.zeros(t_nodes.shape[0], dtype=torch.float64)
        for idx_t, t in enumerate(t_nodes):
            u_base = a * (t * x) + b
            u_on = u_base.clone()
            u_off = u_base.clone()
            u_on[i] = a[i] * x[i] + b[i]
            u_off[i] = b[i]
            h[idx_t] = _eval_g_from_u(u_on, coeffs) - _eval_g_from_u(u_off, coeffs)

        m_i, *_ = torch.linalg.lstsq(V, h.unsqueeze(-1))
        m_i = m_i.squeeze(-1)
        m[i] = m_i
        phi[i] = torch.sum(alphas[: max_degree + 1] * m_i)

    return phi, m


def test_diag_t1_selectors_oracle():
    torch.manual_seed(0)
    D = 6
    K = 3
    max_degree = K - 1
    t_nodes = torch.linspace(0.0, 1.0, steps=max_degree + 1, dtype=torch.float64)
    seeds = [0, 1, 2]

    for seed in seeds:
        rng = torch.Generator()
        rng.manual_seed(seed)
        x = torch.empty(D, dtype=torch.float64).uniform_(-1.0, 1.0, generator=rng)
        a = torch.empty(D, dtype=torch.float64).uniform_(-1.0, 1.0, generator=rng)
        b = torch.empty(D, dtype=torch.float64).uniform_(-1.0, 1.0, generator=rng)
        coeffs = _make_coeffs(D, K, seed=seed + 123)

        # Precompute v(S) table for all subsets using off=[0,1].
        v_table = torch.zeros(1 << D, dtype=torch.float64)
        for mask in range(1 << D):
            u = b.clone()
            for j in range(D):
                if (mask >> j) & 1:
                    u[j] = a[j] * x[j] + b[j]
            v_table[mask] = _eval_g_from_u(u, coeffs)

        phi_enum = _exact_shapley_enum(D, v_table)
        phi_tn, m = _tnshap_gi_vandermonde(D, K, x, a, b, coeffs, t_nodes)

        err = (phi_tn - phi_enum).abs()
        max_abs = err.max().item()
        phi_ok = torch.allclose(phi_tn, phi_enum, rtol=1e-8, atol=1e-8)

        # Size-aggregated deltas vs coefficients mapping:
        # agg_s_i = sum_{u=0..min(s,max_degree)} C(D-1-u, s-u) * m_i[u]
        agg = _agg_s_i(D, v_table)
        comb = lambda n, k: math.comb(n, k) if 0 <= k <= n else 0
        pred = torch.zeros_like(agg)
        for i in range(D):
            for s in range(D):
                total = 0.0
                for u in range(0, min(s, max_degree) + 1):
                    total += comb(D - 1 - u, s - u) * m[i, u].item()
                pred[i, s] = total

        coeff_err = (pred - agg).abs()
        coeff_max = coeff_err.max().item()
        coeff_ok = torch.allclose(pred, agg, rtol=1e-8, atol=1e-8)

        if not (phi_ok and coeff_ok):
            i0 = 0
            table = [
                f"s={s}: m_i[s]={m[i0, s].item():.6e}, "
                f"agg_s_i={agg[i0, s].item():.6e}, pred={pred[i0, s].item():.6e}"
                for s in range(max_degree + 1)
            ]
            raise AssertionError(
                "TN-SHAP oracle mismatch.\n"
                f"seed={seed}\n"
                f"phi_max_abs_err={max_abs:.3e}, per_i_err={err.tolist()}\n"
                f"coeff_max_abs_err={coeff_max:.3e}\n"
                + "\n".join(table)
            )

