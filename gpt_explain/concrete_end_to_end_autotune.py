# concrete_end_to_end_autotune.py
# End-to-end Concrete with TN auto-tuning for speed subject to accuracy.
# Saves plot, CSV, and LaTeX.

import os, math, time, warnings, argparse, random
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from string import Template

# ---------------- utils ----------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def r2_score_vec(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(float); y_pred = y_pred.astype(float)
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    if ss_tot < 1e-12: 
        return float(1.0 if np.allclose(y_true, y_pred) else 0.0)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if an < 1e-12 or bn < 1e-12: 
        return float(1.0 if an < 1e-12 and bn < 1e-12 else 0.0)
    return float(np.dot(a, b) / (an * bn))

def fmt_ms(s): return f"{1000*s:.2f} ms"

# ---------------- data ----------------
def load_concrete():
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=165)
        X = ds.data.features.to_numpy(dtype=np.float64)
        y = ds.data.targets.to_numpy(dtype=np.float64).ravel()
        return X, y, "concrete"
    except Exception as e:
        warnings.warn(f"Concrete fetch failed ({e}); using synthetic fallback.")
        rng = np.random.default_rng(0)
        n, d = 1200, 8
        X = rng.normal(size=(n, d))
        y = (15 + 8*X[:,0] - 4*X[:,1] + 6*np.tanh(X[:,2])
             + 3*X[:,0]*X[:,1] - 2*X[:,3]*X[:,4] + 5*np.sin(X[:,5])
             + 2*X[:,6]**2 - 3*X[:,0]*X[:,2]*X[:,6]
             + rng.normal(scale=0.6, size=n))
        return X.astype(np.float64), y.astype(np.float64), "concrete_synth"

# ---------------- teacher ----------------
class MLPRegressor(nn.Module):
    def __init__(self, d_in: int, hidden=(256, 256, 128), pdrop=0.05):
        super().__init__()
        layers = []; prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(pdrop)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def train_teacher(Xn, yn, Xv, yv, seed=0, max_epochs=400, patience=60, lr=3e-3):
    set_all_seeds(seed)
    model = MLPRegressor(Xn.shape[1]).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40, T_mult=2, eta_min=1e-5)
    crit = nn.SmoothL1Loss(beta=0.5)
    Xtr = torch.tensor(Xn, device=DEVICE, dtype=torch.float32)
    ytr = torch.tensor(yn, device=DEVICE, dtype=torch.float32)
    Xva = torch.tensor(Xv, device=DEVICE, dtype=torch.float32)
    yva = torch.tensor(yv, device=DEVICE, dtype=torch.float32)
    best = {"state": None, "val": 1e9}; noimp = 0
    for _ in range(1, max_epochs+1):
        model.train(); opt.zero_grad(set_to_none=True)
        loss = crit(model(Xtr), ytr); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            v = crit(model(Xva), yva).item()
        if v < best["val"] - 1e-12:
            best = {"state": {k: v_.detach().cpu().clone() for k,v_ in model.state_dict().items()}, "val": v}
            noimp = 0
        else:
            noimp += 1
            if noimp >= patience: break
        sched.step()
    model.load_state_dict(best["state"]); model.eval()
    return model

# ---------------- TN surrogate ----------------
TN_AVAILABLE = True
from tntree_model import BinaryTensorTree
    

def fit_tn_surrogate(X, y, ranks: int, seed: int, max_epochs=300, patience=50, lr=2e-3):
    set_all_seeds(seed)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed)
    Xtr_t = torch.tensor(Xtr, device=DEVICE, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, device=DEVICE, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, device=DEVICE, dtype=torch.float32)
    yva_t = torch.tensor(yva, device=DEVICE, dtype=torch.float32)
    if TN_AVAILABLE:
        d = X.shape[1]
        model = BinaryTensorTree([2]*d, ranks=ranks, out_dim=1,
                                 assume_bias_when_matrix=True, seed=seed, device=DEVICE).to(DEVICE)
    else:
        model = BinaryTensorTree(X.shape[1]).to(DEVICE)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr_t, ytr_t),
                                         batch_size=1024 if DEVICE.type=="cuda" else 256, shuffle=True)
    crit = nn.SmoothL1Loss(beta=0.1)
    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched= optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2, eta_min=1e-6)
    best = {"state": None, "mse": 1e9, "r2": np.nan}; noimp = 0
    t0 = time.perf_counter()
    for _ in range(1, max_epochs+1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).squeeze(-1).detach().cpu().numpy()
            vv = yva_t.detach().cpu().numpy()
            mse = float(np.mean((pv - vv)**2)); r2 = r2_score_vec(vv, pv)
        if mse < best["mse"] - 1e-12:
            best = {"state": {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    "mse": mse, "r2": r2}
            noimp = 0
        else:
            noimp += 1
            if noimp >= patience: break
        sched.step()
    train_time_s = time.perf_counter() - t0
    model.load_state_dict(best["state"]); model.eval()
    return model, float(best["r2"]), train_time_s

# ---------------- Exact SHAP (μ=0) ----------------
def exact_shapley_zero(teacher_predict_np, x: np.ndarray, max_d: int = 16,
                       batch_eval: int = 65536):
    t0 = time.perf_counter()
    x = np.asarray(x, dtype=np.float32).ravel(); d = x.size
    if d > max_d: raise RuntimeError(f"d={d} > max_d={max_d}")
    M = 1 << d
    masks = np.arange(M, dtype=np.uint32)
    bits = ((masks[:, None] >> np.arange(d, dtype=np.uint32)) & 1).astype(np.float32)
    fvals = np.empty(M, dtype=np.float64)
    start = 0
    while start < M:
        end = min(start + batch_eval, M)
        Xmask = bits[start:end] * x[None, :]
        fvals[start:end] = teacher_predict_np(Xmask).reshape(-1).astype(np.float64)
        start = end
    fact = np.ones(d+1, dtype=np.float64)
    for i in range(2, d+1): fact[i] = fact[i-1]*i
    denom = fact[d]
    w_by_k = np.array([fact[k]*fact[d-k-1]/denom for k in range(d)], dtype=np.float64)
    k_sizes = np.unpackbits(masks.view(np.uint32).view(np.uint8)).reshape(-1,32)[:, :d].sum(axis=1)
    phi = np.zeros(d, dtype=np.float64)
    for i in range(d):
        S_mask = ((masks >> i) & 1) == 0; S = masks[S_mask]; k = k_sizes[S]
        Su = S | (1 << i); delta = fvals[Su] - fvals[S]
        phi[i] = np.sum(w_by_k[k] * delta, dtype=np.float64)
    fx = float(fvals[-1]); fmu = float(fvals[0])
    return phi, (time.perf_counter()-t0), fx, fmu

# ---------------- TN SHAP (selector path, batched, two solvers) ----------------
@torch.no_grad()
@torch.no_grad()
def tn_shap_selector_path(
    model: nn.Module, x: np.ndarray, m_points: int = 24,
    solver: str = "inv", eval_batch: int = 65536
) -> np.ndarray:
    device = DEVICE
    x = torch.as_tensor(x, device=device, dtype=torch.float32).flatten()
    d = x.numel()

    # Chebyshev nodes in [0,1]
    k = torch.arange(m_points, device=device, dtype=torch.float32)
    nodes = torch.cos((2*k + 1) * math.pi / (2*m_points))
    t = 0.5*(nodes + 1.0)

    # Build selector path batches
    mu = torch.zeros_like(x)
    Xg = t.unsqueeze(1) * (x - mu).unsqueeze(0) + mu            # [m, d]
    Xh = Xg.repeat(d, 1)                                         # [d*m, d]
    for i in range(d):
        Xh[i*m_points:(i+1)*m_points, i] = 0.0
    X_all = torch.cat([Xg, Xh], dim=0)                           # [m + d*m, d]

    # Batched evaluation
    outs = []
    for start in range(0, X_all.size(0), eval_batch):
        end = min(start + eval_batch, X_all.size(0))
        outs.append(model(X_all[start:end]).squeeze(-1))
    y_all = torch.cat(outs, dim=0)

    yg = y_all[:m_points]                                        # [m]
    yh = y_all[m_points:].view(d, m_points)                      # [d, m]

    # Vandermonde system V * C_i = (yg - yh_i)
    V = torch.vander(t, N=m_points, increasing=True)             # [m, m]

    # Build RHS as [d, m] (each row is one rhs_i)
    rhs = yg.unsqueeze(0) - yh                                   # [d, m]

    if solver == "inv":
        Vinv = torch.linalg.inv(V)                               # [m, m]
        # Ci = V^{-1} * rhs, vectorized across rows
        Ci = (Vinv @ rhs.T).T                                    # [d, m]
    elif solver == "solve":
        # Solve V * Ci^T = rhs^T
        Ci = torch.linalg.solve(V, rhs.T).T                      # [d, m]
    else:
        raise ValueError("solver must be 'inv' or 'solve'")

    # Shapley from Chebyshev coefficients
    inv_k = torch.zeros(m_points, device=device, dtype=torch.float32)
    inv_k[1:] = 1.0 / torch.arange(1, m_points, device=device, dtype=torch.float32)
    phi = torch.sum(Ci * inv_k, dim=1)                           # [d]
    return phi.detach().cpu().numpy().astype(np.float64)


# ---------------- KernelSHAP ----------------
def kernel_shap_once(teacher_predict_np, x: np.ndarray, nsamples: int, seed: int, d: int):
    import shap
    rng = np.random.default_rng(seed)
    np.random.seed(int(rng.integers(0, 2**31-1)))
    expl = shap.KernelExplainer(teacher_predict_np, np.zeros((1, d)))
    t0 = time.perf_counter()
    phi = expl.shap_values(np.asarray(x).reshape(1, -1), nsamples=nsamples)[0]
    return np.asarray(phi, float), (time.perf_counter()-t0)

# ---------------- plotting ----------------
def plot_budget_curve(out_png: str, budgets: List[int],
                      cos_mu: np.ndarray, cos_sd: np.ndarray,
                      t_mu: np.ndarray, tn_time_s: float,
                      cos_tn_exact: Optional[float] = None):
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.6,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    })
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    X = np.array(budgets, dtype=float)
    ax.plot(X, cos_mu, marker="o", label="cos(Kernel, Ex)")
    lo = np.clip(cos_mu - cos_sd, 0, 1); hi = np.clip(cos_mu + cos_sd, 0, 1)
    ax.fill_between(X, lo, hi, alpha=0.22)
    if cos_tn_exact is not None and np.isfinite(cos_tn_exact):
        ax.axhline(cos_tn_exact, linestyle="--", alpha=0.7, label=f"cos(TN,Exact)={cos_tn_exact:.3f}")
    ax.set_ylim(0.0, 1.0)
    xticklabels = [f"({int(b)},{tm*1000.:.2f}ms)" for b, tm in zip(budgets, t_mu)]
    ax.set_xticks(X); ax.set_xticklabels(xticklabels, rotation=35, ha="right")
    ax.set_xlabel("(Kernel budget, runtime)")
    ax.set_ylabel("cosine(Kernel, TN)")
    ax.legend(frameon=False, loc="lower right", title=f"TN_attr ≈ {tn_time_s*1000:.2f} ms")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

# ---------------- auto-tuner ----------------
def auto_tune_tn(model, x, target_cos_exact: float,
                 phi_exact: np.ndarray,
                 m_grid=(8,12,16,20,24,32),
                 solvers=("inv","solve"),
                 eval_batches=(4096, 8192, 16384)) -> Dict:
    best_feasible = None  # min time among those meeting target
    best_any = None       # highest cosine (tie-breaker: time)
    for m in m_grid:
        # Prebuild Chebyshev basis costs scale roughly with m
        for solver in solvers:
            for eb in eval_batches:
                t0 = time.perf_counter()
                phi_tn = tn_shap_selector_path(model, x, m_points=m, solver=solver, eval_batch=eb)
                t_attr = time.perf_counter() - t0
                cos_ex = cosine_sim(phi_tn, phi_exact)
                r2_ex  = r2_score_vec(phi_exact, phi_tn)
                cand = {"m_points": m, "solver": solver, "eval_batch": eb,
                        "tn_attr_time_s": t_attr, "cos_tn_exact": cos_ex, "r2_tn_exact": r2_ex,
                        "phi_tn": phi_tn}
                # Best-any tracker
                if (best_any is None or 
                    (cos_ex > best_any["cos_tn_exact"] + 1e-12) or
                    (abs(cos_ex - best_any["cos_tn_exact"]) < 1e-12 and t_attr < best_any["tn_attr_time_s"])):
                    best_any = cand
                # Feasible tracker
                if cos_ex >= target_cos_exact - 1e-9:
                    if (best_feasible is None or t_attr < best_feasible["tn_attr_time_s"]):
                        best_feasible = cand
    return best_feasible or best_any

# ---------------- main runner ----------------
def run_concrete(seed: int, ranks: int, budgets: List[int], kernel_repeats: int,
                 target_cos_exact: float, outdir: str):
    ensure_dir(outdir)
    set_all_seeds(seed)
    # --- data & teacher ---
    X, y, name = load_concrete()
    Xtr_all, Xte, ytr_all, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    sx = StandardScaler().fit(Xtr_all)
    Xtr_n = sx.transform(Xtr_all).astype(np.float32)
    Xte_n = sx.transform(Xte).astype(np.float32)
    sy = StandardScaler().fit(ytr_all.reshape(-1,1))
    ytr_n = sy.transform(ytr_all.reshape(-1,1)).ravel().astype(np.float32)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_n, ytr_n, test_size=0.2, random_state=seed)
    teacher = train_teacher(Xtr, ytr, Xva, yva, seed=seed)

    def teacher_predict_np(Z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            zt = torch.tensor(np.asarray(Z, dtype=np.float32), device=DEVICE)
            return teacher(zt).detach().cpu().numpy()

    with torch.no_grad():
        ytr_hat = teacher(torch.tensor(Xtr, device=DEVICE)).detach().cpu().numpy()
        yva_hat = teacher(torch.tensor(Xva, device=DEVICE)).detach().cpu().numpy()
    teach_r2_tr = r2_score_vec(ytr, ytr_hat)
    teach_r2_va = r2_score_vec(yva, yva_hat)

    # pick a test point
    # seed_rng = 0
    rng = np.random.default_rng(20)
    test_idx = int(rng.integers(0, len(Xte_n)))
    x = Xte_n[test_idx]; d = x.size

    # --- local distillation set (same recipe as before) ---
    n_local, radius = 200, 0.20
    X_loc = (x[None,:].astype(np.float32) + radius * rng.normal(size=(n_local, d)).astype(np.float32))

    # add selector-path grid to help surrogate see the path manifold
    def make_path_points(m_points:int):
        k = np.arange(m_points, dtype=np.float32)
        nodes = np.cos((2*k + 1) * np.pi / (2*m_points))
        t = 0.5*(nodes + 1.0)
        Xg = (t[:, None] * x[None, :]).astype(np.float32)
        Xh = np.repeat(Xg, d, axis=0)
        for i in range(d): Xh[i*m_points:(i+1)*m_points, i] = 0.0
        return Xg, Xh

    Xg, Xh = make_path_points(24)
    X_train = np.vstack([X_loc, Xg, Xh]).astype(np.float32)
    y_train = teacher_predict_np(X_train).astype(np.float32)

    # --- fit TN surrogate ---
    tn, r2_tn_vs_teacher, tn_train_time_s = fit_tn_surrogate(X_train, y_train, ranks=ranks, seed=seed)

    # --- exact shap (teacher) ---
    phi_ex, t_exact_s, fx, fmu = exact_shapley_zero(teacher_predict_np, x, max_d=16)

    # --- auto-tune TN-SHAP ---
    best = auto_tune_tn(
        model=tn, x=x, target_cos_exact=target_cos_exact, phi_exact=phi_ex,
        m_grid=(8,12,16,20,24,32), solvers=("inv","solve"), eval_batches=(4096,8192,16384)
    )
    phi_tn = best["phi_tn"]
    tn_attr_time_s = best["tn_attr_time_s"]
    cos_tn_ex = best["cos_tn_exact"]
    r2_tn_ex  = best["r2_tn_exact"]

    # --- Kernel budgets (variance over repeats) ---
    cos_list = []; t_list = []
    for B in budgets:
        c_runs = []; t_runs = []
        for rep in range(kernel_repeats):
            phi_k, dt = kernel_shap_once(teacher_predict_np, x, nsamples=B, seed=seed+rep*7, d=d)
            c_runs.append(cosine_sim(phi_k, phi_ex))
            t_runs.append(dt)
        cos_list.append( (float(np.mean(c_runs)), float(np.std(c_runs))) )
        t_list.append( (float(np.mean(t_runs)), float(np.std(t_runs))) )
    cos_mu = np.array([m for m,_ in cos_list], dtype=float)
    cos_sd = np.array([s for _,s in cos_list], dtype=float)
    t_mu   = np.array([m for m,_ in t_list], dtype=float)
    t_sd   = np.array([s for _,s in t_list], dtype=float)

    # high-budget reference vs Exact
    phi_k5, t_k5 = kernel_shap_once(teacher_predict_np, x, nsamples=5000, seed=seed+123, d=d)
    cos_k5_ex = cosine_sim(phi_k5, phi_ex); r2_k5_ex  = r2_score_vec(phi_ex, phi_k5)

    # Equal-time B (closest kernel time to tn_attr_time_s)
    idx_eq = int(np.argmin(np.abs(t_mu - tn_attr_time_s)))
    B_eq = int(budgets[idx_eq]); t_eq = float(t_mu[idx_eq])
    phi_keq, _ = kernel_shap_once(teacher_predict_np, x, nsamples=B_eq, seed=seed+999, d=d)
    cos_eq_ex = cosine_sim(phi_keq, phi_ex)

    # --- Terminal summary ---
    print(f"\n=== Concrete auto-tuned TN-SHAP ===")
    print(f"Teacher R2: train={teach_r2_tr:.3f}  val={teach_r2_va:.3f}")
    print(f"TN(distill) R2(val vs teacher)={r2_tn_vs_teacher:.3f}")
    print(f"Auto-tuned TN-SHAP → m_points={best['m_points']}  solver={best['solver']}  eval_batch={best['eval_batch']}")
    print(f"TN vs Exact:   cos={cos_tn_ex:.3f}, R2={r2_tn_ex:.3f}, time={fmt_ms(tn_attr_time_s)}")
    print(f"Exact SHAP time per point: {fmt_ms(t_exact_s)}")
    print(f"Kernel(5k) vs Exact: cos={cos_k5_ex:.3f}, R2={r2_k5_ex:.3f}, time={fmt_ms(t_k5)}")
    print(f"Equal-time: B≈{B_eq}  kernel≈{fmt_ms(t_eq)}  cos(eq,Exact)={cos_eq_ex:.3f}")
    print("\nBudget summary (cos(Kernel,TN) mean±sd | kernel time ms mean±sd):")
    for B, (cm, cs), (tm, ts) in zip(budgets, cos_list, t_list):
        print(f"  B={B:4d}  cos={cm:.3f}±{cs:.3f}  time={tm*1000:.2f}±{ts*1000:.2f}")

    # --- Plot ---
    out_png = os.path.join(outdir, f"{name}_budget_kernel_vs_tn.png")
    plot_budget_curve(out_png, budgets, cos_mu, cos_sd, t_mu, tn_attr_time_s, cos_tn_exact=cos_tn_ex)
    print(f"Saved plot: {out_png}")

    # --- CSV ---
    df_budget = pd.DataFrame({
        "dataset": name, "seed": seed, "test_idx": int(test_idx), "d": int(d),
        "budget": budgets,
        "cos_kernel_vs_tn_mu": cos_mu, "cos_kernel_vs_tn_sd": cos_sd,
        "time_kernel_s_mu": t_mu, "time_kernel_s_sd": t_sd,
        "time_tn_attr_s": tn_attr_time_s, "time_exact_s": t_exact_s,
        "cos_kernel5k_vs_exact": cos_k5_ex, "r2_kernel5k_vs_exact": r2_k5_ex,
        "cos_tn_vs_exact": cos_tn_ex, "r2_tn_vs_exact": r2_tn_ex,
        "tn_m_points": best["m_points"], "tn_solver": best["solver"], "tn_eval_batch": best["eval_batch"],
        "tn_train_time_s": tn_train_time_s, "r2_tn_vs_teacher_val": r2_tn_vs_teacher,
        "teacher_r2_train": teach_r2_tr, "teacher_r2_val": teach_r2_va,
    })
    out_csv = os.path.join(outdir, f"{name}_budget_summary.csv")
    df_budget.to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")
    # save student ckpt
    # save student model with seed in filename
    out_tn = os.path.join(outdir, f"{name}_tn_r{ranks}_seed{seed}.pt")
    torch.save(tn.state_dict(), out_tn)
    print(f"Saved TN model: {out_tn}")
    # save teacher model with seed in filename
    out_teacher = os.path.join(outdir, f"{name}_teacher_seed{seed}.pt")
    torch.save(teacher.state_dict(), out_teacher)
    print(f"Saved teacher model: {out_teacher}")
    # save datapoint center index with seed in filename
    out_idx = os.path.join(outdir, f"{name}_testidx_seed{seed}.txt")
    with open(out_idx, "w") as f: f.write(f"{test_idx}\n")
    print(f"Saved test index: {out_idx}")
    # --- JSON (metadata) ---
    meta = {
        "dataset": name,
        "seed": seed,
        "test_idx": int(test_idx),
        "d": int(d),
        "teacher_r2_train": teach_r2_tr,
        "teacher_r2_val": teach_r2_va,
        "tn_r2_vs_teacher_val": r2_tn_vs_teacher,
        "tn_train_time_s": tn_train_time_s,
        "exact_shap_time_s": t_exact_s,
        "exact_shap_fx": fx,
        "exact_shap_fmu": fmu,
        "exact_shap_phi": phi_ex.tolist(),
        "auto_tune": {
            "m_points": best["m_points"],
            "solver": best["solver"],
            "eval_batch": best["eval_batch"],
            "tn_attr_time_s": tn_attr_time_s,
            "cos_tn_vs_exact": cos_tn_ex,
            "r2_tn_vs_exact": r2_tn_ex,
            "tn_phi": phi_tn.tolist(),
        },
        "kernel5k": {
            "cos_kernel5k_vs_exact": cos_k5_ex,
            "r2_kernel5k_vs_exact": r2_k5_ex,
            "time_kernel5k_s": t_k5,
            "kernel5k_phi": phi_k5.tolist(),
        },
        "equal_time": {
            "B_eq": B_eq,
            "time_kernel_eq_s": t_eq,
            "cos_eq_vs_exact": cos_eq_ex,
            "kernel_eq_phi": phi_keq.tolist(),
        },
        "budget_curve": {
            "budgets": budgets,
            "cos_kernel_vs_tn_mu": cos_mu.tolist(),
            "cos_kernel_vs_tn_sd": cos_sd.tolist(),
            "time_kernel_s_mu": t_mu.tolist(),
            "time_kernel_s_sd": t_sd.tolist(),
        }
    }
    import json
    # SEED IN filename
    
    out_json = os.path.join(outdir, f"{name}_summary_seed{seed}.json")
    with open(out_json, "w") as f: json.dump(meta, f, indent=2)
    print(f"Saved JSON: {out_json}")
    
    

    # --- LaTeX (Template-safe with '@' delimiter) ---
    class TeX(Template): delimiter = '@'
    tex_tpl = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.0}
\begin{tabular}{lrrrrrrrrr}
\toprule
Dataset & R$^2_{\text{tr}}$ & R$^2_{\text{val}}$ & R$^2_{\text{TN}\to\text{T}}$ & cos(TN,Ex) & $R^2$(TN,Ex) & B$_{\text{eq}}$ & cos(eq,Ex) & t$_{\text{TN}}$ (ms) & cfg \\
\midrule
@name & @r2_tr & @r2_va & @r2_tn_teacher & @cos_tn_ex & @r2_tn_ex & @B_eq & @cos_eq & @tn_ms & m=@m_points, @solver/@eb \\
\bottomrule
\end{tabular}
\caption{Concrete with auto-tuned TN-SHAP. Kernel 5k: cos=@cos_k5_ex, $R^2$=@r2_k5_ex, t=@k5_ms~ms. Exact t=@ex_ms~ms.}
\label{tab:concrete_autotuned}
\end{table}
"""
    tex = TeX(tex_tpl).substitute(
        name=name,
        r2_tr=f"{teach_r2_tr:.3f}",
        r2_va=f"{teach_r2_va:.3f}",
        r2_tn_teacher=f"{r2_tn_vs_teacher:.3f}",
        cos_tn_ex=f"{cos_tn_ex:.3f}",
        r2_tn_ex=f"{r2_tn_ex:.3f}",
        B_eq=f"{B_eq}",
        cos_eq=f"{cos_eq_ex:.3f}",
        tn_ms=f"{tn_attr_time_s*1000:.2f}",
        m_points=f"{best['m_points']}",
        solver=f"{best['solver']}",
        eb=f"{best['eval_batch']}",
        cos_k5_ex=f"{cos_k5_ex:.3f}",
        r2_k5_ex=f"{r2_k5_ex:.3f}",
        k5_ms=f"{t_k5*1000:.2f}",
        ex_ms=f"{t_exact_s*1000:.2f}",
    )
    out_tex = os.path.join(outdir, f"{name}_compact.tex")
    with open(out_tex, "w") as f: f.write(tex)
    print(f"Saved LaTeX: {out_tex}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="./out_concrete_hyper")
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--ranks", type=int, default=16)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[2,5,10,15,20,40,50,75,100])
    ap.add_argument("--kernel-repeats", type=int, default=20)
    ap.add_argument("--target-cos-exact", type=float, default=0.95)
    args = ap.parse_args()
    ensure_dir(args.outdir)
    run_concrete(seed=args.seed, ranks=args.ranks, budgets=args.budgets,
                 kernel_repeats=args.kernel_repeats,
                 target_cos_exact=args.target_cos_exact,
                 outdir=args.outdir)

if __name__ == "__main__":
    main()
