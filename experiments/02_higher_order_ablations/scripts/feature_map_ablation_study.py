#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# -*- coding: utf-8 -*-
"""
Train-and-evaluate ablation over feature-map output sizes (m_fmap) on one dataset.

For each m in M_LIST:
  - build (or reuse) the masked K-point single-grid dataset
  - train a FeatureMappedTN (per-leaf phys_dim = m+1)
      * record training time
      * if final train R^2 < threshold, retry with new seeds up to N times
  - save run artifacts in: <outdir>/<dataset>_seed<seed>_K<K>/mgrid<mgrid>_mfmap<m>/
  - evaluate TN selector vs exact teacher at k=1,2,3 on the same K targets
  - write per-idx CSVs and a per-run summary (includes train time and tn_seed)
Finally, write a MASTER ablation CSV across all m with per-k averages.

Requires:
  - tntree_model.py (upgraded) and feature_mapped_tn.py (upgraded)
"""

import os, json, math, time, argparse, warnings, itertools, socket, platform
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.tntree_model import BinaryTensorTree
from src.feature_mapped_tn import make_feature_mapped_tn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- utils -----------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
def chebyshev_nodes_01(m: int) -> np.ndarray:
    if m <= 0: return np.zeros((0,), dtype=np.float32)
    k = np.arange(m, dtype=np.float64)
    nodes = np.cos((2*k + 1) * np.pi / (2*m))
    return ((nodes + 1.0) * 0.5).astype(np.float32)
def cosine_sim(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 and nb < 1e-12: return 1.0
    if na < 1e-12 or nb < 1e-12:  return 0.0
    return float(np.dot(a, b) / (na * nb))
def r2_score_vec(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return float(1.0 - ss_res / (ss_tot + 1e-12))
def mse_vec(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))
def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def get_hardware_info():
    info = {}
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.platform()
    info['architecture'] = platform.architecture()[0]
    info['processor'] = platform.processor()
    info['python_version'] = platform.python_version()
    info['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    info['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        info['current_gpu_device'] = str(DEVICE)
    else:
        info['cuda_version'] = 'N/A'
        info['gpu_count'] = 0
        info['gpu_name'] = 'N/A'
        info['gpu_memory_gb'] = 'N/A'
        info['current_gpu_device'] = str(DEVICE)
    info['torch_version'] = torch.__version__
    info['conda_env'] = os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')
    info['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES', 'All')
    return info

# ----------------- data -----------------
def _split_and_standardize(X: np.ndarray, y: np.ndarray, seed: int):
    X=X.astype(np.float64); y=y.astype(np.float64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed)
    sx = StandardScaler().fit(Xtr2)
    sy = StandardScaler().fit(ytr2.reshape(-1,1))
    return (sx.transform(Xtr2).astype(np.float32), sy.transform(ytr2.reshape(-1,1)).ravel().astype(np.float32),
            sx.transform(Xva).astype(np.float32),  sy.transform(yva.reshape(-1,1)).ravel().astype(np.float32),
            sx.transform(Xte).astype(np.float32),  sy.transform(yte.reshape(-1,1)).ravel().astype(np.float32),
            sx, sy)

def load_dataset_by_name(name: str, seed: int):
    name = name.lower()
    if name == "diabetes":
        from sklearn.datasets import load_diabetes
        ds = load_diabetes(); return (name,*_split_and_standardize(ds.data.astype(float), ds.target.astype(float), seed))
    if name == "california":
        from sklearn.datasets import fetch_california_housing
        ds = fetch_california_housing(); return (name,*_split_and_standardize(ds.data.astype(float), ds.target.astype(float), seed))
    if name == "concrete":
        try:
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=165)
            X = ds.data.features.to_numpy(float); y = ds.data.targets.to_numpy(float).ravel()
        except Exception as e:
            warnings.warn(f"[concrete] fetch failed ({e}); using synthetic fallback.")
            rng=np.random.default_rng(0); n,d=1200,8
            X=rng.normal(size=(n,d))
            y=(15+8*X[:,0]-4*X[:,1]+6*np.tanh(X[:,2])+3*X[:,0]*X[:,1]-2*X[:,3]*X[:,4]+5*np.sin(X[:,5])+2*X[:,6]**2-3*X[:,0]*X[:,2]*X[:,6]+rng.normal(0,0.6,n))
        return (name,*_split_and_standardize(X,y,seed))
    if name in ("energy_y1","energy_y2"):
        try:
            from ucimlrepo import fetch_ucirepo
            ds=fetch_ucirepo(name="Energy efficiency")
            X=ds.data.features.to_numpy(float); Y=ds.data.targets.to_numpy(float)
            y = (Y[:,0] if name=="energy_y1" else Y[:,1]).ravel()
        except Exception as e:
            warnings.warn(f"[{name}] fetch failed ({e}); using synthetic fallback.")
            rng=np.random.default_rng(1 if name=="energy_y1" else 2); n,d=768,8
            X=rng.normal(size=(n,d))
            y=(10+3*X[:,0]-2*X[:,1]+1.5*np.sin(X[:,2])+0.8*X[:,3]*X[:,4]+np.random.default_rng(0).normal(0,0.5,n))
            if name=="energy_y2": y=y+0.7*np.tanh(X[:,5])-1.2*X[:,6]*X[:,1]
        return (name,*_split_and_standardize(X,y,seed))
    raise ValueError(f"Unknown dataset {name}")

# ----------------- teacher -----------------
class MLPRegressor(nn.Module):
    def __init__(self, d_in: int, hidden=(256,256,128), pdrop=0.05):
        super().__init__()
        layers=[]; prev=d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(pdrop)]
            prev=h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def train_teacher(Xtr, ytr, Xva, yva, seed=0, max_epochs=400, patience=60, lr=3e-3):
    set_all_seeds(seed)
    model = MLPRegressor(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40, T_mult=2, eta_min=1e-5)
    crit = nn.SmoothL1Loss(beta=0.5)
    Xtr_t = torch.tensor(Xtr, device=DEVICE); ytr_t = torch.tensor(ytr, device=DEVICE)
    Xva_t = torch.tensor(Xva, device=DEVICE); yva_t = torch.tensor(yva, device=DEVICE)
    best = {"state": None, "val": 1e9}; noimp = 0
    for ep in range(1, max_epochs+1):
        model.train(); opt.zero_grad(set_to_none=True)
        loss = crit(model(Xtr_t), ytr_t); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
        model.eval()
        with torch.no_grad(): v = crit(model(Xva_t), yva_t).item()
        if v < best["val"] - 1e-12:
            best = {"state": {k:v_.detach().cpu().clone() for k,v_ in model.state_dict().items()}, "val": v}; noimp=0
        else:
            noimp += 1
            if noimp >= patience: break
    model.load_state_dict(best["state"]); model.eval()
    return model

# ----------------- exact interventional k-way -----------------
def _factorial(n: int) -> int:
    out = 1
    for i in range(2, n+1): out *= i
    return out
def _weight_u(u_size: int, n: int, k: int) -> float:
    num = _factorial(u_size) * _factorial(max(n - u_size - k, 0))
    den = _factorial(max(n - k + 1, 1))
    return num / den
@torch.no_grad()
def exact_kth_interaction_model(model: nn.Module, x: np.ndarray, k: int, *, device=None):
    from functools import lru_cache
    from itertools import combinations as _comb
    device = device or DEVICE
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device).flatten()
    n = x_t.numel()
    t_eval = 0.0
    @lru_cache(None)
    def f_present_tuple(P_tuple):
        P = list(P_tuple)
        X = x_t.clone()
        if P:
            off = [i for i in range(n) if i not in P]
            if off: X[off] = 0.0
        else:
            X = torch.zeros_like(x_t)
        _sync(); t0 = time.perf_counter()
        y = model(X.unsqueeze(0)).squeeze(-1)
        _sync()
        nonlocal t_eval
        t_eval += (time.perf_counter() - t0)
        return float(y.detach().item())
    all_T = list(_comb(range(n), k))
    phi = np.zeros(len(all_T), dtype=np.float64)
    for idx_T, T in enumerate(all_T):
        Tset = set(T)
        rest = [i for i in range(n) if i not in Tset]
        total = 0.0
        for u_size in range(0, len(rest)+1):
            w = _weight_u(u_size, n, k)
            if w == 0.0: continue
            for U in _comb(rest, u_size):
                delta = 0.0
                for s in range(0, k+1):
                    sign = (-1.0) ** (k - s)
                    for W in _comb(T, s):
                        P = tuple(sorted(set(U).union(W)))
                        delta += sign * f_present_tuple(P)
                total += w * delta
        phi[idx_T] = total
    timing = {"t_exact_model_s": float(t_eval)}
    return phi, timing

# ----------------- TN selector via single grid -----------------
@torch.no_grad()
def tn_selector_any_k_sharedgrid(model: nn.Module, x: np.ndarray, t_nodes: np.ndarray, k: int, *, device=None):
    device = device or DEVICE
    x = np.asarray(x, np.float32).ravel()
    d = x.size
    assert 1 <= k <= d
    t = torch.tensor(np.asarray(t_nodes, np.float32), device=device)
    m = int(t.numel())
    V = torch.vander(t, N=m, increasing=True)
    _sync(); t0 = time.perf_counter()
    Vinv = torch.linalg.inv(V)
    _sync(); t_solve = time.perf_counter() - t0
    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    from itertools import combinations
    subs_by_size = {s: list(combinations(range(d), s)) for s in range(0, k+1)}
    coeffs: Dict[Tuple[int, ...], np.ndarray] = {}

    Xg = t.unsqueeze(1) * x_t.unsqueeze(0)  # [m,d]
    _sync(); t0 = time.perf_counter()
    yg = model(Xg).squeeze(-1)              # [m]
    _sync(); t_eval = time.perf_counter() - t0
    c_empty = (Vinv @ yg.unsqueeze(1)).squeeze(1)
    coeffs[()] = c_empty.detach().cpu().numpy().astype(np.float64)

    for s in range(1, k+1):
        subs = subs_by_size[s]
        if not subs: continue
        Xh = Xg.repeat(len(subs), 1)
        for r, S in enumerate(subs):
            if S:
                Xh[r*m:(r+1)*m, list(S)] = 0.0
        _sync(); t0 = time.perf_counter()
        yh = model(Xh).squeeze(-1).view(len(subs), m)
        _sync(); t_eval += time.perf_counter() - t0
        _sync(); t0 = time.perf_counter()
        cS = (yh @ Vinv.T)
        _sync(); t_solve += time.perf_counter() - t0
        for S, c in zip(subs, cS):
            coeffs[tuple(S)] = c.detach().cpu().numpy().astype(np.float64)

    all_T = list(combinations(range(d), k))
    phi = np.zeros(len(all_T), dtype=np.float64)
    weights = np.zeros(m, dtype=np.float64)
    for r in range(k, m):
        weights[r] = 1.0 / math.comb(r, k)
    for idx_T, T in enumerate(all_T):
        cT = np.zeros(m, dtype=np.float64)
        for s in range(0, k+1):
            for S in itertools.combinations(T, s):
                cT += ((-1.0) ** len(S)) * coeffs[S]   # inclusion-exclusion
        phi[idx_T] = float(np.dot(cT[k:], weights[k:]))
    timing = {"t_eval_s": float(t_eval), "t_solve_s": float(t_solve), "t_total_s": float(t_eval + t_solve)}
    return phi, timing

# ----------------- training TN (returns timing) -----------------
def train_tn_on_masked(masked_X: np.ndarray, y_teacher: np.ndarray, ranks: int, seed: int,
                       fmap_out_dim: int, fmap_hidden: int = 32, lr=2e-2, max_epochs=50,
                       tol=1e-7, amp=True, batch_size=None, print_every=10, patience=50):
    set_all_seeds(seed)
    dev = DEVICE
    X_t = torch.tensor(masked_X, device=dev)
    y_t = torch.tensor(y_teacher, device=dev)
    d = masked_X.shape[1]
    model = make_feature_mapped_tn(
        d_in=d, fmap_out_dim=fmap_out_dim, ranks=ranks, out_dim=1,
        fmap_hidden=fmap_hidden, fmap_act="relu",
        selector_mode="none", seed=seed, device=dev, dtype=torch.float32
    ).to(dev)
    n = X_t.shape[0]
    if batch_size is None:
        batch_size = n if n <= 2048 else min(16384, max(512, n // 16))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and dev.type=="cuda"))
    crit = nn.MSELoss()
    idx = torch.arange(n, device=dev)
    best_state = None
    best_mse = float('inf'); noimp = 0
    y_mean = y_t.mean()
    var_y = torch.var(y_t, unbiased=True).clamp_min(1e-12)
    def eval_full():
        with torch.no_grad():
            pred = model(X_t).squeeze(-1)
            mse_val = crit(pred, y_t).item()
            r2_val = 1.0 - ((pred - y_t).pow(2).mean() / var_y).item()
        return mse_val, r2_val
    t0 = time.perf_counter()
    for ep in range(1, max_epochs+1):
        model.train()
        if batch_size >= n:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and dev.type=="cuda")):
                pred = model(X_t).squeeze(-1)
                loss = crit(pred, y_t)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update()
            mse_epoch = loss.item()
        else:
            perm = idx[torch.randperm(n, device=dev)]
            total_loss = 0.0; seen = 0
            for s in range(0, n, batch_size):
                sel = perm[s:s+batch_size]
                xb = X_t.index_select(0, sel)
                yb = y_t.index_select(0, sel)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(amp and dev.type=="cuda")):
                    pred = model(xb).squeeze(-1)
                    loss = crit(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt); scaler.update()
                total_loss += loss.item() * xb.size(0)
                seen += xb.size(0)
            mse_epoch = total_loss / max(seen,1)
        if mse_epoch < best_mse - 1e-12:
            best_mse = mse_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            noimp = 0
        else:
            noimp += 1
        if (ep % print_every == 0) or ep == 1:
            mse_full, r2_full = eval_full()
            print(f"[TN fmap={fmap_out_dim} seed={seed}] ep {ep:3d} mse={mse_epoch:.3e} best={best_mse:.3e} full={mse_full:.3e} R2={r2_full:.4f}")
        if mse_epoch < tol or noimp > patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - t0
    final_mse, final_r2 = eval_full()
    return model, dict(final_mse=final_mse, final_r2=final_r2, best_mse=best_mse, epochs=ep, train_time_s=elapsed)

# ----------------- main orchestration -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["concrete","energy_y1","energy_y2","diabetes","california"])
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--kpoints", type=int, default=100)
    ap.add_argument("--outdir", type=str, default="./out_local_student_singlegrid")
    ap.add_argument("--grid-m", type=int, default=None, help="Chebyshev nodes for t-grid (shared). Default: n (features)")
    ap.add_argument("--m-fmap-list", type=str, default="1,2,4,8", help="Comma-separated list, e.g., '1,2,4,8'")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--max-triplets", type=int, default=None)
    # NEW: retry policy
    ap.add_argument("--train-r2-min", type=float, default=0.7, help="Minimum acceptable train R^2 before retrying with a new seed")
    ap.add_argument("--train-retries", type=int, default=3, help="Max number of extra seeds to try if R^2 < threshold")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    hw = get_hardware_info()

    # load data + teacher
    name, Xtr, ytr, Xva, yva, Xte, yte, sx, sy = load_dataset_by_name(args.dataset, args.seed)
    d = Xte.shape[1]
    teacher = train_teacher(Xtr, ytr, Xva, yva, seed=args.seed, lr=3e-3)

    # targets
    rng = np.random.default_rng(args.seed)
    n_te = Xte.shape[0]; K = min(args.kpoints, n_te)
    chosen = rng.choice(n_te, size=K, replace=False)
    Xtargets = Xte[chosen].astype(np.float32)

    # grid
    mgrid = int(args.grid_m) if args.grid_m is not None else int(d)
    t_nodes = chebyshev_nodes_01(mgrid)

    # masked dataset (shared across all m_fmap)
    D = d
    masked_rows = []; meta_rows = []; row_idx = 0
    pairs = list(itertools.combinations(range(D), 2))
    trips = list(itertools.combinations(range(D), 3))
    if args.max_pairs is not None and args.max_pairs < len(pairs):
        rng2 = np.random.default_rng(args.seed + 13)
        pairs = sorted(list(rng2.choice(pairs, size=args.max_pairs, replace=False)))
    if args.max_triplets is not None and args.max_triplets < len(trips):
        rng3 = np.random.default_rng(args.seed + 17)
        trips = sorted(list(rng3.choice(trips, size=args.max_triplets, replace=False)))
    for r in range(K):
        x = Xtargets[r]
        # k=1
        for i in range(D):
            for ell, t in enumerate(t_nodes):
                base = (t * x).astype(np.float32)
                x1 = base.copy()
                x0 = base.copy(); x0[i] = 0.0
                masked_rows.extend([x1, x0])
                meta_rows.append([row_idx, r, 1, str((i,)), ell, float(t), "1"]); row_idx += 1
                meta_rows.append([row_idx, r, 1, str((i,)), ell, float(t), "0"]); row_idx += 1
        # k=2
        for (i,j) in pairs:
            for ell, t in enumerate(t_nodes):
                base = (t * x).astype(np.float32)
                x11 = base.copy()
                x10 = base.copy(); x10[j] = 0.0
                x01 = base.copy(); x01[i] = 0.0
                x00 = base.copy(); x00[i] = 0.0; x00[j] = 0.0
                masked_rows.extend([x11,x10,x01,x00])
                meta_rows.append([row_idx, r, 2, str((i,j)), ell, float(t), "11"]); row_idx+=1
                meta_rows.append([row_idx, r, 2, str((i,j)), ell, float(t), "10"]); row_idx+=1
                meta_rows.append([row_idx, r, 2, str((i,j)), ell, float(t), "01"]); row_idx+=1
                meta_rows.append([row_idx, r, 2, str((i,j)), ell, float(t), "00"]); row_idx+=1
        # k=3
        for (i,j,k) in trips:
            for ell, t in enumerate(t_nodes):
                base = (t * x).astype(np.float32)
                for a in (1,0):
                    for b in (1,0):
                        for c in (1,0):
                            z = base.copy()
                            if a==0: z[i] = 0.0
                            if b==0: z[j] = 0.0
                            if c==0: z[k] = 0.0
                            masked_rows.append(z)
                            meta_rows.append([row_idx, r, 3, str((i,j,k)), ell, float(t), f"{a}{b}{c}"]); row_idx += 1
    masked_X = np.stack(masked_rows, axis=0).astype(np.float32)
    with torch.no_grad():
        xt = torch.tensor(masked_X, device=DEVICE)
        y_teacher = teacher(xt).detach().cpu().numpy().astype(np.float32)

    # prep output
    base_root = os.path.join(args.outdir, f"{args.dataset}_seed{args.seed}_K{K}")
    ensure_dir(base_root)

    # master ablation rows (averages over K for each k and m)
    master_rows = []

    m_list = [int(s) for s in args.m_fmap_list.split(",") if s.strip()]
    for m_fmap in m_list:
        run_dir = os.path.join(base_root, f"mgrid{mgrid}_mfmap{m_fmap}")
        ensure_dir(run_dir)
        # save shared artifacts in each run dir for simplicity
        np.save(os.path.join(run_dir, "masked_X.npy"), masked_X)
        np.save(os.path.join(run_dir, "y_teacher.npy"), y_teacher)
        np.save(os.path.join(run_dir, "t_nodes_shared.npy"), t_nodes)
        with open(os.path.join(run_dir, "meta.csv"), "w") as f:
            import csv
            w = csv.writer(f); w.writerow(["row_index","target_idx","order_k","subset","t_index","t_value","toggle"])
            for row in meta_rows: w.writerow(row)
        torch.save(teacher.state_dict(), os.path.join(run_dir, "teacher.pt"))

        # ------- train TN with retry policy -------
        r2_min = float(args.train_r2_min)
        max_retries = int(args.train_retries)
        used_seed = int(args.seed)
        used_info = None
        tn = None

        for attempt in range(0, max_retries + 1):
            if attempt > 0:
                # new seed offset to diversify
                used_seed = int(args.seed + 100 * attempt)
                print(f"[RETRY] fmap={m_fmap}: attempting seed {used_seed} (attempt {attempt}/{max_retries})")
            tn, info = train_tn_on_masked(
                masked_X, y_teacher, ranks=args.rank, seed=used_seed,
                fmap_out_dim=m_fmap, fmap_hidden=32, lr=2e-2, max_epochs=50,
                tol=1e-7, amp=True, batch_size=None, print_every=10, patience=50
            )
            used_info = info
            if info["final_r2"] >= r2_min:
                print(f"[OK] fmap={m_fmap} seed={used_seed}: train R^2={info['final_r2']:.4f} >= {r2_min}")
                break
            else:
                print(f"[LOW] fmap={m_fmap} seed={used_seed}: train R^2={info['final_r2']:.4f} < {r2_min}")
        # save TN after last attempt (either passing threshold or last retry)
        torch.save(tn.state_dict(), os.path.join(run_dir, "tn.pt"))

        # manifest
        manifest = dict(
            dataset=args.dataset, seed=int(args.seed), kpoints=int(K),
            rank=int(args.rank), grid_nodes=int(mgrid), dim=int(d),
            masked_rows=int(masked_X.shape[0]), device=str(DEVICE),
            tn_info=used_info, fmap_out_dim=int(m_fmap), selector_mode="none",
            tn_seed=used_seed, train_time_s=float(used_info.get("train_time_s", float("nan"))),
            train_r2=float(used_info.get("final_r2", float("nan")))
        )
        with open(os.path.join(run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        # ------- evaluation per k and per idx -------
        per_k_scores = {1: [], 2: [], 3: []}
        for local_idx in range(K):
            x = Xtargets[local_idx]
            for k in (1,2,3):
                phi_tn, tinfo = tn_selector_any_k_sharedgrid(tn, x, t_nodes, k=k)
                phi_ex, t_exact = exact_kth_interaction_model(teacher, x, k=k)
                cos = cosine_sim(phi_ex, phi_tn)
                r2  = r2_score_vec(phi_ex, phi_tn)
                mse = mse_vec(phi_ex, phi_tn)
                per_k_scores[k].append((cos, r2, mse))

                # write per-idx CSV like your evaluator
                df = pd.DataFrame([{
                    "dataset": args.dataset, "seed": args.seed, "tn_seed": used_seed,
                    "idx": int(local_idx), "order_k": int(k),
                    "method": "tn_selector", "baseline": "TNShap", "budget": np.nan,
                    "time_s_mu": float(tinfo["t_total_s"]), "time_s_sd": 0.0,
                    "cos_vs_exact_mu": float(cos), "cos_vs_exact_sd": 0.0,
                    "r2_vs_exact_mu":  float(r2),  "r2_vs_exact_sd":  0.0,
                    "mse_vs_exact_mu": float(mse), "mse_vs_exact_sd": 0.0,
                    "time_exact_s_teacher": float(t_exact["t_exact_model_s"]),
                    "train_time_s": float(used_info.get("train_time_s", float("nan"))),
                    "train_final_r2": float(used_info.get("final_r2", float("nan"))),
                    "train_best_mse": float(used_info.get("best_mse", float("nan"))),
                    "hostname": hw["hostname"], "gpu_name": hw["gpu_name"],
                    "gpu_memory_gb": hw["gpu_memory_gb"], "cpu_model": hw.get("processor",""),
                    "torch_version": hw["torch_version"], "cuda_version": hw["cuda_version"],
                    "timestamp": hw["timestamp"]
                }])
                out_csv = os.path.join(run_dir, f"{args.dataset}_idx{local_idx}_order{k}_local_eval.csv")
                df.to_csv(out_csv, index=False)

        # per-run summary: average over idx for each k, plus training stats/seed
        rows = []
        for k in (1,2,3):
            if per_k_scores[k]:
                arr = np.asarray(per_k_scores[k], float)  # [K, 3]
                rows.append({
                    "dataset": args.dataset, "seed": args.seed, "tn_seed": used_seed,
                    "fmap_out_dim": m_fmap, "order_k": k,
                    "cos_vs_exact_mu": float(np.mean(arr[:,0])), "cos_vs_exact_sd": float(np.std(arr[:,0])),
                    "r2_vs_exact_mu":  float(np.mean(arr[:,1])), "r2_vs_exact_sd":  float(np.std(arr[:,1])),
                    "mse_vs_exact_mu": float(np.mean(arr[:,2])), "mse_vs_exact_sd": float(np.std(arr[:,2])),
                    "train_final_mse": used_info["final_mse"], "train_final_r2": used_info["final_r2"],
                    "train_time_s": used_info.get("train_time_s", float("nan"))
                })
        sdf = pd.DataFrame(rows)
        sdf.to_csv(os.path.join(run_dir, f"{args.dataset}_summary_local_eval.csv"), index=False)

        # add to master ablation
        for r in rows:
            master_rows.append(r)

        print(f"[OK] {args.dataset} mfmap={m_fmap} seed_used={used_seed} -> {run_dir}")

    # write MASTER ablation CSV
    master_df = pd.DataFrame(master_rows).sort_values(["order_k","fmap_out_dim"])
    master_csv = os.path.join(base_root, f"{args.dataset}_ablation_fmap_eval_seed{args.seed}.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"[MASTER] {master_csv}")

if __name__ == "__main__":
    main()
