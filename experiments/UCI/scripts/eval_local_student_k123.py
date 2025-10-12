#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# -*- coding: utf-8 -*-
"""
Evaluate TN selector (FewEval/Vandermonde) vs Teacher Exact (interventional, zero baseline)
for k in {1,2,3} on the *same* K target points used to build the local student.

Reads from the build folder (e.g., ./out_local_student_singlegrid/...):
  - t_nodes_shared.npy          (preferred; single grid for all k)
    or t_nodes_order{1,2,3}.npy (fallbacks)
  - teacher.pt, tn.pt
  - manifest.json  (to see K, D, etc.)

Writes:
  - per-idx CSVs: <dataset>_idx{idx}_order{k}_local_eval.csv
  - a summary CSV across all idx: <dataset>_summary_local_eval.csv

Optionally computes baselines (KernelSHAP/SHAPIQ) if --with-baselines is passed.
"""

import os, json, math, time, argparse, warnings, glob
from typing import Optional, Tuple, Dict, List, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import platform
import subprocess
import socket
# add near top with other imports
from src.feature_mapped_tn import FeatureMappedTN  # zero-preserving ψ wrapper



# ----------------- Globals & utils -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def cosine_sim(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 and nb < 1e-12: return 1.0
    if na < 1e-12 or nb < 1e-12:  return 0.0
    return float(np.dot(a, b) / (na * nb))

def get_hardware_info():
    """Collect comprehensive hardware and system information for documentation."""
    info = {}
    
    # Basic system info
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.platform()
    info['architecture'] = platform.architecture()[0]
    info['processor'] = platform.processor()
    info['python_version'] = platform.python_version()
    info['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # CPU info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpu_lines = f.readlines()
        cpu_model = [line.strip() for line in cpu_lines if 'model name' in line]
        if cpu_model:
            info['cpu_model'] = cpu_model[0].split(':')[1].strip()
        cpu_count = [line.strip() for line in cpu_lines if 'processor' in line]
        info['cpu_cores'] = len(cpu_count)
    except:
        info['cpu_model'] = 'Unknown'
        info['cpu_cores'] = os.cpu_count()
    
    # Memory info
    try:
        with open('/proc/meminfo', 'r') as f:
            mem_lines = f.readlines()
        for line in mem_lines:
            if 'MemTotal' in line:
                mem_kb = int(line.split()[1])
                info['memory_gb'] = round(mem_kb / (1024**2), 1)
                break
    except:
        info['memory_gb'] = 'Unknown'
    
    # GPU info
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
    
    # PyTorch info
    info['torch_version'] = torch.__version__
    
    # Environment info
    try:
        info['conda_env'] = os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')
        info['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES', 'All')
    except:
        info['conda_env'] = 'Unknown'
        info['cuda_visible_devices'] = 'Unknown'
    
    return info

def save_hardware_info(outdir: str, filename: str = "hardware_info.json"):
    """Save hardware information to a JSON file."""
    hardware_info = get_hardware_info()
    filepath = os.path.join(outdir, filename)
    with open(filepath, 'w') as f:
        json.dump(hardware_info, f, indent=2)
    print(f"[Hardware] Saved hardware info to: {filepath}")
    return hardware_info

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

# ----------------- Dataset (same split logic as builder) -----------------
def _split_and_standardize(X: np.ndarray, y: np.ndarray, seed: int):
    X = X.astype(np.float64); y = y.astype(np.float64)
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

# ----------------- Models: Teacher + TN -----------------
class MLPRegressor(nn.Module):
    def __init__(self, d_in: int, hidden=(256,256,128), pdrop=0.0):
        super().__init__()
        layers=[]; prev=d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(pdrop)]
            prev=h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def _load_teacher(tea_path: str, d_hint: int) -> nn.Module:
    obj = torch.load(tea_path, map_location="cpu")
    if isinstance(obj, nn.Module): model = obj
    else:
        sd = obj.get("state_dict", obj)
        model = MLPRegressor(d_in=d_hint)
        model.load_state_dict(sd, strict=False)
    return model.to(DEVICE).eval()

try:
    from src.tntree_model import BinaryTensorTree
except Exception:
    BinaryTensorTree = None

def _infer_tn_rank_from_sd(sd):
    for k,v in sd.items():
        if k.startswith("cores.") and isinstance(v, torch.Tensor):
            shp = tuple(v.shape)
            if len(shp) in (2,3): return int(shp[0])
    return 16

def _infer_rank_from_tn_state(sd):
    # works for both bare TN and wrapped TN (keys under 'tn.cores.*')
    for k, v in sd.items():
        if k.startswith(("cores.", "tn.cores.")) and isinstance(v, torch.Tensor):
            shp = tuple(v.shape)
            if len(shp) in (2, 3):  # (r, ...) or (r,2,2)
                return int(shp[0])
    return 16

def _is_wrapper_state_dict(sd):
    # FeatureMappedTN exposes params under 'feature_map.*' and 'tn.*'
    return any(k.startswith("feature_map.") for k in sd.keys())

def _strip_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

def _load_tn(stu_path: str, d_hint: int) -> nn.Module:
    if BinaryTensorTree is None:
        raise RuntimeError("tntree_model not available")

    obj = torch.load(stu_path, map_location="cpu")

    # Case 1: the file is already a constructed module (rare)
    if isinstance(obj, nn.Module):
        return obj.to(DEVICE).eval()

    # Case 2: state dict-ish object
    sd = obj.get("state_dict", obj)

    if _is_wrapper_state_dict(sd):
        # ---- Full FeatureMappedTN was saved ----
        tn_sd = _strip_prefix(sd, "tn.")
        fmap_sd = _strip_prefix(sd, "feature_map.")
        r = _infer_rank_from_tn_state(sd)

        tn_core = BinaryTensorTree(
            [2]*d_hint, ranks=r, out_dim=1,
            assume_bias_when_matrix=True, device=DEVICE
        ).to(DEVICE)
        tn_core.load_state_dict(tn_sd, strict=False)

        model = FeatureMappedTN(
            tn=tn_core, d_in=d_hint, fmap_hidden=32, fmap_act="relu"
        ).to(DEVICE)
        model.feature_map.load_state_dict(fmap_sd, strict=False)
        model.eval()
        return model

    # ---- Bare TN core (legacy) ----
    r = _infer_rank_from_tn_state(sd)
    tn = BinaryTensorTree(
        [2]*d_hint, ranks=r, out_dim=1,
        assume_bias_when_matrix=True, device=DEVICE
    )
    tn.load_state_dict(sd, strict=False)
    return tn.to(DEVICE).eval()


# ----------------- Exact Teacher (interventional, zero baseline) -----------------
def _factorial(n: int) -> int:
    out = 1
    for i in range(2, n+1): out *= i
    return out

def _weight_u(u_size: int, n: int, k: int) -> float:
    # |U|!(n-|U|-k)! / (n-k+1)!
    num = _factorial(u_size) * _factorial(max(n - u_size - k, 0))
    den = _factorial(max(n - k + 1, 1))
    return num / den

@torch.no_grad()
def exact_kth_interaction_model(model: nn.Module, x: np.ndarray, k: int, *, device=None):
    """Exact interventional SII/SV of order k on 'model' at 'x' with baseline 0."""
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
            if w == 0.0:
                continue
            for U in _comb(rest, u_size):
                delta = 0.0
                for s in range(0, k+1):
                    sign = (-1.0) ** (k - s)
                    # sign = (-1.0) ** len(S)
                    for W in _comb(T, s):
                        P = tuple(sorted(set(U).union(W)))
                        delta += sign * f_present_tuple(P)
                total += w * delta
        phi[idx_T] = total

    timing = {"t_exact_model_s": float(t_eval), "n_model_calls": f_present_tuple.cache_info().currsize}
    return phi, timing

# ----------------- TN Selector (FewEval) with one shared grid -----------------
@torch.no_grad()
def tn_selector_any_k_sharedgrid(
    model: nn.Module,
    x: np.ndarray,
    t_nodes: np.ndarray,
    k: int,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Selector-path TN of order k using a *single* Chebyshev grid (t_nodes) for all orders.
    Steps:
      1) Compute power-basis coeffs c^(S) for every subset S with |S|<=k,
         where S means "mask these features to 0" along the t-path.
      2) For each T (|T|=k): c^T_path = sum_{S⊆T} (-1)^(|T|-|S|) c^(S)
      3) Integrate: phi_T = sum_{r>=k} c^T_path[r] / C(r, k)
    """
    device = device or DEVICE
    x = np.asarray(x, np.float32).ravel()
    d = x.size
    assert 1 <= k <= d

    # Grid + Vandermonde inverse
    t = torch.tensor(np.asarray(t_nodes, np.float32), device=device)
    m = int(t.numel())
    V = torch.vander(t, N=m, increasing=True)  # [m, m]
    _sync(); t0 = time.perf_counter()
    Vinv = torch.linalg.inv(V)
    _sync(); t_solve = time.perf_counter() - t0

    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    # Build coeffs for all S with |S|<=k (including empty)
    subs_by_size = {s: list(combinations(range(d), s)) for s in range(0, k+1)}
    coeffs: Dict[Tuple[int, ...], np.ndarray] = {}

    # Global path (S=∅)
    Xg = t.unsqueeze(1) * x_t.unsqueeze(0)  # [m, d]
    _sync(); t0 = time.perf_counter()
    yg = model(Xg).squeeze(-1)              # [m]
    _sync(); t_eval = time.perf_counter() - t0
    c_empty = (Vinv @ yg.unsqueeze(1)).squeeze(1)
    coeffs[()] = c_empty.detach().cpu().numpy().astype(np.float64)

    # Masked paths for s=1..k
    for s in range(1, k+1):
        subs = subs_by_size[s]
        if not subs: continue
        # Batch all masked paths for current s
        Xh = Xg.repeat(len(subs), 1)  # [(len(subs)*m), d]
        for r, S in enumerate(subs):
            if S:
                Xh[r*m:(r+1)*m, list(S)] = 0.0
        _sync(); t0 = time.perf_counter()
        yh = model(Xh).squeeze(-1).view(len(subs), m)  # [len(subs), m]
        _sync(); t_eval += time.perf_counter() - t0
        # Solve coeffs
        _sync(); t0 = time.perf_counter()
        cS = (yh @ Vinv.T)                              # [len(subs), m]
        _sync(); t_solve += time.perf_counter() - t0
        for S, c in zip(subs, cS):
            coeffs[tuple(S)] = c.detach().cpu().numpy().astype(np.float64)

    # For each T with |T|=k: inclusion–exclusion and integrate
    all_T = list(combinations(range(d), k))
    phi = np.zeros(len(all_T), dtype=np.float64)
    weights = np.zeros(m, dtype=np.float64)
    for r in range(k, m):
        weights[r] = 1.0 / math.comb(r, k)

    for idx_T, T in enumerate(all_T):
        cT = np.zeros(m, dtype=np.float64)
        Tset = set(T)
        # sum over S⊆T
        for s in range(0, k+1):
            for S in combinations(T, s):
                # sign = (-1.0) ** (k - s)  # == (-1)^(|T|-|S|)
                sign = (-1.0) ** len(S)

                cT += sign * coeffs[S]
        phi[idx_T] = float(np.dot(cT[k:], weights[k:]))

    timing = {"t_eval_s": float(t_eval), "t_solve_s": float(t_solve), "t_total_s": float(t_eval + t_solve)}
    return phi, timing

# ----------------- Optional baselines -----------------
def _maybe_kernel_shap_k1(predict_np, x, d, nsamples, seed):
    import shap
    rng = np.random.default_rng(seed); np.random.seed(int(rng.integers(0, 2**31-1)))
    expl = shap.KernelExplainer(predict_np, np.zeros((1,d)))
    t0 = time.perf_counter()
    phi = expl.shap_values(np.asarray(x).reshape(1,-1), nsamples=nsamples)[0]
    return np.asarray(phi, float), (time.perf_counter() - t0)

def _maybe_shapiq_any_order(predict_np, X_bg, x, k, budget, approximator, index_name, seed):
    import shapiq
    rng = np.random.default_rng(seed)
    class _NpModel: 
        def __init__(self,fn): self.fn=fn
        def predict(self, X): return self.fn(np.asarray(X, np.float32))
    expl = shapiq.TabularExplainer(
        model=_NpModel(predict_np),
        data=X_bg,
        approximator=approximator,   # 'regression' | 'permutation' | 'montecarlo'
        index=index_name,            # 'SV' | 'SII' | 'FSII'
        max_order=k,
        random_state=int(rng.integers(0, 2**31-1))
    )
    t0=time.perf_counter()
    iv = expl.explain(np.asarray(x,float).reshape(1,-1), budget=budget)
    dt=time.perf_counter()-t0
    if k==1:
        if index_name == "SV":
            vals = np.asarray(iv.get_values(), float).ravel()
            return vals, dt
        tens = iv.get_n_order_values(1)
        return np.asarray(tens, float).ravel(), dt
    tens = iv.get_n_order_values(k)
    d0 = tens.shape[0]
    from itertools import combinations as comb
    return np.asarray([tens[tuple(T)] for T in comb(range(d0),k)], float), dt

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["concrete","energy_y1","energy_y2","diabetes","california"])
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--masked-root", required=True, help="Folder produced by build_local_student_singlegrid.py")
    ap.add_argument("--orders", type=int, nargs="+", default=[1,2,3])
    ap.add_argument("--outdir", type=str, required=True)

    # baselines (off by default)
    ap.add_argument("--with-baselines", action="store_true")
    ap.add_argument("--kernel-budgets", type=int, nargs="+", default=[512])
    ap.add_argument("--shapiq-budgets", type=int, nargs="+", default=[2048])
    ap.add_argument("--repeats", type=int, default=1)
    
    # index range arguments
    ap.add_argument("--start-idx", type=int, default=0, help="Starting index for evaluation")
    ap.add_argument("--end-idx", type=int, default=None, help="Ending index for evaluation (exclusive)")

    args = ap.parse_args()
    set_all_seeds(args.seed)

    # ----- locate nodes and checkpoints -----
    root = args.masked_root
    nodes_shared = os.path.join(root, "t_nodes_shared.npy")
    if os.path.isfile(nodes_shared):
        t_nodes = np.load(nodes_shared)
        per_order_nodes = None
        print(f"[nodes] Using shared grid: {nodes_shared}  (m={t_nodes.size})")
    else:
        alt1 = os.path.join(root, "t_nodes_order1.npy")
        alt2 = os.path.join(root, "t_nodes_order2.npy")
        alt3 = os.path.join(root, "t_nodes_order3.npy")
        if not all(os.path.isfile(p) for p in (alt1, alt2, alt3)):
            raise FileNotFoundError("No t_nodes_shared.npy and missing per-order node files.")
        t_nodes = None
        per_order_nodes = {
            1: np.load(alt1),
            2: np.load(alt2),
            3: np.load(alt3),
        }
        print(f"[nodes] Using per-order grids: m1={per_order_nodes[1].size}, m2={per_order_nodes[2].size}, m3={per_order_nodes[3].size}")

    tea_path = os.path.join(root, "teacher.pt")
    tn_path  = os.path.join(root, "tn.pt")
    if not os.path.isfile(tea_path): raise FileNotFoundError(tea_path)
    if not os.path.isfile(tn_path):  raise FileNotFoundError(tn_path)

    # ----- load dataset, rebuild same K targets -----
    name, Xtr, ytr, Xva, yva, Xte, yte, sx, sy = load_dataset_by_name(args.dataset, args.seed)
    d = Xte.shape[1]
    # Recreate chosen indices the same way builder did
    manifest_path = os.path.join(root, "manifest.json")
    if os.path.isfile(manifest_path):
        man = json.load(open(manifest_path, "r"))
        K = int(man.get("kpoints", min(100, Xte.shape[0])))
    else:
        warnings.warn("manifest.json not found; assuming K from test size or default 100.")
        K = min(100, Xte.shape[0])

    rng = np.random.default_rng(args.seed)
    n_te = Xte.shape[0]
    chosen = rng.choice(n_te, size=min(K, n_te), replace=False)
    Xtargets = Xte[chosen].astype(np.float32)
    print(f"[targets] Using K={Xtargets.shape[0]} targets (reconstructed from seed & split).")

    # ----- load models -----
    teacher = _load_teacher(tea_path, d_hint=d)
    tn      = _load_tn(tn_path, d_hint=d)

    def teacher_predict_np(Z):
        with torch.no_grad():
            zt = torch.tensor(np.asarray(Z, np.float32), device=DEVICE)
            return teacher(zt).detach().cpu().numpy()

    # ----- output -----
    out_root = os.path.join(args.outdir, f"{args.dataset}_seed{args.seed}")
    ensure_dir(out_root)

    # Save hardware information for documentation
    hardware_info = save_hardware_info(out_root)
    
    summary_rows = []

    # Determine index range
    end_idx = args.end_idx if args.end_idx is not None else len(Xtargets)
    start_idx = max(0, args.start_idx)
    end_idx = min(end_idx, len(Xtargets))
    
    print(f"[evaluation] Processing indices {start_idx} to {end_idx-1} (total: {end_idx-start_idx} points)")

    # ----- evaluation -----
    for local_idx in range(start_idx, end_idx):
        x = Xtargets[local_idx]
        for k in args.orders:
            # TN selector via shared/per-order grid
            if t_nodes is not None:
                phi_tn_sel, tinfo = tn_selector_any_k_sharedgrid(tn, x, t_nodes, k=k)
            else:
                phi_tn_sel, tinfo = tn_selector_any_k_sharedgrid(tn, x, per_order_nodes[k], k=k)

            # Teacher exact
            phi_ex, t_exact = exact_kth_interaction_model(teacher, x, k=k)

            # metrics
            cos = cosine_sim(phi_ex, phi_tn_sel)
            r2  = r2_score_vec(phi_ex, phi_tn_sel)
            mse = mse_vec(phi_ex, phi_tn_sel)

            rows = [{
                "method": "tn_selector", "baseline": "TNShap", "budget": np.nan,
                "time_s_mu": float(tinfo["t_total_s"]), "time_s_sd": 0.0,
                "cos_vs_exact_mu": float(cos), "cos_vs_exact_sd": 0.0,
                "r2_vs_exact_mu":  float(r2),  "r2_vs_exact_sd":  0.0,
                "mse_vs_exact_mu": float(mse), "mse_vs_exact_sd": 0.0,
            }]

            # Optional baselines (teacher)
            if args.with_baselines:
                # KernelSHAP (k=1 only)
                if k == 1:
                    for B in args.kernel_budgets:
                        cos_l, r2_l, mse_l, times = [], [], [], []
                        for rep in range(args.repeats):
                            try:
                                vals, dt = _maybe_kernel_shap_k1(teacher_predict_np, x, d, nsamples=B,
                                                                 seed=args.seed + 101*rep + B)
                                times.append(dt)
                                cos_l.append(cosine_sim(phi_ex, vals))
                                r2_l.append(r2_score_vec(phi_ex, vals))
                                mse_l.append(mse_vec(phi_ex, vals))
                            except Exception:
                                times.append(np.nan); cos_l.append(np.nan); r2_l.append(np.nan); mse_l.append(np.nan)
                        rows.append({
                            "method":"kernel_native","baseline":"KernelSHAP","budget":B,
                            "time_s_mu": float(np.nanmean(times)), "time_s_sd": float(np.nanstd(times)),
                            "cos_vs_exact_mu": float(np.nanmean(cos_l)), "cos_vs_exact_sd": float(np.nanstd(cos_l)),
                            "r2_vs_exact_mu":  float(np.nanmean(r2_l)),  "r2_vs_exact_sd":  float(np.nanstd(r2_l)),
                            "mse_vs_exact_mu": float(np.nanmean(mse_l)), "mse_vs_exact_sd": float(np.nanstd(mse_l)),
                        })
                # SHAPIQ
                SHAPIQ = [
                    ("shapiq_reg_sii","KernelSHAPIQ (Reg SII)","regression","SII"),
                    ("shapiq_reg_fsii","RegressionFSII","regression","FSII"),
                    ("shapiq_perm_sii","PermutationSamplingSII","permutation","SII"),
                    ("shapiq_mc_sii","SHAPIQ (MonteCarlo SII)","montecarlo","SII"),
                ]
                for (mkey, disp, approximator, index_name) in SHAPIQ:
                    for B in args.shapiq_budgets:
                        cos_l, r2_l, mse_l, times = [], [], [], []
                        for rep in range(args.repeats):
                            try:
                                vals, dt = _maybe_shapiq_any_order(teacher_predict_np, Xtr, x, k=k, budget=B,
                                                                   approximator=approximator, index_name=index_name,
                                                                   seed=args.seed + 202*rep + B)
                                times.append(dt)
                                cos_l.append(cosine_sim(phi_ex, vals))
                                r2_l.append(r2_score_vec(phi_ex, vals))
                                mse_l.append(mse_vec(phi_ex, vals))
                            except Exception:
                                times.append(np.nan); cos_l.append(np.nan); r2_l.append(np.nan); mse_l.append(np.nan)
                        rows.append({
                            "method":mkey,"baseline":disp,"budget":B,
                            "time_s_mu": float(np.nanmean(times)), "time_s_sd": float(np.nanstd(times)),
                            "cos_vs_exact_mu": float(np.nanmean(cos_l)), "cos_vs_exact_sd": float(np.nanstd(cos_l)),
                            "r2_vs_exact_mu":  float(np.nanmean(r2_l)),  "r2_vs_exact_sd":  float(np.nanstd(r2_l)),
                            "mse_vs_exact_mu": float(np.nanmean(mse_l)), "mse_vs_exact_sd": float(np.nanstd(mse_l)),
                        })

            df = pd.DataFrame(rows)
            df.insert(0, "dataset", args.dataset)
            df.insert(1, "seed", args.seed)
            df.insert(2, "idx", int(local_idx))   # local target index (0..K-1)
            df.insert(3, "order_k", int(k))
            df["time_exact_s_teacher"] = float(t_exact["t_exact_model_s"])
            
            # Add hardware info columns
            df["hostname"] = hardware_info["hostname"]
            df["gpu_name"] = hardware_info["gpu_name"] 
            df["gpu_memory_gb"] = hardware_info["gpu_memory_gb"]
            df["cpu_model"] = hardware_info["cpu_model"]
            df["cpu_cores"] = hardware_info["cpu_cores"]
            df["memory_gb"] = hardware_info["memory_gb"]
            df["torch_version"] = hardware_info["torch_version"]
            df["cuda_version"] = hardware_info["cuda_version"]
            df["timestamp"] = hardware_info["timestamp"]
            
            out_csv = os.path.join(out_root, f"{args.dataset}_idx{local_idx}_order{k}_local_eval.csv")
            df.to_csv(out_csv, index=False)
            print(f"[idx={local_idx} k={k}] -> {out_csv}")

            # add to summary
            best_row = df.iloc[0].to_dict()
            summary_rows.append(best_row)

    # summary file (TN selector rows only)
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        out_sum = os.path.join(out_root, f"{args.dataset}_summary_local_eval.csv")
        sdf.to_csv(out_sum, index=False)
        print(f"[summary] -> {out_sum}")

if __name__ == "__main__":
    main()
