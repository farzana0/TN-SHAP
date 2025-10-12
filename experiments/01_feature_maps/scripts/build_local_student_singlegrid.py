#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# -*- coding: utf-8 -*-
"""
End-to-end: build a single-grid FewEval masked dataset (k=1,2,3), train teacher,
overfit TN on masked rows only, and save everything in one folder.

Saves:
  - masked_X.npy          # [M, D] masked inputs for all k using ONE shared t-grid
  - y_teacher.npy         # [M] teacher(x) on the masked_X
  - meta.csv              # row_index,target_idx,order_k,subset,t_index,t_value,toggle
  - t_nodes_shared.npy    # [m] Chebyshev nodes in [0,1] used for all k
  - teacher.pt            # teacher state_dict
  - tn.pt                 # TN student state_dict
  - manifest.json

Notes
- Zero baseline; for each t in nodes, base = t * x, then mask subset indices to 0.
- k=1: toggles {1,0} on feature i (2 per node)
- k=2: toggles {11,10,01,00} on (i,j) (4 per node)
- k=3: toggles {111,110,101,100,011,010,001,000} on (i,j,k) (8 per node)
- ONE shared Chebyshev grid for all k; by default m = n (minimum that supports k=1..3).
"""

import os, json, csv, math, argparse, itertools, random, warnings, time
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.tntree_model import BinaryTensorTree
from src.feature_mapped_tn import FeatureMappedTN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)

def chebyshev_nodes_01(m: int) -> np.ndarray:
    """Chebyshev–Gauss nodes mapped to [0,1] (use classic cos((2k+1)π/(2m)))."""
    if m <= 0: return np.zeros((0,), dtype=np.float32)
    k = np.arange(m, dtype=np.float64)
    nodes = np.cos((2*k + 1) * np.pi / (2*m))
    t = (nodes + 1.0) * 0.5
    return t.astype(np.float32)

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

# ----- Teacher -----
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
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40, T_mult=2, eta_min=1e-5)
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

# ----- FewEval masked dataset with ONE shared grid -----
def generate_masked_dataset_single_grid(Xtargets: np.ndarray,
                                        t_nodes: np.ndarray,
                                        max_pairs: Optional[int],
                                        max_triplets: Optional[int]):
    """
    Returns:
      masked_X: [M, D]
      meta_rows: CSV rows [row_index,target_idx,order_k,subset,t_index,t_value,toggle]
    """
    N, D = Xtargets.shape
    masked_rows = []; meta_rows = []; row_idx = 0

    pairs = list(itertools.combinations(range(D), 2))
    triplets = list(itertools.combinations(range(D), 3))
    if (max_pairs is not None) and (max_pairs < len(pairs)):
        pairs = random.sample(pairs, max_pairs); pairs.sort()
    if (max_triplets is not None) and (max_triplets < len(triplets)):
        triplets = random.sample(triplets, max_triplets); triplets.sort()

    for r in range(N):
        x = Xtargets[r].astype(np.float32)

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
        for (i,j,k) in triplets:
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

    masked_X = np.stack(masked_rows, axis=0).astype(np.float32) if masked_rows else np.empty((0, D), np.float32)
    return masked_X, meta_rows

# ----- TN student -----
try:
    from src.tntree_model import BinaryTensorTree
except Exception:
    BinaryTensorTree = None

# PATCH START: enhanced TN training

def train_tn_on_masked(masked_X: np.ndarray, y_teacher: np.ndarray,
                       ranks: int, seed: int,
                       max_epochs=100, tol=1e-6, lr=2e-3, amp=True, batch_size=None,
                       fmap_hidden: int = 32, print_every: int = 50,
                       patience: int = 200, target_r2: Optional[float] = None,
                       compile_model: bool = False):
    """Train TN surrogate with several speed / stability optimizations.

    Args:
        masked_X: [N,D]
        y_teacher: [N]
        ranks: TN internal rank
        max_epochs: upper bound on epochs
        tol: MSE early-stop tolerance
        lr: learning rate
        amp: mixed precision on CUDA
        batch_size: optional batch size (auto if None)
        fmap_hidden: hidden size of feature map MLP
        print_every: logging interval
        patience: early stopping patience (no improvement in MSE)
        target_r2: if set, stop early once train R2 >= target
        compile_model: use torch.compile for potential speedup (PyTorch 2.x)
    """
    if BinaryTensorTree is None:
        raise RuntimeError("tntree_model.BinaryTensorTree not found.")
    set_all_seeds(seed)
    dev = DEVICE
    X_t = torch.tensor(masked_X, device=dev)
    y_t = torch.tensor(y_teacher, device=dev)
    d = masked_X.shape[1]

    tn_core = BinaryTensorTree([2]*d, ranks=ranks, out_dim=1,
                               assume_bias_when_matrix=True, seed=seed, device=dev).to(dev)
    model = FeatureMappedTN(tn=tn_core, d_in=d, fmap_hidden=fmap_hidden, fmap_act="relu").to(dev)

    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")  # noqa
            compiled = True
        except Exception:
            compiled = False
    else:
        compiled = False

    n = X_t.shape[0]
    if batch_size is None:
        # heuristic: if very small dataset, do full batch to remove Python overhead
        if n <= 2048:
            batch_size = n
        else:
            batch_size = min(16384, max(512, n // 16))

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and dev.type=="cuda"))
    crit = nn.MSELoss()

    idx = torch.arange(n, device=dev)
    best_state = None
    best_mse = float('inf')
    noimp = 0
    t0 = time.perf_counter()

    # Precompute stats for R2
    y_mean = y_t.mean()
    var_y = torch.var(y_t, unbiased=True).clamp_min(1e-12)

    def eval_full():
        with torch.no_grad():
            pred = model(X_t).squeeze(-1)
            mse_val = crit(pred, y_t).item()
            r2_val = 1.0 - ((pred - y_t).pow(2).mean() / var_y).item()
        return mse_val, r2_val

    for ep in range(1, max_epochs+1):
        model.train()
        # full batch path
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
            total_loss = 0.0
            seen = 0
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

        improved = mse_epoch < best_mse - 1e-12
        if improved:
            best_mse = mse_epoch
            # keep lightweight copy (avoid deep copies every epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            noimp = 0
        else:
            noimp += 1

        need_log = (ep % print_every == 0) or improved or (ep == 1)
        if need_log:
            mse_full, r2_full = eval_full()
            print(f"[TN] ep {ep:4d}  mse_epoch={mse_epoch:.4e}  best_mse={best_mse:.4e}  full_mse={mse_full:.4e}  R2={r2_full:.4f}{' *' if improved else ''}{' [compiled]' if compiled and ep==1 else ''}")
            if target_r2 is not None and r2_full >= target_r2:
                print(f"[TN] target R2 {target_r2} reached at epoch {ep}")
                break

        if mse_epoch < tol:
            print(f"[TN] tol {tol:.1e} reached at epoch {ep}")
            break
        if noimp > patience:
            print(f"[TN] early stop (no improvement {noimp} > patience {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - t0
    final_mse, final_r2 = eval_full()
    print(f"[TN] done in {elapsed:.2f}s  final_mse={final_mse:.4e}  final_R2={final_r2:.4f}")
    return model, dict(final_mse=final_mse, final_r2=final_r2, best_mse=best_mse, epochs=ep, elapsed=elapsed)

# PATCH END

# ----- Main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["concrete","energy_y1","energy_y2","diabetes","california"])
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--kpoints", type=int, default=100)
    ap.add_argument("--outdir", type=str, default="./out_local_student_singlegrid")
    ap.add_argument("--m", type=int, default=None, help="# Chebyshev nodes (shared across k). Default: n (features).")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--max-triplets", type=int, default=None)
    ap.add_argument("--teacher-lr", type=float, default=3e-3)
    ap.add_argument("--tn-lr", type=float, default=2e-2)
    ap.add_argument("--tn-max-epochs", type=int, default=6)
    ap.add_argument("--tn-tol", type=float, default=1e-6)
    ap.add_argument("--no-amp", action="store_true")
    # New optimization flags
    ap.add_argument("--tn-fmap-hidden", type=int, default=32, help="Hidden width for feature map MLP")
    ap.add_argument("--tn-print-every", type=int, default=50, help="Print interval for TN training")
    ap.add_argument("--tn-patience", type=int, default=150, help="Patience for TN early stopping")
    ap.add_argument("--tn-target-r2", type=float, default=None, help="Early stop when train R2 >= this")
    ap.add_argument("--tn-compile", action="store_true", help="Use torch.compile if available")
    args = ap.parse_args()

    set_all_seeds(args.seed)

    # 1) Data + teacher
    ds_name, Xtr, ytr, Xva, yva, Xte, yte, sx, sy = load_dataset_by_name(args.dataset, args.seed)
    teacher = train_teacher(Xtr, ytr, Xva, yva, seed=args.seed, lr=args.teacher_lr)

    # 2) Choose K targets
    rng = np.random.default_rng(args.seed)
    n_te = Xte.shape[0]; K = min(args.kpoints, n_te)
    chosen = rng.choice(n_te, size=K, replace=False)
    Xtargets = Xte[chosen].astype(np.float32)  # [K, D]
    D = Xtargets.shape[1]

    # 3) Shared Chebyshev grid (minimum default = n = D)
    m_shared = int(args.m) if args.m is not None else int(D)
    t_nodes = chebyshev_nodes_01(m_shared)
    print(f"[Grid] using ONE shared Chebyshev grid: m={m_shared} for D={D} (k=1..3)")

    # 4) Build masked dataset with the one grid
    masked_X, meta_rows = generate_masked_dataset_single_grid(
        Xtargets, t_nodes, max_pairs=args.max_pairs, max_triplets=args.max_triplets
    )
    print(f"[FewEval] masked_X shape: {masked_X.shape}")

    # 5) Query teacher
    with torch.no_grad():
        xt = torch.tensor(masked_X, device=DEVICE)
        y_teacher = teacher(xt).detach().cpu().numpy().astype(np.float32)

    # 6) Train TN on masked only (overfit)
    tn, tn_info = train_tn_on_masked(masked_X, y_teacher,
                              ranks=args.rank, seed=args.seed,
                              max_epochs=args.tn_max_epochs, tol=args.tn_tol,
                              lr=args.tn_lr, amp=(not args.no_amp),
                              fmap_hidden=args.tn_fmap_hidden,
                              print_every=args.tn_print_every,
                              patience=args.tn_patience,
                              target_r2=args.tn_target_r2,
                              compile_model=args.tn_compile)

    # 7) Save
    run_root = os.path.join(args.outdir, f"{args.dataset}_seed{args.seed}_K{K}_m{m_shared}")
    ensure_dir(run_root)

    np.save(os.path.join(run_root, "masked_X.npy"), masked_X)
    np.save(os.path.join(run_root, "y_teacher.npy"), y_teacher)
    np.save(os.path.join(run_root, "t_nodes_shared.npy"), t_nodes)

    # meta
    with open(os.path.join(run_root, "meta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index","target_idx","order_k","subset","t_index","t_value","toggle"])
        for row in meta_rows: w.writerow(row)

    # checkpoints
    torch.save(teacher.state_dict(), os.path.join(run_root, "teacher.pt"))
    if BinaryTensorTree is not None:
        torch.save(tn.state_dict(), os.path.join(run_root, "tn.pt"))

    manifest = dict(
        dataset=args.dataset, seed=int(args.seed),
        kpoints=int(K), rank=int(args.rank),
        grid_nodes=int(m_shared), dim=int(D),
        masked_rows=int(masked_X.shape[0]),
        device=str(DEVICE),
        caps=dict(max_pairs=args.max_pairs, max_triplets=args.max_triplets),
        tn_info=tn_info
    )
    with open(os.path.join(run_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Saved to: {run_root}")
    print("Files:\n - masked_X.npy\n - y_teacher.npy\n - t_nodes_shared.npy\n - meta.csv\n - teacher.pt\n - tn.pt\n - manifest.json")

if __name__ == "__main__":
    main()
