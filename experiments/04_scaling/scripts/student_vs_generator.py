#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (Shortened header: same as before; only important diffs highlighted below)

import os, json, time, math, argparse
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def chebyshev_nodes_01(m: int) -> np.ndarray:
    k = np.arange(m, dtype=np.float64)
    nodes = np.cos((2*k + 1) * np.pi / (2*m))
    return ((nodes + 1.0) * 0.5).astype(np.float32)

# ----- GT loader & exact interactions (same as previous) -----

class SparseMultilinearGT:
    def __init__(self, d: int, coeffs: Dict[Tuple[int, ...], float]):
        self.d = int(d)
        self.coeffs = {tuple(sorted(k)): float(v) for k, v in coeffs.items()}
        self.o1 = {k:v for k,v in self.coeffs.items() if len(k)==1}
        self.o2 = {k:v for k,v in self.coeffs.items() if len(k)==2}
        self.o3 = {k:v for k,v in self.coeffs.items() if len(k)==3}
    def f(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float); one = X.ndim==1
        if one: X=X[None,:]
        out = np.zeros((X.shape[0],), dtype=np.float64)
        for S,w in self.coeffs.items():
            if len(S)==1: out += w * X[:,S[0]]
            elif len(S)==2: out += w * X[:,S[0]]*X[:,S[1]]
            else: out += w * X[:,S[0]]*X[:,S[1]]*X[:,S[2]]
        return out.astype(np.float32) if not one else np.float32(out[0])
    def shapley_k1(self, x): 
        x = np.asarray(x,float).ravel(); d=self.d
        phi = np.zeros(d, float)
        for (i,),w in self.o1.items(): phi[i]+=w
        for (i,j),w in self.o2.items():
            phi[i]+=(w*x[j])/2.0; phi[j]+=(w*x[i])/2.0
        for (i,j,k),w in self.o3.items():
            phi[i]+=(w*x[j]*x[k])/3.0; phi[j]+=(w*x[i]*x[k])/3.0; phi[k]+=(w*x[i]*x[j])/3.0
        return phi.astype(np.float32)
    def sii_k2(self, x):
        x=np.asarray(x,float).ravel(); d=self.d; out=[]
        for i in range(d):
            for j in range(i+1,d):
                out.append(self.o2.get((i,j),0.0)*x[i]*x[j])
        return np.asarray(out,np.float32)
    def sii_k3(self, x):
        x=np.asarray(x,float).ravel(); d=self.d; out=[]
        for i in range(d):
            for j in range(i+1,d):
                for k in range(j+1,d):
                    out.append(self.o3.get((i,j,k),0.0)*x[i]*x[j]*x[k])
        return np.asarray(out,np.float32)

def load_gt_from_root(gt_root: str):
    mani = coeffs_csv = x_path = None
    for fn in os.listdir(gt_root):
        if fn.endswith("_manifest.json"): mani = os.path.join(gt_root, fn)
        if fn.endswith("_coeffs.csv"):    coeffs_csv = os.path.join(gt_root, fn)
        if fn.endswith("_x.npy"):         x_path = os.path.join(gt_root, fn)
    if mani is None or coeffs_csv is None or x_path is None:
        raise FileNotFoundError("Missing *_manifest.json, *_coeffs.csv, or *_x.npy in --gt-root")
    man = json.load(open(mani, "r"))
    d = int(man["d"])
    df = pd.read_csv(coeffs_csv)
    coeffs: Dict[Tuple[int,...], float] = {}
    for _, r in df.iterrows():
        S = tuple(int(s) for s in str(r["indices"]).split(";"))
        coeffs[S] = float(r["weight"])
    gt = SparseMultilinearGT(d, coeffs)
    x = np.load(x_path).astype(np.float32).ravel()
    assert x.size == d
    return gt, x, man, coeffs_csv, x_path

# ----- TN + wrapper -----

try:
    from tntree_model import BinaryTensorTree
except Exception:
    BinaryTensorTree = None

try:
    from feature_mapped_tn import FeatureMappedTN
except Exception:
    FeatureMappedTN = None

def train_tn_fast_fullbatch(
    X: np.ndarray, y: np.ndarray, d: int, rank: int, seed: int,
    lr_adam=2e-3, adam_epochs=200, lbfgs_steps=20,
    tol_abs=1e-8, tol_rel=1e-8, target_r2=0.999,
    fmap_hidden=32, use_amp=True, use_compile=False
):
    if BinaryTensorTree is None:  raise RuntimeError("tntree_model.BinaryTensorTree not found.")
    if FeatureMappedTN is None:   raise RuntimeError("feature_mapped_tn.FeatureMappedTN not found.")
    set_all_seeds(seed)
    dev = DEVICE
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev).flatten()

    tn_core = BinaryTensorTree([2]*d, ranks=rank, out_dim=1,
                               assume_bias_when_matrix=True, seed=seed, device=dev).to(dev)
    model = FeatureMappedTN(tn=tn_core, d_in=d, fmap_hidden=fmap_hidden, fmap_act="relu").to(dev)
    if use_compile and hasattr(torch, "compile"):
        try: model = torch.compile(model, mode="reduce-overhead")
        except Exception: pass
    model.train()

    var_y = torch.var(y_t, unbiased=True).clamp_min(1e-12)
    mse_threshold = float(min(tol_abs, tol_rel * var_y.item()))

    base_kwargs = dict(lr=lr_adam, weight_decay=0.0, eps=1e-8)
    if dev.type == "cuda":
        try: opt = torch.optim.AdamW(model.parameters(), fused=True, **base_kwargs)
        except TypeError: opt = torch.optim.AdamW(model.parameters(), **base_kwargs)
    else:
        opt = torch.optim.AdamW(model.parameters(), **base_kwargs)
    crit = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and dev.type=="cuda"))

    t0 = time.perf_counter()
    best_state, best_mse = None, float("inf")
    for ep in range(1, adam_epochs+1):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(use_amp and dev.type=="cuda")):
            pred = model(X_t).squeeze(-1)
            loss = crit(pred, y_t)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        mse = float(loss.detach().item())
        if mse < best_mse:
            best_mse = mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            with torch.no_grad():
                pred_full = model(X_t).squeeze(-1)
                r2 = float(1.0 - ((pred_full - y_t).pow(2).mean() / var_y).item())
                if (r2 >= target_r2) or (mse <= mse_threshold):
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad(set_to_none=True)
        pred = model(X_t).squeeze(-1)
        loss = crit(pred, y_t)
        loss.backward()
        return loss
    for _ in range(lbfgs_steps):
        loss = lbfgs.step(closure)
        if float(loss.detach().item()) <= mse_threshold:
            break

    elapsed = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        pred = model(X_t).squeeze(-1)
        mse = float(crit(pred, y_t).item())
        r2  = float(1.0 - ((pred - y_t).pow(2).mean() / var_y).item())
    return model, {"train_time_s": elapsed, "final_mse": mse, "final_r2": r2, "adam_epochs": ep}

@torch.no_grad()
def tn_selector_any_k_sharedgrid(model: nn.Module, x: np.ndarray, t_nodes: np.ndarray, k: int):
    from itertools import combinations
    x = np.asarray(x, np.float32).ravel()
    d = x.size
    t = torch.tensor(np.asarray(t_nodes, np.float32), device=DEVICE)
    m = int(t.numel())
    V = torch.vander(t, N=m, increasing=True)
    t0 = time.perf_counter()
    Vinv = torch.linalg.inv(V)
    t_solve = time.perf_counter() - t0
    x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    subs_by_size = {s: list(combinations(range(d), s)) for s in range(0, k+1)}
    coeffs: Dict[Tuple[int, ...], np.ndarray] = {}
    Xg = t.unsqueeze(1) * x_t.unsqueeze(0)
    t1 = time.perf_counter()
    yg = model(Xg).squeeze(-1)
    t_eval = time.perf_counter() - t1
    coeffs[()] = ((Vinv @ yg.unsqueeze(1)).squeeze(1)).detach().cpu().numpy().astype(np.float64)
    for s in range(1, k+1):
        subs = subs_by_size[s]
        if not subs: continue
        Xh = Xg.repeat(len(subs), 1)
        mR = m
        for r, S in enumerate(subs):
            Xh[r*mR:(r+1)*mR, list(S)] = 0.0
        t2 = time.perf_counter()
        yh = model(Xh).squeeze(-1).view(len(subs), m)
        t_eval += time.perf_counter() - t2
        t3 = time.perf_counter()
        cS = (yh @ Vinv.T)
        t_solve += time.perf_counter() - t3
        for S, c in zip(subs, cS):
            coeffs[tuple(S)] = c.detach().cpu().numpy().astype(np.float64)
    all_T = list(combinations(range(d), k))
    phi = np.zeros(len(all_T), dtype=np.float64)
    weights = np.zeros(m, dtype=np.float64)
    for r in range(k, m): weights[r] = 1.0 / math.comb(r, k)
    from itertools import combinations as comb
    for idx_T, T in enumerate(all_T):
        cT = np.zeros(m, dtype=np.float64)
        for s in range(0, k+1):
            for S in comb(T, s):
                cT += ((-1.0) ** len(S)) * coeffs[S]
        phi[idx_T] = float(np.dot(cT[k:], weights[k:]))
    return phi, {"t_eval_s": float(t_eval), "t_solve_s": float(t_solve), "t_total_s": float(t_eval + t_solve)}

def build_masked_dataset_at_x(x: np.ndarray, t_nodes: np.ndarray):
    from itertools import combinations
    x = np.asarray(x, np.float32).ravel()
    d = x.size
    masked = []
    # k=1
    for i in range(d):
        for t in t_nodes:
            base = (t * x).astype(np.float32)
            x1 = base
            x0 = base.copy(); x0[i] = 0.0
            masked += [x1, x0]
    # k=2
    for (i,j) in combinations(range(d), 2):
        for t in t_nodes:
            base = (t * x).astype(np.float32)
            x11 = base
            x10 = base.copy(); x10[j] = 0.0
            x01 = base.copy(); x01[i] = 0.0
            x00 = base.copy(); x00[i] = 0.0; x00[j] = 0.0
            masked += [x11, x10, x01, x00]
    # k=3
    for (i,j,k) in combinations(range(d), 3):
        for t in t_nodes:
            base = (t * x).astype(np.float32)
            for a in (1,0):
                for b in (1,0):
                    for c in (1,0):
                        z = base.copy()
                        if a==0: z[i]=0.0
                        if b==0: z[j]=0.0
                        if c==0: z[k]=0.0
                        masked.append(z)
    return np.stack(masked, axis=0).astype(np.float32)

# ---- SHAPIQ wrapper (unchanged) ----
def run_shapiq(gt: "SparseMultilinearGT", x: np.ndarray, k: int, budget: int, approximator: str, seed: int):
    try:
        import shapiq
    except Exception as e:
        raise e
    rng = np.random.default_rng(seed)
    X_bg = np.zeros((1, gt.d), dtype=np.float32)
    class _Wrap:
        def __init__(self, fn): self.fn=fn
        def predict(self, X): return gt.f(np.asarray(X, np.float32))
    expl = shapiq.TabularExplainer(
        model=_Wrap(gt.f), data=X_bg, approximator=approximator, index="SII", max_order=k,
        random_state=int(rng.integers(0, 2**31-1))
    )
    t0=time.perf_counter()
    iv = expl.explain(np.asarray(x,float).reshape(1,-1), budget=budget)
    dt=time.perf_counter()-t0
    if k==1:
        try: vals = np.asarray(iv.get_values(), float).ravel()
        except Exception: vals = np.asarray(iv.get_n_order_values(1), float).ravel()
        return vals, dt, f"SHAPIQ-{approximator}-k1"
    tens = iv.get_n_order_values(k)
    if k==2:
        vals = np.asarray([tens[i,j] for i in range(gt.d) for j in range(i+1,gt.d)], float)
        return vals, dt, f"SHAPIQ-{approximator}-k2"
    vals = np.asarray([tens[i,j,k3] for i in range(gt.d) for j in range(i+1,gt.d) for k3 in range(j+1,gt.d)], float)
    return vals, dt, f"SHAPIQ-{approximator}-k3"

def cosine_np(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 and nb < 1e-12: return 1.0
    if na < 1e-12 or nb < 1e-12:  return 0.0
    return float(np.dot(a, b) / (na * nb))
def r2_score_vec(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel(); y_pred = np.asarray(y_pred, float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return float(1.0 - ss_res / (ss_tot + 1e-12))
def mse_vec(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel(); y_pred = np.asarray(y_pred, float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--orders", type=int, nargs="+", default=[1,2,3])
    ap.add_argument("--strategy", choices=["random","masked"], required=True)
    ap.add_argument("--n-random", type=int, default=5000)
    ap.add_argument("--x-dist", choices=["normal","uniform","binary"], default="normal")
    ap.add_argument("--adam-epochs", type=int, default=40)
    ap.add_argument("--lbfgs-steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--target-r2", type=float, default=0.9995)
    ap.add_argument("--tol-abs", type=float, default=1e-8)
    ap.add_argument("--tol-rel", type=float, default=1e-8)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fmap-hidden", type=int, default=32)
    ap.add_argument("--with-shapiq", action="store_true", default=False)
    ap.add_argument("--shapiq-approximator", choices=["regression","permutation","montecarlo"], default="regression")
    ap.add_argument("--shapiq-budget", type=int, default=2000)
    args = ap.parse_args()

    ensure_dir(args.outdir); set_all_seeds(args.seed)

    gt, x, man, coeffs_csv, x_path = load_gt_from_root(args.gt_root)
    d = gt.d
    t_nodes = chebyshev_nodes_01(d)

    if args.strategy == "random":
        if args.x_dist == "normal":
            Xtr = np.random.normal(0.0, 1.0, size=(args.n_random, d)).astype(np.float32)
        elif args.x_dist == "uniform":
            Xtr = np.random.uniform(-1.0, 1.0, size=(args.n_random, d)).astype(np.float32)
        else:
            Xtr = (np.random.rand(args.n_random, d) < 0.5).astype(np.float32)
        ytr = gt.f(Xtr)
    else:
        Xtr = build_masked_dataset_at_x(x, t_nodes)
        ytr = gt.f(Xtr)

    tn, tinfo = train_tn_fast_fullbatch(
        Xtr, ytr, d=d, rank=args.rank, seed=args.seed,
        lr_adam=args.lr, adam_epochs=args.adam_epochs, lbfgs_steps=args.lbfgs_steps,
        tol_abs=args.tol_abs, tol_rel=args.tol_rel, target_r2=args.target_r2,
        fmap_hidden=args.fmap_hidden, use_amp=(not args.no_amp), use_compile=args.compile
    )
    ckpt = os.path.join(args.outdir, "student_ckpt.pt")
    torch.save(tn.state_dict(), ckpt)

    rows = []
    def add_row(k, method, runtime, est, gt_vals, strategy):
        rows.append({
            "seed": args.seed, "d": d, "k": int(k),
            "method": method, "baseline": "TNShap", "budget": "",
            "runtime_s": float(runtime),
            "cosine_sim": cosine_np(gt_vals, est),
            "r2": r2_score_vec(gt_vals, est),
            "mse": mse_vec(gt_vals, est),
            "student_train_time_s": float(tinfo["train_time_s"]),
            "student_train_r2": float(tinfo["final_r2"]),
            "student_train_mse": float(tinfo["final_mse"]),
            "strategy": strategy,
        })

    gt_k = {}
    if 1 in args.orders: gt_k[1] = gt.shapley_k1(x)
    if 2 in args.orders: gt_k[2] = gt.sii_k2(x)
    if 3 in args.orders: gt_k[3] = gt.sii_k3(x)

    for k in args.orders:
        phi_tn, tsel = tn_selector_any_k_sharedgrid(tn, x, t_nodes, k=k)
        np.save(os.path.join(args.outdir, f"student_phi_k{k}.npy"), phi_tn)
        np.save(os.path.join(args.outdir, f"gt_phi_k{k}.npy"), gt_k[k])
        add_row(k, "tn_selector", tsel["t_total_s"], phi_tn, gt_k[k], args.strategy)

    if args.with_shapiq:
        try:
            for k in args.orders:
                vals, dt, label = run_shapiq(gt, x, k=k, budget=args.shapiq_budget,
                                             approximator=args.shapiq_approximator, seed=args.seed+100*k)
                rows.append({
                    "seed": args.seed, "d": d, "k": int(k),
                    "method": label, "baseline": "SHAPIQ", "budget": args.shapiq_budget,
                    "runtime_s": float(dt),
                    "cosine_sim": cosine_np(gt_k[k], vals),
                    "r2": r2_score_vec(gt_k[k], vals),
                    "mse": mse_vec(gt_k[k], vals),
                    "student_train_time_s": float(tinfo["train_time_s"]),
                    "student_train_r2": float(tinfo["final_r2"]),
                    "student_train_mse": float(tinfo["final_mse"]),
                    "strategy": args.strategy,
                })
        except Exception as e:
            rows.append({
                "seed": args.seed, "d": d, "k": -1,
                "method": "SHAPIQ-error", "baseline": "SHAPIQ", "budget": args.shapiq_budget,
                "runtime_s": float("nan"), "cosine_sim": float("nan"),
                "r2": float("nan"), "mse": float("nan"),
                "student_train_time_s": float(tinfo["train_time_s"]),
                "student_train_r2": float(tinfo["final_r2"]),
                "student_train_mse": float(tinfo["final_mse"]),
                "strategy": args.strategy, "error": f"{type(e).__name__}: {e}",
            })

    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "student_results.csv"), index=False)
    with open(os.path.join(args.outdir, "student_manifest.json"), "w") as f:
        json.dump({
            "gt_root": os.path.abspath(args.gt_root), "d": d, "seed": args.seed, "rank": args.rank,
            "orders": args.orders, "strategy": args.strategy, "train_info": tinfo,
            "ckpt": os.path.abspath(ckpt), "target_r2": args.target_r2,
        }, f, indent=2)
    print("[OK] student done:", os.path.join(args.outdir, "student_results.csv"))

if __name__ == "__main__":
    main()
