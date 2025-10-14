#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Readonly evaluator over precomputed seed_* runs.

- Loads teacher, TN, center idx TXT, and saved SII path bundle (.npy/.json) from seed_*.
- k=1: TN Shapley via saved selector path (TN time = forward+solve only), Exact-1 (teacher),
        KernelSHAP(budgets), ShapIQ(budgets).
- k>=2: Exact-k (teacher & TN, zero baseline, optionally capped subsets), ShapIQ(budgets).

Writes ONLY to --outdir/seed_<seed>/<dataset>/. Never modifies seed_*.
"""

import os, re, math, time, argparse, warnings, json, glob, random
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def set_all_seeds(seed: int):
    torch.manual_seed(seed); 
    (torch.cuda.is_available() and torch.cuda.manual_seed_all(seed))
    np.random.seed(seed); random.seed(seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def fmt_ms(s): return f"{1000.0*float(s):.2f} ms"
def cosine_sim(a,b):
    a=np.asarray(a,float).ravel(); b=np.asarray(b,float).ravel()
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return 1.0 if (na<1e-12 and nb<1e-12) else (0.0 if (na<1e-12 or nb<1e-12) else float(np.dot(a,b)/(na*nb)))
def r2_score_vec(y_true, y_pred):
    y_true=np.asarray(y_true,float).ravel(); y_pred=np.asarray(y_pred,float).ravel()
    ss_res=float(np.sum((y_true-y_pred)**2)); ss_tot=float(np.sum((y_true-np.mean(y_true))**2))
    return 1.0 if (ss_tot<1e-12 and np.allclose(y_true,y_pred)) else float(1.0-ss_res/(ss_tot+1e-12))

# ---------- data loaders (match your splits) ----------
def _split_and_standardize(X,y,seed):
    X=X.astype(np.float64); y=y.astype(np.float64)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=seed)
    Xtr2,Xva,ytr2,yva = train_test_split(Xtr,ytr,test_size=0.2,random_state=seed)
    sx=StandardScaler().fit(Xtr2); sy=StandardScaler().fit(ytr2.reshape(-1,1))
    return (sx.transform(Xtr2).astype(np.float32),
            sy.transform(ytr2.reshape(-1,1)).ravel().astype(np.float32),
            sx.transform(Xva).astype(np.float32),
            sy.transform(yva.reshape(-1,1)).ravel().astype(np.float32),
            sx.transform(Xte).astype(np.float32),
            sy.transform(yte.reshape(-1,1)).ravel().astype(np.float32),
            sx, sy)

def load_dataset_by_name(name, seed):
    name=name.lower()
    if name=="concrete":
        try:
            from ucimlrepo import fetch_ucirepo
            ds=fetch_ucirepo(id=165); X=ds.data.features.to_numpy(float); y=ds.data.targets.to_numpy(float).ravel()
        except Exception as e:
            warnings.warn(f"[concrete] fetch failed ({e}); using synthetic.")
            rng=np.random.default_rng(0); n,d=1200,8
            X=rng.normal(size=(n,d)); y=(15+8*X[:,0]-4*X[:,1]+6*np.tanh(X[:,2])+3*X[:,0]*X[:,1]-
                2*X[:,3]*X[:,4]+5*np.sin(X[:,5])+2*X[:,6]**2-3*X[:,0]*X[:,2]*X[:,6]+rng.normal(0,0.6,n))
        return (name,*_split_and_standardize(X,y,seed))
    if name=="diabetes":
        from sklearn.datasets import load_diabetes
        ds=load_diabetes(); return (name,*_split_and_standardize(ds.data.astype(float), ds.target.astype(float), seed))
    if name=="california":
        from sklearn.datasets import fetch_california_housing
        ds=fetch_california_housing(); return (name,*_split_and_standardize(ds.data.astype(float), ds.target.astype(float), seed))
    if name in ("energy_y1","energy_y2"):
        try:
            from ucimlrepo import fetch_ucirepo
            ds=fetch_ucirepo(name="Energy efficiency"); X=ds.data.features.to_numpy(float)
            Y=ds.data.targets.to_numpy(float); y=Y[:,0 if name=="energy_y1" else 1].ravel()
        except Exception as e:
            warnings.warn(f"[{name}] fetch failed ({e}); synthetic.")
            rng=np.random.default_rng(1 if name=="energy_y1" else 2); n,d=768,8
            X=rng.normal(size=(n,d))
            y=(10+3*X[:,0]-2*X[:,1]+1.5*np.sin(X[:,2])+0.8*X[:,3]*X[:,4]+np.random.default_rng(0).normal(0,0.5,n))
            if name=="energy_y2": y=y+0.7*np.tanh(X[:,5])-1.2*X[:,6]*X[:,1]
        return (name,*_split_and_standardize(X,y,seed))
    raise ValueError(f"Unknown dataset {name}")

# ---------- model loaders ----------
# ----------------------------- teacher MLP ----------------------------- #
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

def _reconstruct_mlp_from_state_dict(sd: Dict[str, torch.Tensor], d_in: Optional[int] = None) -> MLPRegressor:
    """
    Rebuild an MLPRegressor architecture from a plain state_dict.
    - Works whether keys are flat (e.g., '0.weight', '2.weight', ...) or nested (e.g., 'net.0.weight', ...).
    - If only the last layer exists, builds a 1-layer MLP.
    """
    # collect all linear weight tensors in order of appearance
    weight_keys = [k for k in sd.keys() if k.endswith(".weight")]
    if not weight_keys:
        raise RuntimeError("State dict has no '.weight' tensors; cannot reconstruct MLP.")

    # preserve the ordering the model likely had (by numeric segment before '.weight')
    def _extract_order(k: str) -> int:
        # handles 'net.0.weight' or '0.weight'
        parts = k.split(".")
        for p in parts[::-1]:
            if p.isdigit(): return int(p)
        return 10**9  # unknown -> push to end
    weight_keys = sorted(weight_keys, key=_extract_order)

    in_sizes=[]; out_sizes=[]
    for k in weight_keys:
        w = sd[k]
        if not isinstance(w, torch.Tensor) or w.ndim != 2:
            continue
        out_sizes.append(int(w.shape[0]))
        in_sizes.append(int(w.shape[1]))

    if not in_sizes or not out_sizes:
        raise RuntimeError("Could not infer layer shapes from state_dict.")

    # input dim
    if d_in is None:
        d_in = int(in_sizes[0])

    # hidden sizes are all but the final out layer
    # e.g., for weights: [Linear(d_in->h1), Linear(h1->h2), Linear(h2->1)],
    #       out_sizes[:-1] = [h1, h2]
    hidden = [int(o) for o in out_sizes[:-1]]

    model = MLPRegressor(d_in=d_in, hidden=tuple(hidden), pdrop=0.0)
    # try load using raw keys; if they were under a prefix (e.g., 'net.'), still OK with strict=False
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing and all(k.startswith("net.") for k in sd.keys()):
        # if all keys are nested under 'net.', try stripping 'net.' (rare)
        remapped = {k.split("net.",1)[1] if k.startswith("net.") else k: v for k,v in sd.items()}
        model.load_state_dict(remapped, strict=False)
    return model

def load_teacher(teacher_pth: str, d_hint: Optional[int] = None) -> nn.Module:
    obj = torch.load(teacher_pth, map_location="cpu")
    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict):
        # some checkpoints store {'state_dict': ..., ...}, others are raw sd
        sd = obj.get("state_dict", obj)
        # if it’s a Lightning-style sd with 'model.' prefix, drop it
        if any(k.startswith("model.") for k in sd.keys()):
            sd = {k.split("model.",1)[1]: v for k,v in sd.items()}
        model = _reconstruct_mlp_from_state_dict(sd, d_in=d_hint)
        # load again to fill any remaining buffers
        model.load_state_dict(sd, strict=False)
    else:
        raise RuntimeError(f"Unrecognized teacher checkpoint type: {type(obj)}")
    return model.to(DEVICE).eval()


from tntree_model import BinaryTensorTree
def _infer_tn_meta_from_state_dict(sd,d_opt,r_opt):
    if d_opt is not None and r_opt is not None: return int(d_opt),int(r_opt)
    leaf=[k.split(".")[1] for k in sd if k.startswith("leaf_mlps.") and k.endswith(".0.weight")]
    d=len(set(leaf)) if leaf else (int(d_opt) if d_opt is not None else None)
    r=None
    for k,v in sd.items():
        if k.startswith("cores.") and isinstance(v,torch.Tensor):
            shp=tuple(v.shape); 
            if len(shp) in (2,3): r=int(shp[0]); break
    if r_opt is not None: r=int(r_opt)
    if d is None or r is None: raise RuntimeError("Infer TN meta failed")
    return d,r
def _build_blank_tn(d,rank,device): 
    return BinaryTensorTree([2]*d, ranks=rank, out_dim=1, assume_bias_when_matrix=True, seed=None, device=device).to(device)
def load_student_tn(pth,d_hint, r_hint):
    obj=torch.load(pth,map_location="cpu")
    if isinstance(obj,BinaryTensorTree):
        return obj.to(DEVICE).eval(), int(getattr(obj,"ranks", r_hint if r_hint is not None else 1))
    elif isinstance(obj,dict):
        sd=obj.get("state_dict",obj); d,r=_infer_tn_meta_from_state_dict(sd,d_hint,r_hint)
        tn=_build_blank_tn(d,r,DEVICE); tn.load_state_dict(sd,strict=False); tn.eval(); return tn,r
    else: raise RuntimeError(f"Bad TN ckpt {pth}")

# ---------- scan seed_* + center idx ----------
TEACH_RE=re.compile(r"(?P<ds>[a-zA-Z0-9_]+)_teacher_seed(?P<seed>\d+)\.pt$")
TN_RE   =re.compile(r"(?P<ds>[a-zA-Z0-9_]+)_tn_r(?P<rank>\d+)_seed(?P<seed>\d+)\.pt$")
IDX_RE  =re.compile(r"(?P<ds>[a-zA-Z0-9_]+)_testidx_seed(?P<seed>\d+)\.txt$", re.IGNORECASE)
def find_runs(base_dir):
    for seed_dir in sorted(glob.glob(os.path.join(base_dir,"**","seed_*"),recursive=True)):
        if not os.path.isdir(seed_dir): continue
        mseed=re.search(r"seed_(\d+)$",seed_dir); 
        if not mseed: continue
        seed=int(mseed.group(1)); teachers={}; tns={}; ranks={}
        for f in os.listdir(seed_dir):
            if not f.endswith(".pt"): continue
            mT=TEACH_RE.match(f); mS=TN_RE.match(f)
            if mT: teachers[mT.group("ds").lower()] = os.path.join(seed_dir,f)
            if mS: tns[mS.group("ds").lower()] = os.path.join(seed_dir,f); ranks[mS.group("ds").lower()] = int(mS.group("rank"))
        for ds in sorted(set(teachers)&set(tns)):
            yield (seed, seed_dir, ds, teachers[ds], tns[ds], ranks[ds])
def _find_test_idx_txt(seed_dir, ds, seed):
    exact=os.path.join(seed_dir,f"{ds}_testidx_seed{seed}.txt")
    if os.path.isfile(exact):
        with open(exact,"r") as fh: return int(fh.readline().strip())
    for f in os.listdir(seed_dir):
        m=IDX_RE.match(f)
        if m and m.group("seed")==str(seed) and m.group("ds").lower()==ds.lower():
            with open(os.path.join(seed_dir,f),"r") as fh: return int(fh.readline().strip())
    return None
def _find_test_idx_from_csv(seed_dir, ds):
    for csv_path in glob.glob(os.path.join(seed_dir,"*_budget_summary.csv")):
        try:
            df=pd.read_csv(csv_path)
            if "dataset" in df.columns and "test_idx" in df.columns:
                sub=df[df["dataset"].astype(str).str.lower()==ds.lower()]
                if not sub.empty: return int(sub["test_idx"].dropna().astype(int).iloc[0])
        except Exception: pass
    return None
def resolve_center_index(seed_dir, ds, seed):
    idx=_find_test_idx_txt(seed_dir,ds,seed)
    if idx is not None: print(f"[center] Using TXT index for {ds}/seed{seed}: {idx}"); return idx
    idx=_find_test_idx_from_csv(seed_dir,ds)
    if idx is not None: print(f"[center] Using CSV-derived index for {ds}/seed{seed}: {idx}"); return idx
    raise FileNotFoundError(f"Missing center test_idx for {ds}/seed{seed} in {seed_dir}")

# ---------- order-1 TN via saved path (time excludes preprocessing) ----------
def chebyshev_nodes_01(m, device=None):
    k=torch.arange(m,dtype=torch.float32,device=device)
    return 0.5*(torch.cos((2*k+1)*math.pi/(2*m))+1.0)

@torch.no_grad()
def tn_shap_from_loaded_paths_with_timing(model, x, spans_json, device=None):
    if device is None: device=DEVICE
    x_t=torch.as_tensor(x,device=device,dtype=torch.float32).flatten(); d=int(x_t.numel())
    spans=spans_json.get("spans",[]); one=[s for s in spans if int(s.get("order",-1))==1]
    if len(one)<d: raise RuntimeError(f"order-1 spans missing ({len(one)}/{d})")
    phi=torch.zeros(d,device=device,dtype=torch.float32); t_f=0.0; t_s=0.0
    for s in one:
        i=int(s["subset"][0]); m=int(s["m"]); 
        t=chebyshev_nodes_01(m,device=device)
        Xg=t.unsqueeze(1)*x_t.unsqueeze(0); Xh=Xg.clone(); Xh[:,i]=0.0
        X=torch.cat([Xg,Xh],dim=0)
        torch.cuda.synchronize() if device.type=="cuda" else None
        t0=time.perf_counter()
        y=model(X).squeeze(-1)
        torch.cuda.synchronize() if device.type=="cuda" else None
        t_f+=time.perf_counter()-t0
        yg, yh = y[:m], y[m:]
        V=torch.vander(t,N=m,increasing=True); rhs=(yg-yh).unsqueeze(1)
        torch.cuda.synchronize() if device.type=="cuda" else None
        t1=time.perf_counter()
        try: c=torch.linalg.solve(V,rhs).squeeze(1)
        except RuntimeError: c=torch.linalg.lstsq(V,rhs).solution.squeeze(1)
        torch.cuda.synchronize() if device.type=="cuda" else None
        t_s+=time.perf_counter()-t1
        inv_k=torch.zeros_like(c); inv_k[1:]=1.0/torch.arange(1,m,device=device,dtype=torch.float32)
        phi[i]=torch.sum(c*inv_k)
    return phi.detach().cpu().numpy().astype(np.float64), float(t_f), float(t_s)

# ---------- exact k-SII at zero ----------
def iter_k_tuples(d,k):
    from itertools import combinations
    yield from combinations(range(d),k)

def exact_k_sii_zero(predict_np, x, k, T_subsets=None, batch_eval=1<<16):
    t0=time.perf_counter()
    x=np.asarray(x,np.float32).ravel(); d=x.size; 
    M=1<<d; masks=np.arange(M,dtype=np.uint32)
    bits=((masks[:,None]>>np.arange(d,dtype=np.uint32))&1).astype(np.float32)
    fvals=np.empty(M,np.float64); s=0
    while s<M:
        e=min(s+batch_eval,M); X=(bits[s:e]*x[None,:]).astype(np.float32)
        fvals[s:e]=predict_np(X).reshape(-1).astype(np.float64); s=e
    fact=np.ones(d+1,np.float64)
    for i in range(2,d+1): fact[i]=fact[i-1]*i
    # HERE HAS CHANGED!
    denom =  fact[d - k + 1]
    # denom=fact[d-1]*float(k)
    w_by_s=np.array([fact[s]*fact[d-s-k]/denom for s in range(d-k+1)],np.float64)
    k_sizes=np.unpackbits(masks.view(np.uint32).view(np.uint8)).reshape(-1,32)[:,:d].sum(1)
    if T_subsets is None: T_subsets=list(iter_k_tuples(d,k))
    out=[]
    from itertools import combinations
    for T in T_subsets:
        T_bit=0
        for t in T: T_bit|=(1<<t)
        mask_T=(masks & T_bit)==0
        S_all=masks[mask_T]; s_sizes=k_sizes[mask_T]; wS=w_by_s[s_sizes]
        delta=np.zeros_like(wS,np.float64)
        for r in range(k+1):
            sign=(-1.0)**(k-r)
            for U in combinations(T,r):
                U_bit=0
                for u in U: U_bit|=(1<<u)
                idx=S_all|U_bit; delta+=sign*fvals[idx]
        out.append(np.sum(wS*delta,dtype=np.float64))
    return np.asarray(out,np.float64), (time.perf_counter()-t0)

# ---------- ShapIQ for any k ----------
class _NpModel: 
    def __init__(self,fn): self.fn=fn
    def predict(self,X): return self.fn(np.asarray(X,np.float32))
def shapiq_k_order(predict_np, X_bg, x, k, budget):
    import shapiq
    expl=shapiq.TabularExplainer(model=_NpModel(predict_np), data=X_bg, index="k-SII", max_order=k)
    iv=expl.explain(np.asarray(x,float).reshape(1,-1), budget=budget)
    tensor=iv.get_n_order_values(k)  # d x ... x d
    d=tensor.shape[0]
    from itertools import combinations
    return np.asarray([tensor[tuple(T)] for T in combinations(range(d),k)], float)

# ---------- KernelSHAP (k=1 only) ----------
def kernel_shap_order1(predict_np, x, d, nsamples, seed):
    import shap
    rng=np.random.default_rng(seed); np.random.seed(int(rng.integers(0,2**31-1)))
    expl=shap.KernelExplainer(predict_np, np.zeros((1,d)))
    t0=time.perf_counter(); phi=expl.shap_values(np.asarray(x).reshape(1,-1), nsamples=nsamples)[0]
    return np.asarray(phi,float), (time.perf_counter()-t0)

# ---------- persistent selection in EVAL folder (append-only) ----------
def get_or_append_selection(run_out_kdir, d, k, cap, rng):
    ensure_dir(run_out_kdir); sel_path=os.path.join(run_out_kdir,f"selection_order{k}.json")
    from itertools import combinations
    all_T=list(combinations(range(d),k))
    existing=[]
    if os.path.isfile(sel_path):
        try: existing=json.load(open(sel_path,"r")).get("subsets",[])
        except Exception: existing=[]
    existing_set={tuple(map(int,T)) for T in existing}
    if cap is None:
        to_add=[T for T in all_T if T not in existing_set]
        if to_add:
            existing.extend([list(T) for T in to_add])
            json.dump({"order":k,"d":d,"subsets":existing}, open(sel_path,"w"), indent=2)
        return [tuple(T) for T in existing]
    if len(existing_set)>=cap:
        return [tuple(map(int,T)) for T in existing][:cap]
    remaining=[T for T in all_T if T not in existing_set]
    if remaining:
        need=cap-len(existing_set); idx=rng.choice(len(remaining), size=min(need,len(remaining)), replace=False)
        add=[remaining[i] for i in idx]; existing.extend([list(T) for T in add])
        json.dump({"order":k,"d":d,"subsets":existing}, open(sel_path,"w"), indent=2)
    return [tuple(map(int,T)) for T in existing][:cap]

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--orders-k", type=int, nargs="+", default=[1,2,3,4,5,6,7])
    ap.add_argument("--budgets", type=int, nargs="+", default=[2,5,10,15,20,40,50,75,100], help="Kernel budgets for k=1")
    ap.add_argument("--shapiq-budgets", type=int, nargs="+", default=[64,128,256,512,1024, 3000, 5000], help="ShapIQ budgets for ALL k (including k=1)")
    ap.add_argument("--seed-shap", type=int, default=2711)
    ap.add_argument("--max-subsets-per-order", type=int, default=None)
    ap.add_argument("--d", type=int, default=None)     # only if TN state_dict lacks meta
    ap.add_argument("--rank", type=int, default=None)
    args=ap.parse_args()

    set_all_seeds(args.seed_shap)
    runs=list(find_runs(args.base))
    if args.datasets:
        allow={d.lower() for d in args.datasets}
        runs=[r for r in runs if r[2].lower() in allow]
    if not runs: print("No matching seed_* runs."); return

    for (seed, seed_dir, ds_name, teacher_pth, tn_pth, tn_rank) in runs:
        print("\n"+"="*78); print(f"Seed={seed} | Dataset={ds_name} | seed_dir={seed_dir}")
        test_idx=resolve_center_index(seed_dir, ds_name, seed)

        # required SII path bundle for k=1 TN
        pts_npy=os.path.join(seed_dir,f"{ds_name}_sii_points_seed{seed}.npy")
        pts_json=os.path.join(seed_dir,f"{ds_name}_sii_points_seed{seed}.json")
        if not (os.path.isfile(pts_npy) and os.path.isfile(pts_json)):
            print(f"[skip] missing path bundle for {ds_name}/seed{seed}"); continue
        spans_meta=json.load(open(pts_json,"r"))

        name, Xtr, ytr, Xva, yva, Xte, yte, sx, sy = load_dataset_by_name(ds_name, seed)
        if not (0<=test_idx<len(Xte)): print(f"[skip] test_idx {test_idx} out of range"); continue
        x=Xte[int(test_idx)].astype(np.float32); d=Xte.shape[1]

        teacher=load_teacher(teacher_pth, d_hint=d)
        @torch.no_grad()
        def teacher_predict_np(Z):
            zt=torch.tensor(np.asarray(Z,np.float32),device=DEVICE); return teacher(zt).detach().cpu().numpy()
        tn, tn_rank_infer = load_student_tn(tn_pth, d_hint=d, r_hint=tn_rank)
        @torch.no_grad()
        def student_predict_np(Z):
            zt=torch.tensor(np.asarray(Z,np.float32),device=DEVICE); return tn(zt).detach().cpu().numpy()

        run_out=os.path.join(args.outdir,f"seed_{seed}",ds_name); ensure_dir(run_out)
        run_out_kdir=os.path.join(run_out,"kSII"); ensure_dir(run_out_kdir)

        # ---------------- k = 1 ----------------
        print(f"[k=1] TN Shapley via saved path; KernelSHAP + ShapIQ baselines")
        phi_tn, t_forward, t_solve = tn_shap_from_loaded_paths_with_timing(tn, x, spans_meta)
        tn_time_s = t_forward + t_solve

        # Exact-1 (teacher)
        phi1_exact, t_exact1 = exact_k_sii_zero(teacher_predict_np, x, k=1)

        rows=[]
        # KernelSHAP budgets
        for iB,B in enumerate(args.budgets):
            phi_k, dt = kernel_shap_order1(teacher_predict_np, x, d, nsamples=B, seed=args.seed_shap+13*iB)
            rows.append({
                "method":"kernel","budget":B,
                "cos_vs_exact": cosine_sim(phi_k, phi1_exact),
                "r2_vs_exact":  r2_score_vec(phi1_exact, phi_k),
                "cos_vs_tn":    cosine_sim(phi_k, phi_tn),
                "r2_vs_tn":     r2_score_vec(phi_tn, phi_k),
                "time_s": float(dt)
            })
        # ShapIQ budgets (k=1)
        for iB,B in enumerate(args.shapiq_budgets):
            try:
                phi_sh = shapiq_k_order(teacher_predict_np, Xtr, x, k=1, budget=B)
                t_sh = np.nan  # shapiq returns fast; if you need timing, wrap with time.perf_counter
                rows.append({
                    "method":"shapiq","budget":B,
                    "cos_vs_exact": cosine_sim(phi_sh, phi1_exact),
                    "r2_vs_exact":  r2_score_vec(phi1_exact, phi_sh),
                    "cos_vs_tn":    cosine_sim(phi_sh, phi_tn),
                    "r2_vs_tn":     r2_score_vec(phi_tn, phi_sh),
                    "time_s": t_sh
                })
            except Exception as e:
                rows.append({"method":"shapiq","budget":B,"cos_vs_exact":np.nan,"r2_vs_exact":np.nan,"cos_vs_tn":np.nan,"r2_vs_tn":np.nan,"time_s":np.nan,"error":f"{type(e).__name__}: {e}"})

        df1=pd.DataFrame(rows)
        df1.insert(0,"dataset",ds_name); df1.insert(1,"seed",seed); df1.insert(2,"test_idx",int(test_idx)); df1.insert(3,"d",d); df1.insert(4,"order_k",1)
        df1["time_tn_forward_s"]=t_forward; df1["time_tn_solve_s"]=t_solve; df1["time_tn_total_s"]=tn_time_s
        df1["time_exact1_s_teacher"]=t_exact1
        out_csv1=os.path.join(run_out,f"{ds_name}_order1_eval_idx{test_idx}.csv")
        df1.to_csv(out_csv1,index=False)
        np.save(os.path.join(run_out, f"{ds_name}_order1_phi_tn_seed{seed}_idx{test_idx}.npy"), phi_tn)
        np.save(os.path.join(run_out, f"{ds_name}_order1_phi_teacher_exact_idx{test_idx}.npy"), phi1_exact)
        print(f"[k=1] wrote {out_csv1} | TN={fmt_ms(tn_time_s)} (fwd={fmt_ms(t_forward)}+solve={fmt_ms(t_solve)}) | Exact1={fmt_ms(t_exact1)}")

        # ---------------- k >= 2 ----------------
        rng=np.random.default_rng(args.seed_shap)
        for k in args.orders_k:
            if not (1<=k<=d): print(f"[k={k}] skip"); continue
            T_subsets = get_or_append_selection(run_out_kdir, d=d, k=k, cap=args.max_subsets_per_order, rng=rng)
            print(f"[k={k}] exact teacher/TN on {len(T_subsets)} subsets" + ("" if args.max_subsets_per_order is None else f" (cap={args.max_subsets_per_order})"))
            phiK_exact, t_exact = exact_k_sii_zero(teacher_predict_np, x, k=k, T_subsets=T_subsets)
            phiK_tn,    t_tn    = exact_k_sii_zero(student_predict_np,  x, k=k, T_subsets=T_subsets)

            rows=[]
            # ShapIQ budgets (k>=2)
            for B in args.shapiq_budgets:
                try:
                    t0=time.perf_counter()
                    phiK_sh = shapiq_k_order(teacher_predict_np, Xtr, x, k=k, budget=B)
                    t_sh=time.perf_counter()-t0
                    rows.append({"method":"shapiq","budget":B,"cos_vs_exact":cosine_sim(phiK_sh,phiK_exact),"r2_vs_exact":r2_score_vec(phiK_exact,phiK_sh),"time_s":t_sh})
                except Exception as e:
                    rows.append({"method":"shapiq","budget":B,"cos_vs_exact":np.nan,"r2_vs_exact":np.nan,"time_s":np.nan,"error":f"{type(e).__name__}: {e}"})

            dfk=pd.DataFrame(rows if rows else [{}])
            dfk.insert(0,"dataset",ds_name); dfk.insert(1,"seed",seed); dfk.insert(2,"test_idx",int(test_idx)); dfk.insert(3,"d",d); dfk.insert(4,"order_k",k)
            dfk["time_exact_s_teacher"]=float(t_exact); dfk["time_exact_s_tn"]=float(t_tn)
            dfk["cos_tn_vs_exact"]=cosine_sim(phiK_tn,phiK_exact); dfk["r2_tn_vs_exact"]=r2_score_vec(phiK_exact,phiK_tn)
            dfk["n_subsets_evaluated"]=len(T_subsets)

            out_csvk=os.path.join(run_out_kdir,f"{ds_name}_order{k}_idx{test_idx}.csv")
            dfk.to_csv(out_csvk,index=False)
            np.save(os.path.join(run_out_kdir,f"{ds_name}_order{k}_teacher_exact_idx{test_idx}.npy"), phiK_exact)
            np.save(os.path.join(run_out_kdir,f"{ds_name}_order{k}_tn_exact_idx{test_idx}.npy"),      phiK_tn)
            print(f"[k={k}] wrote {out_csvk} | TN vs Exact cos={cosine_sim(phiK_tn,phiK_exact):.3f} R²={r2_score_vec(phiK_exact,phiK_tn):.3f} "
                  f"| t_exact={fmt_ms(t_exact)} t_tn={fmt_ms(t_tn)}")

    print("\nDone.")

if __name__ == "__main__":
    main()
