#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# run_d_sweep_gpu.py
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.tntree_model import BinaryTensorTree
from tn_shap_batched import tn_shap_batched

# ---------------- Config ---------------- #
torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2711
np.random.seed(SEED)
torch.manual_seed(SEED)

RANK_GT = 8
RANK_STUDENT = 5

N_TRAIN = 10000
BATCH = 128
LR = 5e-3
EPOCHS = 200
CLIP = 0.5
EARLY_STOP_R2 = 0.97

N_TEST_POINTS = 10
MAX_EXACT_D = 15


# ---------------- Utils ---------------- #
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if y_true.numel() != y_pred.numel():
        return float("nan")
    ss_res = torch.sum((y_true - y_pred) ** 2)
    y_mean = y_true.mean()
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel(); b = b.ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 and nb < 1e-12: return 1.0
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


# ---------------- Models ---------------- #
def make_ground_truth_tn(d: int, rank: int, seed: int = SEED) -> BinaryTensorTree:
    torch.manual_seed(seed)
    effective_rank = min(rank, d + 2)
    m = BinaryTensorTree([2] * d, ranks=effective_rank, out_dim=1,
                         seed=seed, device=DEVICE, dtype=torch.float64).to(DEVICE)
    with torch.no_grad():
        init_scale = 0.1
        for _, p in m.cores.items():
            if p.ndim == 2 and p.shape[0] == 2:
                p.data.normal_(0, init_scale)
                p[1, :] = torch.clamp(p[1, :], -0.5, 0.5)
            else:
                p.data.normal_(0, init_scale * 0.5)
    return m

class AffineJitterWrappedModel(nn.Module):
    """
    y = alpha * f(x) + gamma * <w, x - 0.5> + beta
    - gamma is zero unless we need to inject variance to hit target std/range.
    - w is a fixed random vector (not learned).
    """
    def __init__(self, base: nn.Module, alpha: float, beta: float,
                 w: Optional[torch.Tensor] = None, gamma: float = 0.0):
        super().__init__()
        self.base = base
        self.register_buffer("_alpha", torch.tensor(alpha, dtype=torch.float64))
        self.register_buffer("_beta",  torch.tensor(beta,  dtype=torch.float64))
        if w is None:
            self.register_buffer("_w", None)
        else:
            self.register_buffer("_w", w.to(dtype=torch.float64))
        self.register_buffer("_gamma", torch.tensor(gamma, dtype=torch.float64))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y = self._alpha * self.base(X) + self._beta
        if self._w is not None and float(self._gamma.item()) != 0.0:
            # center X to [0,1] baseline at 0.5 for balanced linear term
            lin = torch.matmul(X - 0.5, self._w)
            y = y + self._gamma * lin.unsqueeze(-1) if y.ndim == 2 else y + self._gamma * lin
        return y

    def predict(self, X):
        return self.forward(X)

def compute_affine_and_jitter(
    base: nn.Module,
    d: int,
    target_std: Optional[float],
    target_mean: Optional[float],
    minmax: Optional[Tuple[float, float]],
    probe_n: int = 4096,
    rng_seed: int = SEED,
) -> Tuple[float, float, Optional[torch.Tensor], float]:
    """
    Probe base f(X) for X~U([0,1]^d) and compute (alpha, beta, w, gamma).
    If minmax provided -> match range; else match mean/std. If std too small,
    pick a random direction w and choose gamma to hit the desired std exactly.
    """
    with torch.no_grad():
        Xp = torch.rand(probe_n, d, device=DEVICE, dtype=torch.float64)
        yp = base(Xp).flatten()
        mu = float(yp.mean().item())
        sd = float(yp.std().item())
        y_min = float(yp.min().item())
        y_max = float(yp.max().item())

    # 1) If minmax requested, map span to [lo, hi]
    if minmax is not None:
        lo, hi = float(minmax[0]), float(minmax[1])
        span = y_max - y_min
        if span < 1e-12:
            # Degenerate span: inject linear term to manufacture span first.
            g = torch.Generator(device=DEVICE); g.manual_seed(rng_seed)
            w = torch.randn(d, device=DEVICE, dtype=torch.float64, generator=g)
            w = w / (w.norm() + 1e-12)
            with torch.no_grad():
                lin = torch.matmul(Xp - 0.5, w)
                sd_lin = float(lin.std().item())
            # Choose gamma so that new span approximates (hi - lo). Use std proxy -> span ≈ 6*std for uniform-ish.
            desired_std = (hi - lo) / 6.0
            gamma = desired_std / max(sd_lin, 1e-12)
            # Now set alpha=1, beta to align min to lo
            # Re-probe for a better beta estimate:
            with torch.no_grad():
                y_new = base(Xp).flatten() + gamma * torch.matmul(Xp - 0.5, w)
                cur_min = float(y_new.min().item())
                cur_max = float(y_new.max().item())
                alpha = (hi - lo) / max(cur_max - cur_min, 1e-12)
                beta  = lo - alpha * cur_min
            return float(alpha), float(beta), w, float(gamma * alpha)

        alpha = (hi - lo) / max(span, 1e-12)
        beta  = lo - alpha * y_min
        # Check std after mapping; if too small compared to desired span, add a tiny jitter to avoid exactly constant:
        if target_std is not None and target_std > 0:
            # likely not necessary, but keep no-jitter for minmax path
            return float(alpha), float(beta), None, 0.0
        return float(alpha), float(beta), None, 0.0

    # 2) Otherwise, target mean/std
    if target_std is None and target_mean is None:
        return 1.0, 0.0, None, 0.0

    # Base affine first
    alpha = 1.0
    if target_std is not None and target_std > 0 and sd >= 1e-12:
        alpha = target_std / sd
    beta = 0.0 if target_mean is None else (target_mean - alpha * mu)

    # If base sd was tiny, we need jitter to hit target_std
    if (target_std is not None and target_std > 0) and sd < 1e-12:
        g = torch.Generator(device=DEVICE); g.manual_seed(rng_seed)
        w = torch.randn(d, device=DEVICE, dtype=torch.float64, generator=g)
        w = w / (w.norm() + 1e-12)
        with torch.no_grad():
            lin = torch.matmul(Xp - 0.5, w)
            sd_lin = float(lin.std().item())
        # We want std(alpha*f + gamma*lin) ~= target_std; alpha*sd ~ 0, so gamma = target_std / sd_lin
        gamma = target_std / max(sd_lin, 1e-12)
        return float(alpha), float(beta), w, float(gamma)

    # Otherwise, no jitter needed
    return float(alpha), float(beta), None, 0.0


def make_student_tn(d: int, rank: int, seed: int = SEED + 100) -> BinaryTensorTree:
    torch.manual_seed(seed)
    m = BinaryTensorTree([2] * d, ranks=rank, out_dim=1,
                         seed=seed, device=DEVICE, dtype=torch.float64).to(DEVICE)
    with torch.no_grad():
        for p in m.parameters():
            p.data.normal_(0, 0.01)
    return m


# ---------------- Data ---------------- #
def sample_X(n_samples: int, d: int) -> torch.Tensor:
    torch.manual_seed(SEED)
    xs = []
    if d <= 4:
        n_grid = max(2, int(round(n_samples ** (1.0 / d))))
        grid = torch.linspace(0.1, 0.9, n_grid, device=DEVICE, dtype=torch.float64)
        meshes = torch.meshgrid(*([grid] * d), indexing="ij")
        Xg = torch.stack([m.reshape(-1) for m in meshes], dim=1)
        if Xg.size(0) > n_samples // 2:
            idx = torch.randperm(Xg.size(0), device=DEVICE)[: n_samples // 2]
            Xg = Xg[idx]
        xs.append(Xg)
    n_so_far = sum(x.size(0) for x in xs) if xs else 0
    n_rand = max(0, n_samples - n_so_far)
    if n_rand > 0:
        xs.append(torch.rand(n_rand, d, device=DEVICE, dtype=torch.float64))
    X = torch.cat(xs, dim=0) if xs else torch.rand(n_samples, d, device=DEVICE, dtype=torch.float64)
    if X.size(0) > n_samples:
        X = X[torch.randperm(X.size(0), device=DEVICE)[:n_samples]]
    elif X.size(0) < n_samples:
        pad = torch.rand(n_samples - X.size(0), d, device=DEVICE, dtype=torch.float64)
        X = torch.cat([X, pad], dim=0)
    return X[torch.randperm(X.size(0), device=DEVICE)]

def gen_test_points(n_points: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    centers = rng.uniform(-0.5, 0.5, size=(n_points, d))
    noise = rng.normal(0, 0.2, size=(n_points, d))
    return np.clip(centers + noise, -1.0, 1.0)


# ---------------- Train ---------------- #
def train_student(teacher: nn.Module, student: BinaryTensorTree, X: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        y = teacher(X).reshape(-1, 1)
        print(f"Target stats: mean={y.mean().item():+.3f}, std={y.std().item():.3f}, range=[{y.min().item():+.3f}, {y.max().item():+.3f}]")
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=True)
    loss_fn = nn.MSELoss()
    best_r2, patience = -1e9, 0
    for epoch in range(1, EPOCHS + 1):
        student.train()
        run = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = student(xb).reshape(-1, 1)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), CLIP)
            opt.step()
            run += loss.item() * xb.size(0)
        train_mse = run / len(ds)
        sched.step(train_mse)
        if epoch % 10 == 0 or epoch == 1:
            student.eval()
            with torch.no_grad():
                y_pred = student(X).reshape(-1, 1)
                mse = loss_fn(y_pred, y).item()
                r2 = r2_score(y.flatten(), y_pred.flatten())
            print(f"Epoch {epoch:4d} | batch-MSE={train_mse:.3e} | full-MSE={mse:.3e} | R^2={r2:.6f} | lr={opt.param_groups[0]['lr']:.2e}")
            if r2 > best_r2: best_r2, patience = r2, 0
            else: patience += 1
            if r2 > EARLY_STOP_R2 or patience >= 20:
                print(f"Early stop at epoch {epoch} (R^2={r2:.5f})")
                break
        if train_mse < 1e-12:
            print(f"Early stop at epoch {epoch}: MSE ~ 0")
            break
    student.eval()
    with torch.no_grad():
        y_pred = student(X).reshape(-1, 1)
        final_mse = loss_fn(y_pred, y).item()
        final_r2 = r2_score(y.flatten(), y_pred.flatten())
    return {"final_mse": final_mse, "train_r2": final_r2}


# ---------------- Reference Shapley ---------------- #
def exact_shapley_tn(model: nn.Module, x01: np.ndarray) -> np.ndarray:
    x = np.asarray(x01, float)
    d = len(x)
    if d > 20:
        raise ValueError("Exact Shapley infeasible for d > 20")
    def f(x_eval: np.ndarray) -> float:
        xx = torch.tensor(x_eval, device=DEVICE, dtype=torch.float64).unsqueeze(0)
        with torch.no_grad():
            return float(model(xx).squeeze().item())
    phi = np.zeros(d, float)
    all_idx = list(range(d))
    for i in range(d):
        ssum = 0.0
        other = [j for j in all_idx if j != i]
        n_other = len(other)
        for mask in range(1 << n_other):
            subset = [other[j] for j in range(n_other) if (mask >> j) & 1]
            s = len(subset)
            x_with_i = np.zeros(d, float); x_with_i[subset + [i]] = x[subset + [i]]
            x_without_i = np.zeros(d, float); x_without_i[subset] = x[subset]
            marginal = f(x_with_i) - f(x_without_i)
            w = math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d)
            ssum += w * marginal
        phi[i] = ssum
    return phi

def efficient_shapley_tn(model: nn.Module, x01: np.ndarray, n_pts: int = 128) -> np.ndarray:
    x_t = torch.tensor(np.asarray(x01, float), device=DEVICE, dtype=torch.float64).flatten()
    d = x_t.numel()
    phi = torch.zeros(d, device=DEVICE, dtype=torch.float64)
    ts = torch.linspace(0, 1, n_pts, device=DEVICE, dtype=torch.float64)
    with torch.no_grad():
        for i in range(d):
            ssum = 0.0
            for t in ts:
                x_with_i = torch.zeros_like(x_t); x_with_i[: i + 1] = t * x_t[: i + 1]
                x_without_i = torch.zeros_like(x_t); x_without_i[: i] = t * x_t[: i]
                ssum += model(x_with_i.unsqueeze(0)).item() - model(x_without_i.unsqueeze(0)).item()
            phi[i] = ssum / n_pts
    return phi.cpu().numpy()


# ---------------- SHAPIQ (optional) ---------------- #
def shapiq_available() -> bool:
    try:
        import shapiq  # noqa: F401
        return True
    except Exception:
        return False

class _NpModel:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            X_t = torch.tensor(X, device=self.device, dtype=torch.float64)
            if X_t.ndim == 1:
                X_t = X_t.unsqueeze(0)
            y = self.model(X_t).detach().cpu().numpy().ravel()
        return y

def shapiq_sv_single(model: nn.Module, x01: np.ndarray, budget: int, seed: int = 42) -> np.ndarray:
    import shapiq
    np_model = _NpModel(model, DEVICE)
    d = x01.size
    X_bg = np.zeros((1, d), dtype=np.float32)
    explainer = shapiq.TabularExplainer(
        model=np_model,
        data=X_bg,
        approximator="regression",
        index="SV",
        max_order=1,
        random_state=seed,
    )
    iv = explainer.explain(x01.reshape(1, -1).astype(np.float32), budget=budget)
    try:
        vals = np.asarray(iv.get_values(), float).ravel()
    except Exception:
        vals = np.asarray(iv.get_n_order_values(1), float).ravel()
    return vals

def shapiq_batch(model: nn.Module, X01: np.ndarray, budget: int, seed: int = 42):
    import shapiq  # noqa: F401
    N, d = X01.shape
    phis = np.zeros((N, d), dtype=np.float64)
    _sync(); t0 = time.time()
    for i in range(N):
        phis[i] = shapiq_sv_single(model, X01[i], budget=budget, seed=seed + i)
    _sync(); total_ms = (time.time() - t0) * 1000.0
    return phis, total_ms / max(1, N)


# ---------------- Evaluation ---------------- #
def evaluate_dimension(
    d: int,
    compare: str,
    m_points: int,
    feat_chunk: int,
    do_shapiq: bool,
    shapiq_budget: int,
    target_std: float,
    target_mean: float,
    minmax: Optional[Tuple[float, float]],
) -> Dict:
    print("\n" + "=" * 60)
    print(f"Evaluating dimension d={d}")
    print("=" * 60)

    teacher_raw = make_ground_truth_tn(d, RANK_GT, SEED)

    # Compute affine + (optional) jitter to guarantee desired spread
    alpha, beta, w, gamma = compute_affine_and_jitter(
        base=teacher_raw, d=d,
        target_std=target_std, target_mean=target_mean,
        minmax=minmax, probe_n=4096, rng_seed=SEED + d
    )
    teacher = AffineJitterWrappedModel(teacher_raw, alpha=alpha, beta=beta, w=w, gamma=gamma).to(DEVICE)

    # Student
    student = make_student_tn(d, RANK_STUDENT, SEED + 100)

    # Training data
    Xtrain = sample_X(N_TRAIN, d)
    _sync(); t0 = time.time()
    train_stats = train_student(teacher, student, Xtrain)
    _sync(); train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s | R^2={train_stats['train_r2']:.5f}")

    # What to attribute/compare against
    ref_model = teacher if compare == "teacher" else student

    # Test set in [-1,1] -> [0,1]
    X_test = gen_test_points(N_TEST_POINTS, d)
    X01 = (X_test + 1.0) / 2.0

    # TN-SHAP (batched)
    _sync(); t0 = time.time()
    phi_tn_all = tn_shap_batched(ref_model, X01, m_points=m_points,
                                 feature_chunk=feat_chunk, device=DEVICE)
    _sync(); tn_time_total_ms = (time.time() - t0) * 1000.0
    tn_ms_per_point = tn_time_total_ms / len(X01)

    # Reference
    exact_times, phi_ref_all = [], []
    for i in range(N_TEST_POINTS):
        xi = X01[i]
        _sync(); t1 = time.time()
        if d <= MAX_EXACT_D:
            phi_ref = exact_shapley_tn(ref_model, xi)
            exact_kind = "exact"
        else:
            phi_ref = efficient_shapley_tn(ref_model, xi)
            exact_kind = "efficient"
        _sync(); dt = (time.time() - t1) * 1000.0
        exact_times.append(dt)
        phi_ref_all.append(phi_ref)
    phi_ref_all = np.stack(phi_ref_all, axis=0)

    cos_tn = [cosine(phi_tn_all[i], phi_ref_all[i]) for i in range(N_TEST_POINTS)]

    # SHAPIQ (optional)
    shapiq_ms_per_point = None
    shapiq_cos_mean = None
    shapiq_cos_std = None
    shapiq_ok = False
    if do_shapiq and shapiq_available():
        try:
            phi_shapiq, sh_ms = shapiq_batch(ref_model, X01, budget=shapiq_budget, seed=42)
            shapiq_ms_per_point = sh_ms
            cos_sh = [cosine(phi_shapiq[i], phi_ref_all[i]) for i in range(N_TEST_POINTS)]
            shapiq_cos_mean = float(np.mean(cos_sh))
            shapiq_cos_std = float(np.std(cos_sh))
            shapiq_ok = True
            print(f"SHAPIQ: {sh_ms:.2f} ms/pt, cos={shapiq_cos_mean:.6f} ± {shapiq_cos_std:.6f}")
        except Exception as e:
            print(f"SHAPIQ FAILED: {e}")
    else:
        if do_shapiq:
            print("SHAPIQ not available; skipping.")

    print(f"TN-SHAP (batched): {tn_ms_per_point:.2f} ms/pt, cos={np.mean(cos_tn):.6f} ± {np.std(cos_tn):.6f}")
    print(f"Reference ({exact_kind}): {np.mean(exact_times):.2f} ms/pt")
    print(f"(Affine/Jitter) alpha={alpha:.6g}, beta={beta:.6g}, gamma={gamma:.6g}")

    res = {
        "dimension": d,
        "train_r2": train_stats["train_r2"],
        "tn_ms_per_point": float(tn_ms_per_point),
        "tn_cos_mean": float(np.mean(cos_tn)),
        "tn_cos_std": float(np.std(cos_tn)),
        "ref_ms_per_point": float(np.mean(exact_times)),
        "compare": compare,
        "m_points": m_points,
        "feature_chunk": feat_chunk,
        "reference_kind": exact_kind,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "target_std": float(target_std),
        "target_mean": float(target_mean),
        "minmax": "" if minmax is None else list(map(float, minmax)),
    }
    if shapiq_ok:
        res.update({
            "shapiq_ms_per_point": float(shapiq_ms_per_point),
            "shapiq_cos_mean": float(shapiq_cos_mean),
            "shapiq_cos_std": float(shapiq_cos_std),
            "shapiq_budget": int(shapiq_budget),
        })
    return res


# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="10,20,30,40,50")
    ap.add_argument("--compare", choices=["teacher", "student"], default="teacher")
    ap.add_argument("--m_points", type=int, default=6, help="# Chebyshev nodes (4–8 good for multilinear)")
    ap.add_argument("--feature_chunk", type=int, default=64, help="features processed per h_i batch")

    # Target spread controls
    ap.add_argument("--target-std", type=float, default=2.0, help="force outputs to have this std (default 2.0)")
    ap.add_argument("--target-mean", type=float, default=0.0, help="force outputs to have this mean")
    ap.add_argument("--minmax", type=str, default="", help="optional 'lo,hi' range to map outputs into; overrides std/mean")

    # Optional SHAPIQ comparison
    ap.add_argument("--do_shapiq", action="store_true", help="run SHAPIQ comparison if installed")
    ap.add_argument("--shapiq_budget", type=int, default=256, help="query budget per instance for SHAPIQ SV regression")

    ap.add_argument("--outdir", type=str, default="out_gpu")
    args = ap.parse_args()

    dims = [int(s) for s in args.dims.split(",") if s.strip()]

    minmax_tuple = None
    if args.minmax:
        parts = [p.strip() for p in args.minmax.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("--minmax expects 'lo,hi'")
        lo, hi = float(parts[0]), float(parts[1])
        if not (hi > lo):
            raise ValueError("minmax requires hi > lo")
        minmax_tuple = (lo, hi)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    results = []
    for d in dims:
        res = evaluate_dimension(
            d=d,
            compare=args.compare,
            m_points=args.m_points,
            feat_chunk=args.feature_chunk,
            do_shapiq=args.do_shapiq,
            shapiq_budget=args.shapiq_budget,
            target_std=args.target_std,
            target_mean=args.target_mean,
            minmax=minmax_tuple,
        )
        results.append(res)

    out_json = Path(args.outdir) / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # also write a CSV
    hdr = [
        "dimension","train_r2","compare","reference_kind",
        "tn_ms_per_point","tn_cos_mean","tn_cos_std","ref_ms_per_point",
        "m_points","feature_chunk","alpha","beta","gamma",
        "target_std","target_mean","minmax",
        "shapiq_ms_per_point","shapiq_cos_mean","shapiq_cos_std","shapiq_budget"
    ]
    out_csv = Path(args.outdir) / "summary.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(hdr) + "\n")
        for r in results:
            row = [
                r.get("dimension"),
                f"{r.get('train_r2'):.6f}",
                r.get("compare"),
                r.get("reference_kind"),
                f"{r.get('tn_ms_per_point'):.3f}",
                f"{r.get('tn_cos_mean'):.6f}",
                f"{r.get('tn_cos_std'):.6f}",
                f"{r.get('ref_ms_per_point'):.3f}",
                r.get("m_points"),
                r.get("feature_chunk"),
                f"{r.get('alpha'):.6g}",
                f"{r.get('beta'):.6g}",
                f"{r.get('gamma'):.6g}",
                f"{r.get('target_std'):.3f}",
                f"{r.get('target_mean'):.3f}",
                "" if not r.get("minmax") else f"{r['minmax'][0]:.3f}:{r['minmax'][1]:.3f}",
                f"{r.get('shapiq_ms_per_point', float('nan')):.3f}" if 'shapiq_ms_per_point' in r else "",
                f"{r.get('shapiq_cos_mean', float('nan')):.6f}" if 'shapiq_cos_mean' in r else "",
                f"{r.get('shapiq_cos_std', float('nan')):.6f}" if 'shapiq_cos_std' in r else "",
                r.get("shapiq_budget") if 'shapiq_budget' in r else "",
            ]
            f.write(",".join(map(str, row)) + "\n")

    print(f"\nSaved summary to: {out_json}")
    print(f"Saved CSV to:     {out_csv}")
    if args.do_shapiq and not shapiq_available():
        print("Note: SHAPIQ not installed; SHAPIQ columns will be empty.")


if __name__ == "__main__":
    main()
