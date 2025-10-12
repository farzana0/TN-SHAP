#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# -*- coding: utf-8 -*-

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List, Sequence
import numpy as np
import matplotlib.pyplot as plt

# -------- ensure parent dir helper --------
def _ensure_parent_dir(pth: Path | str) -> Path:
    p = Path(pth)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

# Use your minimal plotting helpers; fall back to inline if not found.
try:
    from tnshap_plot_template import plot_curves, plot_heatmap
except Exception:
    def _format_axes(ax, xlabel: str, ylabel: str, title=None):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if title: ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.5)

    def plot_curves(x, y_series: Dict[str, Sequence[float]], ylabel, title, outfile,
                    xlabel="Step / Order index", legend=True, figsize=(5.0,3.2), linewidth=1.7, dpi=300):
        fig, ax = plt.subplots(figsize=figsize)
        for name, y in y_series.items():
            ax.plot(x, y, label=name, linewidth=linewidth)
        _format_axes(ax, xlabel, ylabel, title)
        if legend: ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        outfile = _ensure_parent_dir(outfile)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)
        return str(outfile)

    def plot_heatmap(Z: np.ndarray, xlabel: str, ylabel: str, title: str, outfile: str,
                     extent=None, figsize=(5.3,3.6), dpi=300, add_colorbar=True):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(Z, aspect="auto", origin="lower", extent=extent)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        if add_colorbar: fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        fig.tight_layout()
        outfile = _ensure_parent_dir(outfile)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)
        return str(outfile)

# ---------------- utils ----------------

def parse_tuple_key(s: str) -> Tuple[int, ...]:
    t = ast.literal_eval(s)
    return (t,) if isinstance(t, int) else tuple(int(x) for x in t)

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def safe_r2(y: np.ndarray, yhat: np.ndarray, atol=1e-10, rtol=1e-8) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    mu = y.mean(); sst = np.sum((y - mu)**2)
    if sst < max(atol, rtol*(abs(mu)+atol)):
        return 1.0 if np.allclose(y, yhat, atol=atol, rtol=rtol) else 0.0
    ssr = np.sum((y - yhat)**2)
    return float(1 - ssr/sst)

def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat)**2)))

# -------------- core experiments (three folders) --------------

def load_core_exp(expdir: Path, base: str, max_order: int) -> Dict:
    out = {"name": base, "per_order": {}}
    # optional summary.json
    summ = expdir / f"{base}_summary.json"
    if summ.exists():
        S = load_json(summ)
        out["train_r2"] = S.get("train_r2", None)
    for k in range(1, max_order+1):
        gtf = expdir / f"{base}_sii_gt_order{k}.json"
        stf = expdir / f"{base}_sii_student_order{k}.json"
        if not (gtf.exists() and stf.exists()):
            continue
        GTd = {parse_tuple_key(kc): float(vc) for kc, vc in load_json(gtf).items()}
        STd = {parse_tuple_key(kc): float(vc) for kc, vc in load_json(stf).items()}
        keys = sorted(GTd.keys())
        y = np.array([GTd[S] for S in keys], float)
        yhat = np.array([STd.get(S, 0.0) for S in keys], float)
        out["per_order"][k] = {
            "keys": keys, "gt": y, "st": yhat,
            "safe_r2": safe_r2(y, yhat),
            "rmse": rmse(y, yhat),
            "gt_maxabs": float(np.max(np.abs(y))) if len(y) else 0.0,
        }
    return out

# -------------- ablation: rank sweep --------------

def load_rank_sweep(csv_path: Path) -> Dict:
    rows = []
    with open(csv_path, "r") as f:
        R = csv.DictReader(f)
        for r in R:
            rows.append({
                "rank_student": int(r["rank_student"]),
                "train_r2": float(r["train_r2"]),
                "r2_order_1": float(r["r2_order_1"]),
                "r2_order_2": float(r["r2_order_2"]),
                "r2_order_3": float(r["r2_order_3"]),
            })
    rows = sorted(rows, key=lambda z: z["rank_student"])
    return {"rows": rows}

# -------------- ablation: LR/epoch sweep --------------

def load_lr_sweep(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, "r") as f:
        R = csv.DictReader(f)
        for r in R:
            rows.append({
                "epoch": int(r["epoch"]),
                "lr": float(r["lr"]),
                "mse": float(r["mse"]),
                "train_r2": float(r["train_r2"]),
                "r2_order_1": float(r["r2_order_1"]),
                "r2_order_2": float(r["r2_order_2"]),
                "r2_order_3": float(r["r2_order_3"]),
                "ckpt": r.get("ckpt", ""),
            })
    return rows

# -------------- latex helpers --------------

def latex_table_rank_sweep(rows: List[Dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{rcccc}")
    lines.append(r"\toprule")
    lines.append(r"Rank & Train $R^2$ & SII $R^2$ (o1) & SII $R^2$ (o2) & SII $R^2$ (o3) \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(f"{r['rank_student']} & {r['train_r2']:.3f} & {r['r2_order_1']:.3f} & {r['r2_order_2']:.3f} & {r['r2_order_3']:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Rank ablation (GT TN rank=14). SII $R^2$ at anchor $x_0$.}")
    lines.append(r"\label{tab:rank-ablation}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def latex_table_lr_final(rows: List[Dict]) -> str:
    # one row per LR using the last epoch for that LR
    by_lr = {}
    for r in rows:
        lr = r["lr"]
        if lr not in by_lr or r["epoch"] > by_lr[lr]["epoch"]:
            by_lr[lr] = r
    L = sorted(by_lr.items(), key=lambda z: z[0])
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{rcccccc}")
    lines.append(r"\toprule")
    lines.append(r"LR & Epoch & MSE & Train $R^2$ & SII $R^2$ (o1) & SII $R^2$ (o2) & SII $R^2$ (o3) \\")
    lines.append(r"\midrule")
    for lr, r in L:
        lines.append(f"{lr:.0e} & {r['epoch']} & {r['mse']:.3e} & {r['train_r2']:.3f} & {r['r2_order_1']:.3f} & {r['r2_order_2']:.3f} & {r['r2_order_3']:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Loss/R$^2$ vs SII $R^2$ (final epoch per learning rate) for generic multilinear.}")
    lines.append(r"\label{tab:lr-vs-sii}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def latex_table_core(core_exps: List[Dict]) -> str:
    # orders 1..3 only
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Experiment & Train $R^2$ & SII $R^2$ (o1) & SII $R^2$ (o2) & SII $R^2$ (o3) \\")
    lines.append(r"\midrule")
    for e in core_exps:
        r2o1 = e["per_order"].get(1, {}).get("safe_r2", np.nan)
        r2o2 = e["per_order"].get(2, {}).get("safe_r2", np.nan)
        r2o3 = e["per_order"].get(3, {}).get("safe_r2", np.nan)
        trr2 = e.get("train_r2", np.nan)
        lines.append(f"{e['name']} & {trr2:.3f} & {r2o1:.3f} & {r2o2:.3f} & {r2o3:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Teacher types vs student: SII $R^2$ at $x_0$ (orders 1–3).}")
    lines.append(r"\label{tab:core-teachers}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

# -------------- plotting helpers --------------
def plot_lr_sweep_combined(rows: List[Dict], outdir: Path, lr_choice: float = None):
    """
    One figure: Train R^2 and SII R^2 (orders 1..3) vs epoch on the same axes.
    If lr_choice is None, picks the middle LR from the sweep.
    Saves: figs/generic_sweep/train_and_sii_r2_vs_epoch_lr<lr>.png
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # group rows by learning rate
    by_lr = {}
    for r in rows:
        by_lr.setdefault(r["lr"], []).append(r)

    if not by_lr:
        return

    # choose LR (middle if not provided)
    lrs_sorted = sorted(by_lr.keys())
    if lr_choice is None:
        lr_choice = lrs_sorted[len(lrs_sorted) // 2]

    seq = sorted(by_lr[lr_choice], key=lambda z: z["epoch"])
    epochs = [z["epoch"] for z in seq]

    y_series = {
        "Train $R^2$":      [z["train_r2"] for z in seq],
        "SII $R^2$ (o1)":   [z["r2_order_1"] for z in seq],
        "SII $R^2$ (o2)":   [z["r2_order_2"] for z in seq],
        "SII $R^2$ (o3)":   [z["r2_order_3"] for z in seq],
    }

    tag = f"{lr_choice:.0e}"
    plot_curves(
        x=epochs,
        y_series=y_series,
        ylabel="$R^2$",
        title=f"Train and SII $R^2$ vs epoch (LR={lr_choice:.0e})",
        outfile=str(outdir / f"train_and_sii_r2_vs_epoch_lr{tag}.png"),
        xlabel="Epoch",
        legend=True,
    )

def plot_per_order_lines(core_exps: List[Dict], outdir: Path, orders: int):
    outdir.mkdir(parents=True, exist_ok=True)
    for e in core_exps:
        K = sorted([k for k in e["per_order"].keys() if 1 <= k <= orders])
        if not K: 
            continue
        r2s = [e["per_order"][k]["safe_r2"] for k in K]
        plot_curves(
            x=K,
            y_series={e["name"]: r2s},
            ylabel="safe-$R^2$ (student vs GT)",
            title=f"{e['name']} (per-order SII)",
            outfile=str(outdir / f"{e['name']}_per_order_r2.png"),
            xlabel="Interaction order k",
            legend=False,
        )

def plot_rank_heatmap_and_curves(sweep_rows: List[Dict], outdir: Path, orders=3):
    outdir.mkdir(parents=True, exist_ok=True)
    if not sweep_rows:
        return
    ranks = sorted({r["rank_student"] for r in sweep_rows})
    Z = np.full((orders, len(ranks)), np.nan, dtype=float)
    for i, k in enumerate(range(1, orders+1)):
        for j, rnk in enumerate(ranks):
            r = [row for row in sweep_rows if row["rank_student"] == rnk][0]
            Z[i, j] = r[f"r2_order_{k}"]
    plot_heatmap(
        Z, xlabel="Student rank", ylabel="Order k",
        title="safe-$R^2$ (GT TN rank=14)",
        outfile=str(outdir / "heatmap_gt14.png"),
        extent=(min(ranks)-0.5, max(ranks)+0.5, 0.5, orders+0.5)
    )
    # Rank curves (avg over orders ≤Kcap)
    series = {}
    for Kcap in (1, 2, 3):
        y = []
        for rnk in ranks:
            row = [row for row in sweep_rows if row["rank_student"] == rnk][0]
            vals = [row[f"r2_order_{k}"] for k in range(1, Kcap+1)]
            y.append(float(np.mean(vals)))
        series[f"avg over orders ≤{Kcap}"] = y
    plot_curves(
        x=ranks, y_series=series,
        ylabel="mean safe-$R^2$",
        title="Rank ablation (GT TN rank=14)",
        outfile=str(outdir / "rank_curves_gt14.png"),
        xlabel="Student rank", legend=True
    )

def plot_lr_sweep(rows: List[Dict], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Train R² vs epoch per LR
    by_lr = {}
    for r in rows:
        by_lr.setdefault(r["lr"], []).append(r)
    series_tr = {}
    for lr, lst in sorted(by_lr.items(), key=lambda z: z[0]):
        lst = sorted(lst, key=lambda z: z["epoch"])
        series_tr[f"LR {lr:.0e}"] = [z["train_r2"] for z in lst]
    epochs = sorted({r["epoch"] for r in rows})
    plot_curves(
        x=epochs, y_series=series_tr,
        ylabel="Train $R^2$", title="Generic LR sweep: train $R^2$ vs epoch",
        outfile=str(outdir / "trainr2_vs_epoch.png"),
        xlabel="Epoch", legend=True
    )
    # SII R² vs epoch for a representative LR (middle LR)
    lrs_sorted = sorted(by_lr.keys())
    lr_mid = lrs_sorted[len(lrs_sorted)//2]
    mid = sorted(by_lr[lr_mid], key=lambda z: z["epoch"])
    series_sii = {
        "SII $R^2$ (o1)": [z["r2_order_1"] for z in mid],
        "SII $R^2$ (o2)": [z["r2_order_2"] for z in mid],
        "SII $R^2$ (o3)": [z["r2_order_3"] for z in mid],
    }
    plot_curves(
        x=[z["epoch"] for z in mid], y_series=series_sii,
        ylabel="SII $R^2$", title=f"SII $R^2$ vs epoch (LR={lr_mid:.0e})",
        outfile=str(outdir / f"sii_r2_vs_epoch_lr{lr_mid:.0e}.png"),
        xlabel="Epoch", legend=True
    )
    # Scatter: train R² vs SII R² (all points)
    for k in (1, 2, 3):
        x = [r["train_r2"] for r in rows]
        y = [r[f"r2_order_{k}"] for r in rows]
        fig, ax = plt.subplots(figsize=(4,4))
        ax.scatter(x, y, s=12, alpha=0.6)
        lo = min(min(x), min(y)); hi = max(max(x), max(y))
        ax.plot([lo,hi],[lo,hi], linestyle=":")
        ax.set_xlabel("Train $R^2$"); ax.set_ylabel(f"SII $R^2$ (o{k})")
        ax.set_title(f"Train $R^2$ vs SII $R^2$ (o{k})")
        ax.grid(True, linestyle=":", linewidth=0.5)
        fig.tight_layout()
        outp = outdir / f"scatter_trainR2_vs_SIIo{k}.png"
        _ensure_parent_dir(outp)
        fig.savefig(outp, dpi=300)
        plt.close(fig)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_core", type=str, default=".", help="Root containing out_tn_easy, out_tn_hard, out_generic")
    ap.add_argument("--root_abl_rank", type=str, default="abl_tn_rank_sweep")
    ap.add_argument("--root_abl_lr", type=str, default="abl_generic_progress_sweep")
    ap.add_argument("--orders", type=int, default=8)
    ap.add_argument("--out", type=str, default="figs")
    ap.add_argument("--combined_lr", type=float, default=None,
                    help="Optional: specific LR to use for combined Train/SII R² vs epoch plot (e.g., 0.003)")
    args = ap.parse_args()

    outdir = Path(args.out)
    # Pre-create common subfolders to avoid FileNotFoundError
    for d in ["scatters", "per_order_lines", "rank_ablation", "generic_sweep", "tables"]:
        (outdir / d).mkdir(parents=True, exist_ok=True)

    # --- core experiments ---
    core_specs = [
        ("out_tn_easy",  "tn_gt_rank3_student_rank3"),
        ("out_tn_hard",  "tn_gt_rank16_student_rank3"),
        ("out_generic",  "generic_order_leq3_student_rank12"),
    ]
    core_exps = []
    for subdir, base in core_specs:
        expdir = Path(args.root_core) / subdir
        if expdir.exists():
            core_exps.append(load_core_exp(expdir, base, args.orders))

    # plots: per-order lines & scatters (k≤3)
    plot_per_order_lines(core_exps, outdir / "per_order_lines", args.orders)
    for e in core_exps:
        for k in (1, 2, 3):
            if k not in e["per_order"]: 
                continue
            y = e["per_order"][k]["gt"]; yhat = e["per_order"][k]["st"]
            r2 = safe_r2(y, yhat)
            fig, ax = plt.subplots(figsize=(4,4))
            ax.scatter(y, yhat, s=10, alpha=0.6)
            lo, hi = float(min(y.min(), yhat.min())), float(max(y.max(), yhat.max()))
            ax.plot([lo,hi],[lo,hi], linestyle=":")
            ax.set_xlabel("GT SII"); ax.set_ylabel("Student SII")
            ax.set_title(f"{e['name']} | o{k} | safe-$R^2$={r2:.3f}")
            ax.grid(True, linestyle=":", linewidth=0.5)
            fig.tight_layout()
            outp = outdir / "scatters" / f"{e['name']}_scatter_k{k}.png"
            _ensure_parent_dir(outp)
            fig.savefig(outp, dpi=300)
            plt.close(fig)

    # --- rank ablation ---
    rank_csv = Path(args.root_abl_rank) / "sweep.csv"
    if rank_csv.exists():
        sweep = load_rank_sweep(rank_csv)
        plot_rank_heatmap_and_curves(sweep["rows"], outdir / "rank_ablation", orders=3)
        tex_rank = latex_table_rank_sweep(sweep["rows"])
        with open(_ensure_parent_dir(outdir / "tables" / "tab_rank_ablation.tex"), "w") as f:
            f.write(tex_rank)

    # --- LR sweep (loss/R² vs SII R²) ---
    lr_csv = Path(args.root_abl_lr) / "combined_progress.csv"
    if lr_csv.exists():
        rows = load_lr_sweep(lr_csv)
        plot_lr_sweep(rows, outdir / "generic_sweep")
        tex_lr = latex_table_lr_final(rows)
        with open(_ensure_parent_dir(outdir / "tables" / "tab_lr_vs_sii.tex"), "w") as f:
            f.write(tex_lr)
        # one combined plot with both train R2 and SII R2 vs epoch
        plot_lr_sweep_combined(rows, outdir / "generic_sweep", lr_choice=args.combined_lr)

    # --- core table (three teacher types) ---
    if core_exps:
        tex_core = latex_table_core(core_exps)
        with open(_ensure_parent_dir(outdir / "tables" / "tab_core_teachers.tex"), "w") as f:
            f.write(tex_core)

    print(f"Saved figures to {outdir.resolve()}")
    print(f"LaTeX tables saved to { (outdir / 'tables').resolve() }")

if __name__ == "__main__":
    main()
