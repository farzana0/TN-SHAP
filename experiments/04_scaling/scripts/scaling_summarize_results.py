#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# -*- coding: utf-8 -*-
"""
Summarize HD sweep results:
- Read a master CSV (merged across d)
- Emit LaTeX table: d, TN train time, TNShap runtime, Exact runtime, accuracy (cos/R2/MSE)
- Emit plot: d vs TNShap runtime

Inputs:
  --master   path to merged CSV (default: ./out_hd_sweep/master_merged.csv)
  --outdir   output dir for .tex/.csv/.png (default: ./out_hd_sweep/summary)
  --order    interaction order to report: 1, 2, or 3 (default: 2)
  --source   student source to use: student_random | student_masked (default: student_random)
  --tex-name LaTeX filename (default: table_k{order}.tex)
  --plot-name plot filename (default: tnshap_runtime_k{order}.png)
  --agg      aggregate multiple rows per d: mean | median (default: median)
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        x = float(x)
    except Exception:
        return "—"
    if abs(x) >= 1000:
        return f"{x:.0f}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.3f}".rstrip('0').rstrip('.') or "0"

def safe_agg(series: pd.Series, how: str):
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return np.nan
    return float(s.median() if how == "median" else s.mean())

def first_existing(df, cols):
    """Return the first column name in cols that exists in df, else None."""
    for c in cols:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, default="./out_hd_sweep/master_merged.csv")
    ap.add_argument("--outdir", type=str, default="./out_hd_sweep/summary")
    ap.add_argument("--order", type=int, default=2, choices=[1,2,3])
    ap.add_argument("--source", type=str, default="student_random", choices=["student_random","student_masked"])
    ap.add_argument("--tex-name", type=str, default=None)
    ap.add_argument("--plot-name", type=str, default=None)
    ap.add_argument("--agg", type=str, default="median", choices=["mean","median"])
    args = ap.parse_args()

    ensure_dir(args.outdir)
    if args.tex_name is None:
        args.tex_name = f"table_k{args.order}.tex"
    if args.plot_name is None:
        args.plot_name = f"tnshap_runtime_k{args.order}.png"

    if not os.path.isfile(args.master):
        raise FileNotFoundError(args.master)
    df = pd.read_csv(args.master)

    # Normalize essential columns presence
    for c in ["source","method","baseline","k","order_k","d","runtime_s","time_s_mu",
              "time_exact_s_teacher","cosine_sim","r2","mse",
              "student_train_time_s","train_time_s","tn_train_s"]:
        if c not in df.columns:
            df[c] = np.nan

    # Resolve order column: accept 'k' or 'order_k'
    order_col = "k" if "k" in df.columns else ("order_k" if "order_k" in df.columns else None)
    if order_col is None:
        # If order missing, assume all rows belong to requested order
        df["__order__"] = args.order
        order_col = "__order__"

    # Filter to requested order
    df = df[pd.to_numeric(df[order_col], errors="coerce") == args.order].copy()

    # Resolve dimension column
    d_col = first_existing(df, ["d", "dim", "D"])
    if d_col is None:
        raise ValueError("No dimension column found (expected one of: d, dim, D).")

    # Identify TNShap runtime column(s) (from student eval rows)
    tn_runtime_col = first_existing(df, ["runtime_s", "time_s_mu", "t_total_s"])
    # Exact (teacher/GT) runtime column (often carried alongside rows)
    exact_runtime_col = first_existing(df, ["time_exact_s_teacher", "exact_runtime_s", "gt_runtime_s"])

    # Student train time columns
    train_time_col = first_existing(df, ["student_train_time_s", "train_time_s", "tn_train_s"])

    # Accuracy columns (if present)
    cos_col = first_existing(df, ["cosine_sim", "cos_vs_exact_mu"])
    r2_col  = first_existing(df, ["r2", "r2_vs_exact_mu"])
    mse_col = first_existing(df, ["mse", "mse_vs_exact_mu"])

    # Select TNShap rows from the requested student source (fallback if 'source' absent)
    stu = df[(df.get("method") == "tn_selector") & (df.get("baseline") == "TNShap")]
    if "source" in df.columns and not df["source"].isna().all():
        stu = stu[(stu.get("source") == args.source) | (stu.get("source").isna() & True)]

    # Exact rows (generator), or fallback: pull exact time from the TN rows’ column
    gt = df[(df.get("source") == "generator") & (df.get("method") == "gt_exact")]
    # If no explicit generator rows exist, we’ll read exact time from tn rows via exact_runtime_col.

    # Aggregate per d
    rows = []
    d_values = sorted(set(pd.to_numeric(df[d_col], errors="coerce").dropna().astype(int).tolist()))
    for d in d_values:
        stu_d = stu[pd.to_numeric(stu[d_col], errors="coerce") == d]
        gt_d  = gt [pd.to_numeric(gt [d_col], errors="coerce") == d]

        # TNShap runtime (selector path)
        tn_time = safe_agg(stu_d[tn_runtime_col], args.agg) if tn_runtime_col else np.nan
        # Student train time
        train_time = safe_agg(stu_d[train_time_col], args.agg) if train_time_col else np.nan
        # Accuracy vs exact (if available)
        cos = safe_agg(stu_d[cos_col], args.agg) if cos_col else np.nan
        r2  = safe_agg(stu_d[r2_col], args.agg) if r2_col else np.nan
        mse = safe_agg(stu_d[mse_col], args.agg) if mse_col else np.nan

        # Exact runtime: prefer explicit generator rows; otherwise read from the TN rows' exact column
        if not gt_d.empty and tn_runtime_col:
            exact_time = safe_agg(gt_d[tn_runtime_col], args.agg)  # if generator wrote into runtime_s/time_s_mu
        else:
            exact_time = safe_agg(stu_d[exact_runtime_col], args.agg) if exact_runtime_col else np.nan

        rows.append(dict(
            d=int(d),
            student_train_time_s=train_time,
            tnshap_runtime_s=tn_time,
            exact_runtime_s=exact_time,
            cosine_sim=cos,
            r2=r2,
            mse=mse,
        ))

    tab = pd.DataFrame(rows).sort_values("d")

    # LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    cap_src = args.source.replace('_', r'\_')
    latex_lines.append(rf"\caption{{Scalability summary for order $k={args.order}$ ({cap_src}).}}")
    latex_lines.append(r"\begin{tabular}{r r r r r r r}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"$d$ & Train (s) & TNShap (s) & Exact (s) & Cosine $\uparrow$ & $R^2\uparrow$ & MSE $\downarrow$ \\")
    latex_lines.append(r"\midrule")
    for _, r in tab.iterrows():
        latex_lines.append(
            f"{int(r['d'])} & "
            f"{fmt(r['student_train_time_s'])} & "
            f"{fmt(r['tnshap_runtime_s'])} & "
            f"{fmt(r['exact_runtime_s'])} & "
            f"{fmt(r['cosine_sim'])} & "
            f"{fmt(r['r2'])} & "
            f"{fmt(r['mse'])} \\\\"
        )
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    tex_path = os.path.join(args.outdir, args.tex_name)
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"[OK] LaTeX table -> {tex_path}")

    # Plot d vs TNShap runtime
    plot_df = tab.dropna(subset=["tnshap_runtime_s"])
    if not plot_df.empty:
        plt.figure(figsize=(5.0,3.2))
        plt.plot(plot_df["d"].values, plot_df["tnshap_runtime_s"].values, marker="o")
        plt.xlabel("Dimension $d$")
        plt.ylabel("TNShap runtime (s)")
        yvals = plot_df["tnshap_runtime_s"].to_numpy(dtype=float)
        if np.nanmax(yvals) / max(np.nanmin(yvals), 1e-12) > 50:
            plt.yscale("log")
        plt.tight_layout()
        plot_path = os.path.join(args.outdir, args.plot_name)
        plt.savefig(plot_path, dpi=220)
        print(f"[OK] Plot -> {plot_path}")
    else:
        print("[WARN] No TNShap runtime to plot.")

    # Also save numeric table
    tab_csv = os.path.join(args.outdir, f"table_k{args.order}.csv")
    tab.to_csv(tab_csv, index=False)
    print(f"[OK] Numeric table -> {tab_csv}")

if __name__ == "__main__":
    main()
