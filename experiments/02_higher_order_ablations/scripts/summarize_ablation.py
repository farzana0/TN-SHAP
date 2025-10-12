#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize TN feature-map ablation results and export:
  1) One combined summary CSV across all mfmap and k
  2) One LaTeX table (booktabs) with rows=mfmap, grouped columns per order k

It expects a folder layout like:
  <root>/<dataset>_seed<seed>_K<K>/
      mgrid<m>_mfmap<MM>/
        <dataset>_summary_local_eval.csv  # produced by ablate_fmap_and_eval.py
      ...
  and/or the master CSV:
      <dataset>_ablation_fmap_eval_seed<seed>.csv

We will:
  - Read master CSV if present (preferred), else union all per-run summaries.
  - Aggregate (optionally average over k if you pass --aggregate-by-k mean).
  - Emit:
      <dataset>_ablation_summary.csv
      <dataset>_ablation_table.tex
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from glob import glob
from textwrap import dedent

def _find_master_csv(root_dir: str, dataset: str, seed: int):
    patt = os.path.join(root_dir, f"{dataset}_ablation_fmap_eval_seed{seed}.csv")
    return patt if os.path.isfile(patt) else None

def _collect_per_run_summaries(root_dir: str, dataset: str):
    # Look for */<dataset>_summary_local_eval.csv under root_dir
    paths = glob(os.path.join(root_dir, "mgrid*_mfmap*", f"{dataset}_summary_local_eval.csv"))
    dfs = []
    for p in sorted(paths):
        try:
            df = pd.read_csv(p)
            # add mfmap/mgrid parsed from folder
            run_dir = os.path.dirname(p)
            mgrid = None; mfmap = None
            m = re.search(r"mgrid(\d+)", run_dir)
            if m: mgrid = int(m.group(1))
            m = re.search(r"mfmap(\d+)", run_dir)
            if m: mfmap = int(m.group(1))
            if "fmap_out_dim" not in df.columns:
                df["fmap_out_dim"] = mfmap
            if "mgrid" not in df.columns:
                df["mgrid"] = mgrid
            dfs.append(df)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _load_all(root_dir: str, dataset: str, seed: int):
    master = _find_master_csv(root_dir, dataset, seed)
    if master:
        df = pd.read_csv(master)
        # If mgrid is missing, try to infer by peeking run subfolders (optional)
        return df
    # Fallback: union per-run summaries
    return _collect_per_run_summaries(root_dir, dataset)

def _format_float(x, digits=3):
    if pd.isna(x): return "-"
    return f"{x:.{digits}f}"

def build_summary_csv(df_in: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    """
    Normalize columns and write a single summary CSV:
      columns kept: fmap_out_dim, order_k, cos_mu/sd, r2_mu/sd, mse_mu/sd, train_time_s (mean), train_final_r2 (mean)
    """
    if df_in.empty:
        raise RuntimeError("No rows found to summarize.")
    df = df_in.copy()

    # Normalize column names we expect from the ablation script
    rename = {
        "cos_vs_exact_mu": "cos_mu",
        "cos_vs_exact_sd": "cos_sd",
        "r2_vs_exact_mu":  "r2_mu",
        "r2_vs_exact_sd":  "r2_sd",
        "mse_vs_exact_mu": "mse_mu",
        "mse_vs_exact_sd": "mse_sd",
    }
    for k,v in rename.items():
        if k in df.columns:
            df = df.rename(columns={k:v})

    # Ensure required columns exist
    req = ["fmap_out_dim", "order_k", "cos_mu", "r2_mu", "mse_mu"]
    for c in req:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column '{c}' in input summaries, got columns={list(df.columns)}")

    # Optional: carry training stats if present
    # There might be multiple rows per mfmap & k (e.g., multiple seeds/runs); average them.
    agg_map = {
        "cos_mu": "mean", "cos_sd": "mean",
        "r2_mu": "mean",  "r2_sd": "mean",
        "mse_mu": "mean", "mse_sd": "mean",
    }
    if "train_time_s" in df.columns: agg_map["train_time_s"] = "mean"
    if "train_final_r2" in df.columns: agg_map["train_final_r2"] = "mean"
    if "tn_seed" in df.columns: agg_map["tn_seed"] = "first"  # record a representative seed

    group_cols = ["fmap_out_dim", "order_k"]
    out = df.groupby(group_cols, as_index=False).agg(agg_map)

    out.to_csv(out_csv, index=False)
    return out

def build_latex_table(df_summary: pd.DataFrame, out_tex: str,
                      show_sd: bool = False,
                      digits: int = 3,
                      caption: str = "TN selector vs Teacher (Diabetes): cosine, $R^2$, MSE by feature-map size and order $k$",
                      label: str = "tab:ablation_fmap",
                      ):
    """
    Build one LaTeX table with rows = fmap_out_dim, grouped columns per k.
    For each k column group, we print 'cos', 'R^2', 'MSE' (mean only by default).
    """
    # Pivot into a convenient structure: {mfmap: {k: {metric: value}}}
    ks = sorted(df_summary["order_k"].unique())
    fmmaps = sorted(df_summary["fmap_out_dim"].unique())

    # Build column headers
    header_main = ["\\toprule", "\\multirow{2}{*}{$m_{\\text{fmap}}$}"]
    for k in ks:
        header_main.append(f" & \\multicolumn{{3}}{{c}}{{Order $k={k}$}}")
    header_main.append(" \\\\")
    header_sub = ["\\cmidrule(lr){2-" + str(1+3*len(ks)) + "}"]
    # More explicit sub-headers:
    sub = [" "]  # aligns with multirow cell
    for _ in ks:
        sub += [" & cos", " & $R^2$", " & MSE"]
    sub.append(" \\\\ \\midrule")

    # Rows
    lines = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{6pt}", "\\renewcommand{\\arraystretch}{1.15}"]
    ncols = 1 + 3*len(ks)
    lines.append("\\begin{tabular}{" + "c" * ncols + "}")


    lines.append("\\toprule")
    # header lines
    lines.append("\\multirow{2}{*}{$m_{\\text{fmap}}$}" + "".join([f" & \\multicolumn{{3}}{{c}}{{Order $k={k}$}}" for k in ks]) + " \\\\")
    # cmidrules for each k group
    # compute start-end column indices for cmidrules
    # first column is mfmap; for groups: (2..4), (5..7), ...
    cmid = []
    for gi, _ in enumerate(ks):
        start = 2 + gi*3
        end = start + 2
        cmid.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmid))
    # subheader row
    lines.append(" " + "".join([" & cos & $R^2$ & MSE" for _ in ks]) + " \\\\ \\midrule")

    # emit rows
    for m in fmmaps:
        row = [f"{m}"]
        for k in ks:
            row_df = df_summary[(df_summary["fmap_out_dim"] == m) & (df_summary["order_k"] == k)]
            if len(row_df) == 0:
                row += ["-", "-", "-"]
                continue
            r = row_df.iloc[0]
            row += [
                _format_float(r.get("cos_mu"), digits),
                _format_float(r.get("r2_mu"), digits),
                _format_float(r.get("mse_mu"), digits),
            ]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path like: out_local_student_singlegrid/diabetes_seed2711_K100")
    ap.add_argument("--dataset", required=True, help="e.g., diabetes")
    ap.add_argument("--seed", type=int, required=True, help="seed used in ablation")
    ap.add_argument("--out-prefix", type=str, default=None,
                    help="Optional prefix for outputs; defaults to <dataset>")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_prefix = args.out_prefix or args.dataset

    # 1) Load
    df = _load_all(root, args.dataset, args.seed)
    if df.empty:
        raise SystemExit(f"No ablation summaries found under: {root}")

    # 2) Build combined summary CSV
    out_csv = os.path.join(root, f"{out_prefix}_ablation_summary.csv")
    df_summary = build_summary_csv(df, out_csv)
    print(f"[OK] summary CSV -> {out_csv}")

    # 3) Build LaTeX table
    out_tex = os.path.join(root, f"{out_prefix}_ablation_table.tex")
    _ = build_latex_table(
        df_summary,
        out_tex=out_tex,
        caption=f"TN selector vs Teacher on {args.dataset.title()}: cosine, $R^2$, MSE by feature-map size and order $k$",
        label=f"tab:{args.dataset}_ablation_fmap"
    )
    print(f"[OK] LaTeX table  -> {out_tex}")

if __name__ == "__main__":
    main()
