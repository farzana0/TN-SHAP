#!/usr/bin/env python3
# aggregate_local_eval.py
"""
Aggregate per-target CSVs produced by evaluate_local_selector.py and compute
mean/sd across targets for each (order_k, method, budget).

It reads files like:
  <outdir>/<dataset>_seed<seed>/<dataset>_idx{idx}_order{k}_local_eval.csv

Outputs:
  <outdir>/<dataset>_seed<seed>/<dataset>_agg_summary.csv
and a more compact per-order best-by-cos table:
  <outdir>/<dataset>_seed<seed>/<dataset>_agg_bestcos_by_order.csv
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Same --outdir used by evaluator")
    ap.add_argument("--dataset", required=True, choices=["concrete","energy_y1","energy_y2","diabetes","california"])
    ap.add_argument("--seed", type=int, default=2711)
    args = ap.parse_args()

    root = os.path.join(args.outdir, f"{args.dataset}_seed{args.seed}")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Folder not found: {root}")

    files = sorted(glob.glob(os.path.join(root, f"{args.dataset}_idx*_order*_local_eval.csv")))
    if not files:
        raise FileNotFoundError(f"No per-idx CSVs found under {root}")

    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")

    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    # Basic sanity
    keep_cols = [
        "dataset","seed","idx","order_k",
        "method","baseline","budget",
        "time_s_mu","time_s_sd",
        "cos_vs_exact_mu","cos_vs_exact_sd",
        "r2_vs_exact_mu","r2_vs_exact_sd",
        "mse_vs_exact_mu","mse_vs_exact_sd",
        "time_exact_s_teacher"  # from evaluator (same for all rows of a target)
    ]
    df_all = df_all[keep_cols]

    # Average over targets (idx), keep (order_k, method, budget)
    group_cols = ["dataset","seed","order_k","method","baseline","budget"]
    agg = df_all.groupby(group_cols).agg(
        n_targets=("idx","nunique"),
        time_s_mu_mean=("time_s_mu","mean"),
        time_s_mu_std =("time_s_mu","std"),
        cos_vs_exact_mu_mean=("cos_vs_exact_mu","mean"),
        cos_vs_exact_mu_std =("cos_vs_exact_mu","std"),
        r2_vs_exact_mu_mean =("r2_vs_exact_mu","mean"),
        r2_vs_exact_mu_std  =("r2_vs_exact_mu","std"),
        mse_vs_exact_mu_mean=("mse_vs_exact_mu","mean"),
        mse_vs_exact_mu_std =("mse_vs_exact_mu","std"),
        # teacher exact timing: average across targets for reference
        time_exact_s_teacher_mean=("time_exact_s_teacher","mean"),
        time_exact_s_teacher_std =("time_exact_s_teacher","std"),
    ).reset_index()

    out_summary = os.path.join(root, f"{args.dataset}_agg_summary.csv")
    agg.to_csv(out_summary, index=False)
    print(f"[OK] wrote {out_summary}")

    # Convenience table: for each order_k, pick the best-cos method(s)
    best_rows = []
    for k, sub in agg.groupby("order_k"):
        # rank by cosine mean (desc), then by time (asc)
        sub = sub.sort_values(["cos_vs_exact_mu_mean", "time_s_mu_mean"], ascending=[False, True])
        top = sub.head(10)  # keep a few best options
        best_rows.append(top.assign(rank=np.arange(1, len(top)+1)))
    if best_rows:
        best_tbl = pd.concat(best_rows, axis=0, ignore_index=True)
        out_best = os.path.join(root, f"{args.dataset}_agg_bestcos_by_order.csv")
        best_tbl.to_csv(out_best, index=False)
        print(f"[OK] wrote {out_best}")

    # (Optional) pretty print pivots per order
    try:
        for k, sub in agg.groupby("order_k"):
            piv = sub.pivot_table(index=["method","baseline","budget"],
                                  values=["cos_vs_exact_mu_mean","time_s_mu_mean"],
                                  aggfunc="first").sort_values(("cos_vs_exact_mu_mean",""), ascending=False)
            print(f"\n=== Order k={k} (top by cosine) ===")
            print(piv.head(10))
    except Exception:
        pass

if __name__ == "__main__":
    main()
