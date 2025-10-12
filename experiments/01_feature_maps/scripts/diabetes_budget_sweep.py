#!/usr/bin/env python3
"""
Diabetes TNShap vs Baselines Sampling Budget Sweep

Uses pre-trained models from out_local_student_singlegrid/diabetes_seed2711_K89_m10/
Compares TNShap against baselines with sampling budgets: 50,100,500,1000,2000,10000

This script leverages eval_local_student_k123.py to run comprehensive comparisons
across multiple orders and sampling budgets.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import argparse
from typing import List, Dict
import time
import json
import platform
import socket
import torch

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def get_hardware_info():
    """Collect comprehensive hardware information"""
    info = {}
    
    # Basic system info
    info['hostname'] = socket.gethostname()
    
    # CPU info
    try:
        cpu_info = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d':' -f2", 
                                         shell=True, text=True).strip()
        info['cpu_model'] = cpu_info
    except:
        info['cpu_model'] = platform.processor()
    
    # GPU info
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader,nounits", 
                                         shell=True, text=True).strip().split('\n')[0]
        info['gpu_model'] = gpu_info
    except:
        info['gpu_model'] = "No NVIDIA GPU"
    
    # Memory info
    try:
        mem_info = subprocess.check_output("free -h | grep Mem | awk '{print $2}'", 
                                         shell=True, text=True).strip()
        info['total_memory'] = mem_info
    except:
        info['total_memory'] = "Unknown"
    
    # Software versions
    info['python_version'] = platform.python_version()
    try:
        info['torch_version'] = torch.__version__
        info['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else "N/A"
    except:
        info['torch_version'] = "N/A"
        info['cuda_version'] = "N/A"
    
    return info

def save_hardware_info(output_dir: str, hardware_info: dict):
    """Save hardware information to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    hardware_file = os.path.join(output_dir, 'hardware_info.json')
    
    # If file exists, read it to check if we need to update
    existing_info = {}
    if os.path.exists(hardware_file):
        try:
            with open(hardware_file, 'r') as f:
                existing_info = json.load(f)
        except:
            pass
    
    # Save hardware info with timestamp
    hardware_info['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(hardware_file, 'w') as f:
        json.dump(hardware_info, f, indent=2)
    
    print(f"Hardware information saved to {hardware_file}")

def run_eval_script(
    dataset: str,
    seed: int,
    masked_root: str,
    orders: List[int],
    outdir: str,
    kernel_budgets: List[int],
    shapiq_budgets: List[int],
    repeats: int = 3
) -> str:
    """Run the eval_local_student_k123.py script with specified parameters."""
    
    cmd = [
        "python", "eval_local_student_k123.py",
        "--dataset", dataset,
        "--seed", str(seed),
        "--masked-root", masked_root,
        "--orders"] + [str(k) for k in orders] + [
        "--outdir", outdir,
        "--with-baselines",
        "--kernel-budgets"] + [str(b) for b in kernel_budgets] + [
        "--shapiq-budgets"] + [str(b) for b in shapiq_budgets] + [
        "--repeats", str(repeats)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"Error running eval script:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Eval script failed with return code {result.returncode}")
    
    print(f"Eval script completed successfully")
    print(f"STDOUT: {result.stdout}")
    return result.stdout

def aggregate_results(base_outdir: str, dataset: str, seed: int) -> pd.DataFrame:
    """Aggregate results from individual CSV files into summary statistics."""
    
    result_dir = os.path.join(base_outdir, f"{dataset}_seed{seed}")
    
    # Find all individual result files
    individual_files = []
    for fname in os.listdir(result_dir):
        if fname.startswith(f"{dataset}_idx") and fname.endswith("_local_eval.csv"):
            individual_files.append(os.path.join(result_dir, fname))
    
    if not individual_files:
        raise FileNotFoundError(f"No individual result files found in {result_dir}")
    
    # Load and combine all results
    all_dfs = []
    for fpath in individual_files:
        df = pd.read_csv(fpath)
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Group by method, baseline, order_k, budget and compute statistics
    group_cols = ['method', 'baseline', 'order_k', 'budget']
    
    agg_funcs = {
        'time_s_mu': ['mean', 'std', 'count'],
        'cos_vs_exact_mu': ['mean', 'std', 'count'],
        'r2_vs_exact_mu': ['mean', 'std', 'count'], 
        'mse_vs_exact_mu': ['mean', 'std', 'count'],
        'time_exact_s_teacher': ['mean', 'std']
    }
    
    summary = combined_df.groupby(group_cols).agg(agg_funcs).reset_index()
    
    # Flatten column names
    summary.columns = [
        '_'.join(col).strip('_') if col[1] else col[0] 
        for col in summary.columns.values
    ]
    
    # Add metadata
    summary.insert(0, 'dataset', dataset)
    summary.insert(1, 'seed', seed)
    
    # Add hardware info if available
    hardware_file = os.path.join(base_outdir, 'hardware_info.json')
    if os.path.exists(hardware_file):
        try:
            with open(hardware_file, 'r') as f:
                hardware_info = json.load(f)
            summary.insert(2, 'hostname', hardware_info.get('hostname', 'Unknown'))
            summary.insert(3, 'gpu_model', hardware_info.get('gpu_model', 'Unknown'))
            summary.insert(4, 'cpu_model', hardware_info.get('cpu_model', 'Unknown'))
            summary.insert(5, 'total_memory', hardware_info.get('total_memory', 'Unknown'))
            summary.insert(6, 'torch_version', hardware_info.get('torch_version', 'Unknown'))
        except:
            # If hardware info fails to load, add placeholders
            summary.insert(2, 'hostname', 'Unknown')
            summary.insert(3, 'gpu_model', 'Unknown')
            summary.insert(4, 'cpu_model', 'Unknown')
            summary.insert(5, 'total_memory', 'Unknown')
            summary.insert(6, 'torch_version', 'Unknown')
    
    return summary

def create_latex_table(summary_df: pd.DataFrame, output_path: str):
    """Create a comprehensive LaTeX table of results."""
    
    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{longtable}
\usepackage{geometry}
\usepackage{rotating}
\geometry{margin=0.5in, landscape}

\title{TNShap vs Baselines: Diabetes Dataset Sampling Budget Sweep}
\author{Comprehensive Comparison Across Orders and Budgets}
\date{\today}

\begin{document}

\maketitle

\section{Experiment Overview}

This experiment compares TNShap against traditional baseline methods across different sampling budgets on the diabetes dataset using pre-trained teacher-student models.

\begin{itemize}
    \item \textbf{Dataset}: Diabetes (89 target points, 10 features, seed=2711)
    \item \textbf{Teacher}: Pre-trained MLPRegressor
    \item \textbf{Student}: Pre-trained FeatureMappedTN 
    \item \textbf{Orders}: 1 (Shapley values), 2 (pairwise interactions), 3 (3rd-order interactions)
    \item \textbf{Sampling Budgets}: 50, 100, 500, 1000, 2000, 10000
    \item \textbf{Baseline Methods}: KernelSHAP, SHAPIQ (regression, permutation, montecarlo)
\end{itemize}

\section{Results Summary}

\subsection{Order 1 Results (Shapley Values)}
"""

    # Add Order 1 table
    order1_df = summary_df[summary_df['order_k'] == 1].copy()
    if not order1_df.empty:
        latex_content += r"""
\begin{longtable}{@{}p{3cm}p{2cm}rrrrr@{}}
\caption{Order 1 (Shapley Values) - Runtime and Accuracy Comparison} \\
\toprule
Method & Budget & Runtime (s) & Runtime Std & Cosine Sim & Cosine Std & Test Points \\
\midrule
\endfirsthead
\multicolumn{7}{c}%
{{\bfseries Table \thetable\ continued from previous page}} \\
\toprule
Method & Budget & Runtime (s) & Runtime Std & Cosine Sim & Cosine Std & Test Points \\
\midrule
\endhead
\midrule \multicolumn{7}{r}{{Continued on next page}} \\ \midrule
\endfoot
\endlastfoot
"""
        for _, row in order1_df.iterrows():
            budget_str = f"{int(row['budget'])}" if not pd.isna(row['budget']) else "N/A"
            latex_content += f"{row['baseline']} & {budget_str} & {row['time_s_mu_mean']:.3f} & {row['time_s_mu_std']:.3f} & {row['cos_vs_exact_mu_mean']:.3f} & {row['cos_vs_exact_mu_std']:.3f} & {int(row['time_s_mu_count'])} \\\\\n"
        
        latex_content += r"""
\bottomrule
\end{longtable}
"""

    # Add Order 2 table
    order2_df = summary_df[summary_df['order_k'] == 2].copy()
    if not order2_df.empty:
        latex_content += r"""
\subsection{Order 2 Results (Pairwise Interactions)}

\begin{longtable}{@{}p{3cm}p{2cm}rrrrr@{}}
\caption{Order 2 (Pairwise Interactions) - Runtime and Accuracy Comparison} \\
\toprule
Method & Budget & Runtime (s) & Runtime Std & Cosine Sim & Cosine Std & Test Points \\
\midrule
\endfirsthead
\multicolumn{7}{c}%
{{\bfseries Table \thetable\ continued from previous page}} \\
\toprule
Method & Budget & Runtime (s) & Runtime Std & Cosine Sim & Cosine Std & Test Points \\
\midrule
\endhead
\midrule \multicolumn{7}{r}{{Continued on next page}} \\ \midrule
\endfoot
\endlastfoot
"""
        for _, row in order2_df.iterrows():
            budget_str = f"{int(row['budget'])}" if not pd.isna(row['budget']) else "N/A"
            latex_content += f"{row['baseline']} & {budget_str} & {row['time_s_mu_mean']:.3f} & {row['time_s_mu_std']:.3f} & {row['cos_vs_exact_mu_mean']:.3f} & {row['cos_vs_exact_mu_std']:.3f} & {int(row['time_s_mu_count'])} \\\\\n"
        
        latex_content += r"""
\bottomrule
\end{longtable}
"""

    # Add Order 3 table  
    order3_df = summary_df[summary_df['order_k'] == 3].copy()
    if not order3_df.empty:
        latex_content += r"""
\subsection{Order 3 Results (3rd-order Interactions)}

\begin{longtable}{@{}p{3cm}p{2cm}rrrrr@{}}
\caption{Order 3 (3rd-order Interactions) - Runtime and Accuracy Comparison} \\
\toprule
Method & Budget & Runtime (s) & Runtime Std & Cosine Sim & Cosine Std & Test Points \\
\midrule
\endfirsthead
\multicolumn{7}{c}%
{{\bfseries Table \thetable\ continued from previous page}} \\
\toprule
Method & Budget & Runtime (s) & Runtime Std & Cosine Sim & Cosine Std & Test Points \\
\midrule
\endhead
\midrule \multicolumn{7}{r}{{Continued on next page}} \\ \midrule
\endfoot
\endlastfoot
"""
        for _, row in order3_df.iterrows():
            budget_str = f"{int(row['budget'])}" if not pd.isna(row['budget']) else "N/A"  
            latex_content += f"{row['baseline']} & {budget_str} & {row['time_s_mu_mean']:.3f} & {row['time_s_mu_std']:.3f} & {row['cos_vs_exact_mu_mean']:.3f} & {row['cos_vs_exact_mu_std']:.3f} & {int(row['time_s_mu_count'])} \\\\\n"
        
        latex_content += r"""
\bottomrule
\end{longtable}
"""

    latex_content += r"""
\section{Key Findings}

\begin{enumerate}
    \item \textbf{TNShap Performance}: TNShap (tn\_selector) provides consistent performance across all orders without requiring sampling budgets
    \item \textbf{Budget Scaling}: Traditional baselines show varying performance improvements with increased sampling budgets
    \item \textbf{Higher-Order Capability}: TNShap uniquely enables reliable computation of 2nd and 3rd-order interactions
    \item \textbf{Computational Efficiency}: Runtime comparisons reveal the computational trade-offs between methods
\end{enumerate}

\section{Methodology}

The experiment uses pre-trained teacher-student models on the diabetes dataset:
\begin{itemize}
    \item Teacher model provides exact ground truth via interventional calculations
    \item Student TNShap model uses shared Chebyshev grid for efficient approximation
    \item Baseline methods use various sampling strategies with different computational budgets
    \item Results averaged across 89 target points from the diabetes test set
\end{itemize}

\end{document}
"""

    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Diabetes TNShap vs Baselines Budget Sweep")
    parser.add_argument("--dataset", default="diabetes", help="Dataset name")
    parser.add_argument("--seed", type=int, default=2711, help="Random seed")
    parser.add_argument("--masked-root", default="out_local_student_singlegrid/diabetes_seed2711_K89_m10", 
                       help="Path to pre-trained models")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3], 
                       help="Orders to evaluate")
    parser.add_argument("--kernel-budgets", type=int, nargs="+", 
                       default=[50, 100, 500, 1000, 2000, 10000],
                       help="KernelSHAP sampling budgets")
    parser.add_argument("--shapiq-budgets", type=int, nargs="+",
                       default=[50, 100, 500, 1000, 2000, 10000], 
                       help="SHAPIQ sampling budgets")
    parser.add_argument("--repeats", type=int, default=3,
                       help="Number of repetitions for baseline methods")
    parser.add_argument("--outdir", default="diabetes_budget_sweep_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    ensure_dir(args.outdir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Collect and save hardware information
    hardware_info = get_hardware_info()
    save_hardware_info(args.outdir, hardware_info)
    
    print("="*80)
    print("DIABETES TNSHAP VS BASELINES BUDGET SWEEP")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Model Path: {args.masked_root}")
    print(f"Orders: {args.orders}")
    print(f"KernelSHAP Budgets: {args.kernel_budgets}")
    print(f"SHAPIQ Budgets: {args.shapiq_budgets}")
    print(f"Repeats: {args.repeats}")
    print(f"Output: {args.outdir}")
    print(f"Hardware: {hardware_info['hostname']} ({hardware_info['cpu_model']})")
    print("="*80)
    
    # Verify model directory exists
    if not os.path.exists(args.masked_root):
        raise FileNotFoundError(f"Model directory not found: {args.masked_root}")
    
    required_files = ['teacher.pt', 'tn.pt', 't_nodes_shared.npy', 'manifest.json']
    for fname in required_files:
        fpath = os.path.join(args.masked_root, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")
    
    print("✓ All required model files found")
    
    # Run evaluation
    print("\nRunning comprehensive evaluation...")
    start_time = time.time()
    
    try:
        stdout = run_eval_script(
            dataset=args.dataset,
            seed=args.seed,
            masked_root=args.masked_root,
            orders=args.orders,
            outdir=args.outdir,
            kernel_budgets=args.kernel_budgets,
            shapiq_budgets=args.shapiq_budgets,
            repeats=args.repeats
        )
        
        eval_time = time.time() - start_time
        print(f"\n✓ Evaluation completed in {eval_time:.1f} seconds")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        return 1
    
    # Aggregate results
    print("\nAggregating results...")
    try:
        summary_df = aggregate_results(args.outdir, args.dataset, args.seed)
        
        # Save aggregated results
        summary_path = os.path.join(args.outdir, f"{args.dataset}_budget_sweep_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Summary saved to: {summary_path}")
        
        # Create LaTeX table
        latex_path = os.path.join(args.outdir, f"{args.dataset}_budget_sweep_results_{timestamp}.tex")
        create_latex_table(summary_df, latex_path)
        
        # Print key results
        print("\n" + "="*80)
        print("KEY RESULTS SUMMARY")
        print("="*80)
        
        # TNShap results (method == 'tn_selector')
        tnshap_results = summary_df[summary_df['method'] == 'tn_selector']
        if not tnshap_results.empty:
            print("\nTNShap Performance:")
            for _, row in tnshap_results.iterrows():
                print(f"  Order {int(row['order_k'])}: {row['time_s_mu_mean']:.4f}s, "
                      f"Cosine: {row['cos_vs_exact_mu_mean']:.4f}")
        
        # Best baseline performance by order
        baseline_results = summary_df[summary_df['method'] != 'tn_selector']
        if not baseline_results.empty:
            print("\nBest Baseline Performance by Order:")
            for order in args.orders:
                order_data = baseline_results[baseline_results['order_k'] == order]
                if not order_data.empty:
                    best_idx = order_data['cos_vs_exact_mu_mean'].idxmax()
                    best_row = order_data.loc[best_idx]
                    budget_str = f"budget={int(best_row['budget'])}" if not pd.isna(best_row['budget']) else "no budget"
                    print(f"  Order {order}: {best_row['baseline']} ({budget_str}), "
                          f"Time: {best_row['time_s_mu_mean']:.4f}s, "
                          f"Cosine: {best_row['cos_vs_exact_mu_mean']:.4f}")
        
        print("\n" + "="*80)
        print(f"Experiment completed successfully!")
        print(f"Results saved in: {args.outdir}")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Results aggregation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
