#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Create comprehensive LaTeX table showing ALL baselines with ALL budget sweeps.
Shows mean ± std for each baseline per order and budget combination.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_comprehensive_latex_table(df: pd.DataFrame) -> str:
    """Create a comprehensive LaTeX table showing all baselines with all budget sweeps."""
    
    latex_lines = []
    
    # Table header
    latex_lines.append(r"\begin{table}[ht]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Comprehensive Baseline Performance Analysis - All Methods and Budgets}")
    latex_lines.append(r"\label{tab:diabetes_comprehensive}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{")
    latex_lines.append(r"\begin{tabular}{llcccccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Baseline & Order & Budget & Targets & Time (s) & Cosine Sim. & MSE & $R^2$ \\")
    latex_lines.append(r"\midrule")
    
    # Sort by baseline, order, and budget
    df_sorted = df.sort_values(['baseline', 'order_k', 'budget'])
    
    current_baseline = None
    
    for _, row in df_sorted.iterrows():
        baseline = row['baseline']
        order = int(row['order_k'])
        budget = row['budget']
        
        # Add separator lines between baselines
        if current_baseline is not None and current_baseline != baseline:
            latex_lines.append(r"\midrule")
        current_baseline = baseline
        
        # Clean baseline name for LaTeX
        if baseline == 'TNShap':
            baseline_clean = r"\textbf{TN-SHAP}"
        elif baseline == 'KernelSHAPIQ (Reg SII)':
            baseline_clean = "KernelSHAPIQ"
        elif baseline == 'SamplingShapleyApproximation':
            baseline_clean = "SamplingShapley"
        elif baseline == 'PermutationSampling':
            baseline_clean = "PermutationSamp"
        elif baseline == 'KernelSHAPIQ (Reg STI)':
            baseline_clean = "KernelSHAPIQ-STI"
        else:
            baseline_clean = baseline.replace('_', r'\_')
        
        # Format budget
        if pd.isna(budget):
            budget_str = "--"
        else:
            budget_str = f"{int(budget)}"
        
        # Format numbers with mean ± std
        time_str = f"{row['time_s_mu_mean']:.4f} ± {row['time_s_mu_std']:.4f}"
        cosine_str = f"{row['cos_vs_exact_mu_mean']:.4f} ± {row['cos_vs_exact_mu_std']:.4f}"
        mse_str = f"{row['mse_vs_exact_mu_mean']:.6f} ± {row['mse_vs_exact_mu_std']:.6f}"
        r2_str = f"{row['r2_vs_exact_mu_mean']:.3f} ± {row['r2_vs_exact_mu_std']:.3f}"
        targets_str = f"{int(row['n_targets'])}"
        
        latex_lines.append(f"{baseline_clean} & $k={order}$ & {budget_str} & {targets_str} & {time_str} & {cosine_str} & {mse_str} & {r2_str} \\\\")
    
    # Table footer
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"}")
    latex_lines.append(r"\begin{tablenotes}")
    latex_lines.append(r"\footnotesize")
    latex_lines.append(r"\item Values shown as mean ± standard deviation across test instances.")
    latex_lines.append(r"\item Budget indicates number of model evaluations for approximate methods.")
    latex_lines.append(r"\item TN-SHAP entries show performance across different tensor network ranks.")
    latex_lines.append(r"\end{tablenotes}")
    latex_lines.append(r"\end{table}")
    
    return "\n".join(latex_lines)

def print_comprehensive_summary(df: pd.DataFrame):
    """Print a comprehensive summary of all baselines and budgets."""
    
    print("=" * 80)
    print("COMPREHENSIVE BASELINE ANALYSIS - ALL METHODS AND BUDGETS")
    print("=" * 80)
    
    # Group by order for better organization
    for order in sorted(df['order_k'].unique()):
        print(f"\n🎯 ORDER k={int(order)} ANALYSIS:")
        print("=" * 60)
        
        order_data = df[df['order_k'] == order].sort_values(['baseline', 'budget'])
        
        current_baseline = None
        for _, row in order_data.iterrows():
            baseline = row['baseline']
            budget = row['budget']
            
            if current_baseline != baseline:
                print(f"\n📊 {baseline.upper()}:")
                print("-" * 50)
                current_baseline = baseline
            
            budget_str = f"Budget {int(budget)}" if not pd.isna(budget) else "No Budget"
            print(f"  💡 {budget_str}:")
            print(f"    Targets: {int(row['n_targets'])} test instances")
            print(f"    Time: {row['time_s_mu_mean']:.6f} ± {row['time_s_mu_std']:.6f} seconds")
            print(f"    Cosine Sim: {row['cos_vs_exact_mu_mean']:.6f} ± {row['cos_vs_exact_mu_std']:.6f}")
            print(f"    MSE: {row['mse_vs_exact_mu_mean']:.8f} ± {row['mse_vs_exact_mu_std']:.8f}")
            print(f"    R²: {row['r2_vs_exact_mu_mean']:.6f} ± {row['r2_vs_exact_mu_std']:.6f}")

def analyze_budget_scaling(df: pd.DataFrame):
    """Analyze how performance scales with budget for each baseline."""
    
    print("\n" + "=" * 80)
    print("BUDGET SCALING ANALYSIS")
    print("=" * 80)
    
    for baseline in sorted(df['baseline'].unique()):
        baseline_data = df[df['baseline'] == baseline]
        
        # Skip if only one budget configuration
        if len(baseline_data['budget'].dropna().unique()) <= 1:
            continue
            
        print(f"\n📈 {baseline.upper()} BUDGET SCALING:")
        print("-" * 60)
        
        for order in sorted(baseline_data['order_k'].unique()):
            order_data = baseline_data[baseline_data['order_k'] == order].sort_values('budget')
            
            if len(order_data) <= 1:
                continue
                
            print(f"\n  Order k={int(order)}:")
            print(f"    {'Budget':<8} {'Time (s)':<12} {'Cosine Sim':<12} {'MSE':<15} {'R²':<12}")
            print("    " + "-" * 65)
            
            for _, row in order_data.iterrows():
                budget_str = f"{int(row['budget'])}" if not pd.isna(row['budget']) else "N/A"
                print(f"    {budget_str:<8} {row['time_s_mu_mean']:<12.6f} {row['cos_vs_exact_mu_mean']:<12.6f} "
                      f"{row['mse_vs_exact_mu_mean']:<15.8f} {row['r2_vs_exact_mu_mean']:<12.6f}")

def main():
    """Main function to create comprehensive analysis."""
    
    # Load the aggregated summary
    data_path = "results/diabetes_budget_sweep_results/diabetes_seed2711/diabetes_agg_summary.csv"
    
    print("Loading aggregated diabetes results...")
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} baseline-order-budget combinations")
    print(f"Baselines: {sorted(df['baseline'].unique())}")
    print(f"Orders: {sorted(df['order_k'].unique())}")
    print(f"Budget ranges: {sorted(df['budget'].dropna().unique())}")
    
    # Print comprehensive summary
    print_comprehensive_summary(df)
    
    # Budget scaling analysis
    analyze_budget_scaling(df)
    
    # Create LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    
    latex_table = create_comprehensive_latex_table(df)
    print(latex_table)
    
    # Save results
    output_dir = Path("results/diabetes_budget_sweep_results/diabetes_seed2711")
    
    # Save LaTeX table
    with open(output_dir / "comprehensive_baseline_table.tex", "w") as f:
        f.write(latex_table)
    
    # Save detailed CSV
    df_detailed = df.sort_values(['baseline', 'order_k', 'budget'])
    df_detailed.to_csv(output_dir / "comprehensive_baseline_analysis.csv", index=False)
    
    print(f"\n" + "=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"- comprehensive_baseline_table.tex")
    print(f"- comprehensive_baseline_analysis.csv")
    
    # Performance ranking summary
    print("\n" + "=" * 80)
    print("TOP PERFORMERS BY METRIC")
    print("=" * 80)
    
    for order in sorted(df['order_k'].unique()):
        order_data = df[df['order_k'] == order]
        
        print(f"\n🏆 ORDER k={int(order)} CHAMPIONS:")
        print("-" * 50)
        
        # Best in each metric
        fastest = order_data.loc[order_data['time_s_mu_mean'].idxmin()]
        best_cosine = order_data.loc[order_data['cos_vs_exact_mu_mean'].idxmax()]
        lowest_mse = order_data.loc[order_data['mse_vs_exact_mu_mean'].idxmin()]
        best_r2 = order_data.loc[order_data['r2_vs_exact_mu_mean'].idxmax()]
        
        budget_str = lambda x: f" (Budget {int(x)})" if not pd.isna(x) else ""
        
        print(f"🥇 Fastest: {fastest['baseline']}{budget_str(fastest['budget'])} - {fastest['time_s_mu_mean']:.4f}s")
        print(f"🥇 Best Cosine: {best_cosine['baseline']}{budget_str(best_cosine['budget'])} - {best_cosine['cos_vs_exact_mu_mean']:.4f}")
        print(f"🥇 Lowest MSE: {lowest_mse['baseline']}{budget_str(lowest_mse['budget'])} - {lowest_mse['mse_vs_exact_mu_mean']:.6f}")
        print(f"🥇 Best R²: {best_r2['baseline']}{budget_str(best_r2['budget'])} - {best_r2['r2_vs_exact_mu_mean']:.4f}")

if __name__ == "__main__":
    main()
