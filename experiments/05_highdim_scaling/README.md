# §5.4 — High-dimensional Top-k Recovery

This folder reproduces the high-dimensional scaling experiments (paper Section
5.4, Table 5): TN-SHAP with a **tensor-train / MPS** surrogate on synthetic
teachers where only ~10% of the `D` input dimensions are relevant, at
`D ∈ {50, 100}`, under a linear-cost query budget.

Three teacher families are evaluated:

- `poly5`  — degree-5 polynomial
- `poly10` — degree-10 polynomial
- `sqexp`  — squared-exponential

We report Top-k significant-feature recovery accuracy, surrogate fidelity (R²),
surrogate training time, and post-fit attribution (eval) time.

## Pipeline

`run_all_experiments.sh` runs the full generate → train → evaluate pipeline for
each `(task, D)`:

```bash
bash run_all_experiments.sh
```

which for each setting calls, in order:

1. `poly_teacher.py` — generate the teacher and (train/test) data
2. `train_mps_paths.py` — fit the TT/MPS surrogate from structured probes
3. `tnshap_vandermonde_compare_sparse_poly.py` — TN-SHAP attribution + comparison

For the 100-seed runs used to produce the mean±std in the paper, use:

```bash
bash run_sparse_poly_100_seeds.sh     # poly5 / poly10
bash run_sqexp_100_seeds.sh           # sqexp
python aggregate_sparse_poly_all.py   # aggregate poly results
python aggregate_sqexp_stats.py       # aggregate sqexp results
```

Baselines (TreeSHAP / sampling / ProxySPEX) used in the related-work comparison
live in `local_*treeshap*.py`, `local_mc_treeshap_poly_sqexp.py`, and
`proxyspex/`.

## Attribution

The TT/MPS surrogate and the selector-probe machinery are built on the
[TorchMPS](https://github.com/jemisjoky/TorchMPS) library (Jacob Miller, MIT
License), vendored under `torchmps/`. The upstream library README is preserved
as `TORCHMPS_README.md`, and the original license is at `../05_highdim_scaling/LICENSE`.
