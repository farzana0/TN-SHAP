# TN-SHAP: Tractable Shapley Values and Interactions via Tensor Networks

Reference implementation and experiments for the paper:

> **Tractable Shapley Values and Interactions via Tensor Networks**
> Farzaneh Heidari, Chao Li, Guillaume Rabusseau.
> *Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS) 2026.*

TN-SHAP replaces the `O(2ⁿ)` coalition enumeration behind Shapley values and
Shapley-style interaction indices with a *few-evaluation* scheme on a
tensor-network (TN) surrogate. The predictor's local behavior is represented as
a factorized multilinear map, so coalitional quantities become **linear probes**
of a coefficient tensor. Order-1 (single-feature) and order-2 (pairwise)
attributions cost `O(n·poly(χ) + n²)`, where `χ` is the TN's maximal cut rank.

Key idea (Section 3): augment each lifted feature with a thin diagonal
**selector** `Sᵣ(t) = Diag(t, 1)`. Evaluating the surrogate along the diagonal
`(t, …, t)` aggregates all size-`s` coalitions into the coefficient of `tˢ`, so
the `2ⁿ` coalition queries collapse to an `n`-point polynomial interpolation
(a single Vandermonde solve).

---

## Repository structure

```
TN-SHAP/
├── demo_tnshap.py                     # Quick-start: train surrogate, compute SVs, compare to exact
├── src/                               # Core TN-SHAP implementation
│   ├── tntree_model.py                # Binary tensor-tree surrogate
│   ├── feature_mapped_tn.py           # Per-feature learned feature lifts ϕ(x)
│   ├── shapley_computation.py         # Selector probes + Vandermonde interpolation
│   ├── tnshap_exact_enum_verify.py    # Exactness check vs O(2ⁿ) enumeration
│   ├── tnshap_multilinear_selftest.py # Multilinearity self-test
│   ├── utils/                         # Shared Shapley utilities
│   ├── shapiq_benchmark/              # Comparisons against the shapiq library
│   └── extra_experiments/             # Sparse-polynomial teacher/student studies
├── experiments/                       # Paper experiments, organized by section
│   ├── UCI/                           # §5.2  Real-world UCI teachers (Diabetes, Concrete, Energy)
│   ├── 02_higher_order_ablations/     # §5.2 / App. F.4  Feature-map dimensionality study
│   ├── 03_synthetic_experiments/      # §5.3  Rank ablations & training dynamics (+ figures)
│   ├── 04_scaling/                    # §5.1  Synthetic multilinear runtime (d = 10…50)
│   └── 05_highdim_scaling/            # §5.4  High-dimensional top-k recovery (TT/MPS, D = 50,100)
├── figures/                           # Result figures reproduced from the paper (PNG/PDF)
├── docs/                              # Installation / quick-start / methodology notes
├── tests/                             # Correctness tests (t=1 oracle, etc.)
└── gpt_explain/                       # Exploratory LLM token-attribution application (not in paper)
```

---

## Installation

```bash
git clone https://github.com/farzana0/TN-SHAP.git
cd TN-SHAP

# Option 1 — conda (recommended)
conda env create -f environment.yml
conda activate tnshap
pip install -e .

# Option 2 — pip
pip install -r requirements.txt
pip install -e .
```

The high-dimensional scaling experiments (`experiments/05_highdim_scaling/`)
additionally depend on `opt_einsum` and the bundled `torchmps` library, which
are installed from that directory's `requirements.txt`.

## Quick start

```bash
python demo_tnshap.py
```

This trains a tensor-network surrogate, computes TN-SHAP attributions, and (on
synthetic multilinear data) checks them against exact Shapley values.

Minimal API:

```python
import torch
from src import make_feature_mapped_tn
from src.utils import compute_shapley_values_tnshap

model = make_feature_mapped_tn(d_in=10, fmap_out_dim=4, ranks=6, seed=42)
x = torch.randn(1, 10)
phi = compute_shapley_values_tnshap(model, x)   # order-1 Shapley values
```

---

## Reproducing the paper

Each experiment folder contains its own `README.md` with exact commands. The
mapping to paper sections:

| Section / Table·Figure | Folder | What it produces |
|---|---|---|
| §5.1, Table 2 | `experiments/04_scaling/` | Per-instance runtime on synthetic multilinear functions, `d ∈ {10,…,50}`, vs. KernelSHAP-IQ |
| §5.2, Tables 3, 8–10; Fig. 3 | `experiments/UCI/` | TN-SHAP vs. sampling baselines on UCI MLP teachers (Diabetes, Concrete, Energy), orders k=1,2,3 |
| App. F.4, Table 11 | `experiments/02_higher_order_ablations/` | Effect of feature-map width `m_fmap ∈ {1,2,4,8}` on higher-order recovery |
| §5.3, Tables 4, 7; Figs. 4, 5 | `experiments/03_synthetic_experiments/` | Teacher–student rank sweep, training dynamics, rank-ablation heatmap |
| §5.4, Table 5 | `experiments/05_highdim_scaling/` | Top-k feature recovery for poly5 / poly10 / sqexp teachers at `D ∈ {50,100}` with a TT/MPS surrogate |

Example commands:

```bash
# §5.2 — UCI Diabetes, orders 1–3
python experiments/UCI/scripts/uci_evaluate_tnshap_vs_baselines.py --dataset diabetes --orders 1 2 3

# §5.3 — synthetic rank sweep
python experiments/03_synthetic_experiments/scripts/synthetic_rank_sweep_basic.py --seed 42

# §5.4 — high-dimensional scaling pipeline (generate teacher → train MPS → evaluate TN-SHAP)
cd experiments/05_highdim_scaling && bash run_all_experiments.sh
```

### Figures

`figures/` holds the result plots reproduced from the camera-ready paper (no
LaTeX sources are included):

- `fig3_concrete_local_eval_k2.pdf`, `fig3_concrete_local_eval_k3.pdf`, `fig3_legend.pdf` — runtime vs. cosine similarity on Concrete (Figure 3)
- `fig4_training_dynamics.png` — R² vs. epoch for the student fit and SII orders (Figure 4)
- `fig5_rank_ablation_heatmap.png` — safe-R² across student ranks, GT TN rank 14 (Figure 5)

---

## `gpt_explain/` (exploratory, not part of the paper)

`gpt_explain/` is an **exploratory application** of the TN-SHAP machinery to
LLM token attribution: it lifts DistilGPT-2 token embeddings, fits a
tensor-network surrogate over a token sentence, and computes Shapley
interactions over tokens. It is **not** part of the AISTATS paper's claims or
experiments and is included only as a worked example of applying TN surrogates
beyond tabular data. Only the code (scripts + README) is shipped here; the large
generated datasets, embeddings, plots, and checkpoints are omitted — see
`gpt_explain/README.md` for how to regenerate them.

## Acknowledgements

The high-dimensional scaling experiments build on the
[TorchMPS](https://github.com/jemisjoky/TorchMPS) library (Jacob Miller, MIT
License); its sources are vendored under `experiments/05_highdim_scaling/torchmps/`
with the original license preserved at `experiments/05_highdim_scaling/LICENSE`.

## License

This project is released under the MIT License — see [LICENSE](LICENSE).
