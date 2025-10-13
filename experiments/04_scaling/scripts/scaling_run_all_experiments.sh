#!/usr/bin/env bash
set -euo pipefail
# conda activate concepts
# Make local modules importable (tntree_model.py, feature_mapped_tn.py next to scripts)
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

SEED=2711
OUT_BASE=./out_hd_sweep
RANK=64  # Increased rank for better capacity

WITH_SHAPIQ=1
SHAPIQ_APPROX=regression
SHAPIQ_BUDGET=5000  # Increased budget for higher dimensions

mkdir -p "${OUT_BASE}"
MASTER="${OUT_BASE}/master_merged.csv"
: > "${MASTER}"

for D in $(seq 30 10 50); do
  echo "================ d=${D} ================"

  GT_PREFIX="hd${D}"
  GT_DIR="${OUT_BASE}/gt_d${D}"
  mkdir -p "${GT_DIR}"

  echo "[1/3] Generator @ d=${D}"
  if [[ "${WITH_SHAPIQ}" -eq 1 ]]; then
    # Use reduced sparsity for higher dimensions to make the problem more learnable
    if [[ ${D} -ge 50 ]]; then
      N1=6; N2=8; N3=6  # Reduced complexity for d>=50
    else
      N1=8; N2=12; N3=10  # Default complexity
    fi
    python3 groundtruth_multilinear_generator.py \
      --d ${D} \
      --seed ${SEED} \
      --n1 ${N1} --n2 ${N2} --n3 ${N3} \
      --baseline shapiq_regression \
      --budget ${SHAPIQ_BUDGET} \
      --outdir "${GT_DIR}" \
      --prefix "${GT_PREFIX}"
  else
    python3 groundtruth_multilinear_generator.py \
      --d ${D} \
      --seed ${SEED} \
      --outdir "${GT_DIR}" \
      --prefix "${GT_PREFIX}"
  fi

  GT_RESULTS="${GT_DIR}/${GT_PREFIX}_results.csv"

  echo "[2/3a] Student (random) @ d=${D}"
  STU_R_DIR="${OUT_BASE}/student_random_d${D}"
  mkdir -p "${STU_R_DIR}"
  if [[ "${WITH_SHAPIQ}" -eq 1 ]]; then
    # Increase training samples for higher dimensions
    if [[ ${D} -ge 50 ]]; then
      N_RANDOM=10000  # More samples for d>=50
    else
      N_RANDOM=5000   # Default samples
    fi
    python3 student_vs_generator.py \
      --gt-root "${GT_DIR}" \
      --outdir "${STU_R_DIR}" \
      --seed ${SEED} \
      --rank ${RANK} \
      --strategy random \
      --n-random ${N_RANDOM} \
      --with-shapiq \
      --shapiq-approximator ${SHAPIQ_APPROX} \
      --shapiq-budget ${SHAPIQ_BUDGET}
  else
    python3 student_vs_generator.py \
      --gt-root "${GT_DIR}" \
      --outdir "${STU_R_DIR}" \
      --seed ${SEED} \
      --rank ${RANK} \
      --strategy random \
      --n-random 5000
  fi
  STU_R_CSV="${STU_R_DIR}/student_results.csv"

  RUN_MASKED=0
  if [[ ${D} -le 20 ]]; then RUN_MASKED=1; fi

  if [[ "${RUN_MASKED}" -eq 1 ]]; then
    echo "[2/3b] Student (masked) @ d=${D}"
    STU_M_DIR="${OUT_BASE}/student_masked_d${D}"
    mkdir -p "${STU_M_DIR}"
    if [[ "${WITH_SHAPIQ}" -eq 1 ]]; then
      python3 student_vs_generator.py \
        --gt-root "${GT_DIR}" \
        --outdir "${STU_M_DIR}" \
        --seed ${SEED} \
        --rank ${RANK} \
        --strategy masked \
        --with-shapiq \
        --shapiq-approximator ${SHAPIQ_APPROX} \
        --shapiq-budget ${SHAPIQ_BUDGET}
    else
      python3 student_vs_generator.py \
        --gt-root "${GT_DIR}" \
        --outdir "${STU_M_DIR}" \
        --seed ${SEED} \
        --rank ${RANK} \
        --strategy masked
    fi
    STU_M_CSV="${STU_M_DIR}/student_results.csv"
  else
    STU_M_CSV=""
  fi

  echo "[3/3] Merge CSVs for d=${D}"
  MERGED_D="${OUT_BASE}/merged_d${D}.csv"
  python3 - <<PY
import os, pandas as pd
gt = pd.read_csv("${GT_RESULTS}") if os.path.isfile("${GT_RESULTS}") else pd.DataFrame()
sr = pd.read_csv("${STU_R_CSV}") if os.path.isfile("${STU_R_CSV}") else pd.DataFrame()
sm = pd.read_csv("${STU_M_CSV}") if "${STU_M_CSV}" and os.path.isfile("${STU_M_CSV}") else pd.DataFrame()
dfs=[]
for name,df in [("generator",gt),("student_random",sr),("student_masked",sm)]:
    if not df.empty:
        df["source"]=name
        dfs.append(df)
if dfs:
    md = pd.concat(dfs, ignore_index=True)
    md.to_csv("${MERGED_D}", index=False)
    if os.path.isfile("${MASTER}") and os.path.getsize("${MASTER}")>0:
        mm = pd.read_csv("${MASTER}")
        pd.concat([mm, md], ignore_index=True).to_csv("${MASTER}", index=False)
    else:
        md.to_csv("${MASTER}", index=False)
    print("[OK] merged ->", "${MERGED_D}")
else:
    print("[WARN] nothing to merge for d=${D}")
PY

done

echo
echo "[DONE] Sweep complete."
echo "Master CSV: ${MASTER}"
