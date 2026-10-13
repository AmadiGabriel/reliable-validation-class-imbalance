#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-$ROOT/engine_data}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results_reproduced}"
N_JOBS="${N_JOBS:-12}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

for PROBLEM in FP FPs; do
    for SEED in $(seq 1 90); do
        echo "=============================================================="
        echo "LightGBM permutation: problem=${PROBLEM}, seed=${SEED}/90"
        echo "=============================================================="

        python "$ROOT/src/aero_nested_validation.py" \
            --problem "$PROBLEM" \
            --classifier lgbm \
            --run-type perm \
            --seed "$SEED" \
            --data-dir "$DATA_DIR" \
            --results-dir "$RESULTS_DIR" \
            --outer-folds 5 \
            --inner-folds 3 \
            --max-features 50 \
            --n-jobs "$N_JOBS" \
            --skip-existing
    done
done
