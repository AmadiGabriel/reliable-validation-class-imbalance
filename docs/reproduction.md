# Reproduction guide

This document describes how to reproduce the validation experiments and
derived results associated with the study.

## 1. Create the software environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate reliable-validation
```

Alternatively, using pip with Python 3.11:

```bash
python -m pip install -r requirements.txt
```

## 2. Obtain the aero-engine data

The raw aero-engine data are maintained separately from this Git repository.

The analysis expects ten CSV files:

```text
aeroeng1.csv
aeroeng2.csv
aeroeng3.csv
aeroeng4.csv
aeroeng5.csv
aeroeng6.csv
aeroeng7.csv
aeroeng8.csv
aeroeng9.csv
aeroeng10.csv
```

Place them in:

```text
engine_data/
```

See `data/README.md` for the data-access statement.

## 3. Core experiment program

The principal validation program is:

```text
src/aero_nested_validation.py
```

Display the available options with:

```bash
python src/aero_nested_validation.py --help
```

The final experiments use:

- 5 outer stratified folds;
- 3 inner stratified folds;
- a fixed maximum of 50 selected features;
- 90 permutations per classifier/problem combination.

The prediction problems are `FP` and `FPs`.

The classifiers are:

- `ada` -- AdaBoost;
- `lgbm` -- LightGBM;
- `rf` -- Random Forest;
- `xgb` -- XGBoost.

## 4. Observed-label run

For example:

```bash
python src/aero_nested_validation.py \
    --problem FP \
    --classifier rf \
    --run-type true \
    --seed 0 \
    --data-dir engine_data \
    --results-dir results_reproduced \
    --outer-folds 5 \
    --inner-folds 3 \
    --max-features 50
```

## 5. Permutation run

For example:

```bash
python src/aero_nested_validation.py \
    --problem FP \
    --classifier rf \
    --run-type perm \
    --seed 1 \
    --data-dir engine_data \
    --results-dir results_reproduced \
    --outer-folds 5 \
    --inner-folds 3 \
    --max-features 50
```

Permutation seeds 1 through 90 were used for each classifier/problem
combination.

Only labels of the original target observations are permuted.

For FPs, the fixed surrogate-positive observations are not permuted. They are
added only to training subsets and are excluded from inner-validation and
outer-test folds.

## 6. LightGBM permutation runner

All 90 LightGBM permutations for both FP and FPs can be run with:

```bash
scripts/run_lgbm_permutations_local.sh
```

The number of workers can be changed without editing the script:

```bash
N_JOBS=4 scripts/run_lgbm_permutations_local.sh
```

Alternative data and output locations can also be supplied:

```bash
DATA_DIR=/path/to/engine_data \
RESULTS_DIR=/path/to/results \
N_JOBS=4 \
scripts/run_lgbm_permutations_local.sh
```

## 7. Complete production outputs

The full fold-level production outputs are distributed separately from this
Git repository.

To regenerate the summary analysis and figures, extract that archive so that
the repository contains:

```text
results_complete/
```

## 8. Reproduce the summary results

Run:

```bash
python scripts/analyse_complete_results.py
```

The complete analysis contains:

```text
4 classifiers x 2 problems x 90 permutations = 720 permutation runs
```

plus the eight corresponding observed-label runs.

Publication-ready derived outputs are supplied under `results/`:

```text
all_720_permutation_auc_values.csv
complete_validation_summary.csv
paper_results_table.csv
```

## 9. Reproduce the figures

With the complete experimental outputs available under `results_complete/`,
run:

```bash
python scripts/plot_all_classifier_figures.py
```

Newly generated figures are written to `figures_reproduced/`.

The manuscript-ready figures are preserved under `figures/`.

## 10. Performance statistic

The primary scalar performance statistic is the mean AUC across the five
outer evaluation folds.

Pooled out-of-fold AUC is retained as a secondary descriptive measure and is
used for the ROC visualisations.

## 11. Computing resources

Large-scale AdaBoost, Random Forest, and XGBoost experiments were executed
using the University of Leicester ALICE High Performance Computing Facility.

The LightGBM permutation experiments were completed on a local Apple Silicon
workstation.

See `docs/computational_environment.md` for the recorded software
environments.
