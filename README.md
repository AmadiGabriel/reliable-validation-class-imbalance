# Reliable Model Validation under Class Imbalance

This repository contains the source code, derived results, and
reproducibility materials associated with the study:

**Practical Guidelines for Permutation-Based Model Validation in Machine
Learning under Class Imbalance**

Authors: **Amadi Gabriel Udu** and **Andrea Lecchini-Visintini**

## Overview

The study examines model validation under simultaneous class imbalance,
absolute minority-class rarity, and high dimensionality. Particular
attention is given to information leakage arising from supervised feature
selection performed outside the appropriate cross-validation structure.

The final validation procedure combines:

- nested cross-validation;
- supervised model-based feature selection performed within training data;
- classifier hyperparameter optimisation within the inner loop;
- permutation testing of the complete validation procedure;
- training-only surrogate-positive augmentation;
- evaluation exclusively on the original target population.

Four ensemble classifiers are considered:

- AdaBoost;
- LightGBM;
- Random Forest;
- XGBoost.

Two prediction formulations are evaluated:

- **FP**: the original target classification problem;
- **FPs**: the same target problem with a fixed pool of surrogate-positive
  observations available during training only.

## Experimental configuration

The target dataset contains 1,811 observations:

- 1,756 majority-class observations;
- 55 minority-class observations.

The FPs formulation additionally uses 47 fixed surrogate-positive
observations during training. These observations are excluded from inner
validation and outer evaluation folds.

After removal of eight pre-identified high-VIF variables, 1,288 candidate
features remain. Supervised feature selection uses `SelectFromModel`, with a
fixed maximum of 50 selected features.

The final validation procedure uses:

- 5 outer stratified folds;
- 3 inner stratified folds;
- 90 independent label permutations for every classifier/problem pair.

This gives 8 observed-label analyses and 720 permutation analyses.

Only labels belonging to the original target dataset are permuted. The fixed
surrogate-positive training pool is not permuted.

The primary scalar performance statistic is the mean outer-fold AUC.

## Repository structure

```text
.
├── src/
│   └── aero_nested_validation.py
├── scripts/
│   ├── analyse_complete_results.py
│   ├── plot_all_classifier_figures.py
│   └── run_lgbm_permutations_local.sh
├── results/
│   ├── all_720_permutation_auc_values.csv
│   ├── complete_validation_summary.csv
│   └── paper_results_table.csv
├── figures/
│   └── publication figures
├── data/
│   └── README.md
├── docs/
├── environment.yml
├── requirements.txt
└── README.md

```

The `results/` directory contains the final derived results reported in the
paper. The complete fold-level and permutation output tree is archived
separately because it is not required for normal use of the repository.

## Data availability

The raw aero-engine dataset is maintained separately from this source-code
repository.

A separate Zenodo data record will be used for the dataset. File access may
be public or restricted depending on the applicable redistribution
permissions.

Researchers interested in obtaining access may contact the corresponding
author, subject to the permissions and restrictions applicable to the
original data.

The code, analysis workflow, derived summary results, and manuscript figures
are provided in this repository.

## Installation

A reference Conda environment is provided:

```bash
conda env create -f environment.yml
conda activate reliable-validation
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

The reference environment is based primarily on the environment used for the
ALICE high-performance computing experiments, with Matplotlib added for
analysis and figure generation.

## Running an experiment

The core program is:

```bash
python src/aero_nested_validation.py --help
```

For example, an observed-label Random Forest FP run can be started with:

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

A permutation run is obtained by changing `--run-type true` to
`--run-type perm` and supplying the required permutation seed.

The supplied local LightGBM runner executes all 90 permutations for both FP
and FPs:

```bash
scripts/run_lgbm_permutations_local.sh
```

The number of workers can be changed without editing the script:

```bash
N_JOBS=4 scripts/run_lgbm_permutations_local.sh
```

## Reproducing the final analyses

The full archived experiment outputs should first be extracted into:

```text
results_complete/
```

Then run:

```bash
python scripts/analyse_complete_results.py
```

Newly reproduced summary tables will be written to:

```text
results_reproduced/
```

Publication-ready summary tables are already available under `results/`.

Figures can be regenerated using:

```bash
python scripts/plot_all_classifier_figures.py
```

and will be written to:

```text
figures_reproduced/
```

The figures used in the manuscript are preserved under `figures/`.

## Computational resources

Large-scale permutation experiments for AdaBoost, Random Forest, and XGBoost
were executed using the University of Leicester ALICE High Performance
Computing Facility.

The LightGBM permutation experiments were completed on a local Apple Silicon
workstation.

See `docs/computational_environment.md` for software-environment details.

## Reproducibility note

Exact numerical replication can depend on the software environment, hardware,
parallel execution, and implementation details of the underlying machine
learning libraries. Random seeds and the principal software versions used for
the production experiments are documented where available.

## Citation

A `CITATION.cff` file accompanies the repository. A permanent Zenodo DOI will
be added following publication of version 1.0.0.

## Acknowledgement

This research used the ALICE High Performance Computing Facility at the
University of Leicester.

## Licence

The source code in this repository is released under the MIT License. See
`LICENSE` for details.

The raw aero-engine dataset is not included in this repository and is not
covered by the software licence. Data access and redistribution remain
subject to the permissions applicable to the original dataset.
