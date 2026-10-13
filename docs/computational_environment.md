# Computational environments

The experiments and post-processing associated with this study were
performed across two computational environments.

## ALICE high-performance computing environment

The large-scale AdaBoost, Random Forest, and XGBoost experiments were
executed using the University of Leicester ALICE High Performance
Computing Facility.

The recorded `aero-ml` environment was:

| Package | Version |
|---|---:|
| Python | 3.11.17 |
| NumPy | 2.4.6 |
| pandas | 3.0.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| LightGBM | 4.7.0 |
| XGBoost | 3.4.2 |

Matplotlib was not installed in the ALICE model-computation environment.

## Local analysis and plotting environment

Final result aggregation and manuscript figure generation were performed
on a local Apple Silicon workstation.

The recorded analysis environment was:

| Package | Version |
|---|---:|
| Python | 3.11.13 |
| NumPy | 2.3.1 |
| pandas | 2.3.1 |
| scikit-learn | 1.7.1 |
| joblib | 1.5.1 |
| Matplotlib | 3.10.0 |

The LightGBM permutation experiments were also executed on the local
workstation using the same production validation script. The exact
historical LightGBM package version used for those completed local runs
was not preserved.

The repository therefore provides a common reference environment in
`environment.yml` and `requirements.txt` for executing the complete
workflow.

## Parallel execution

The local LightGBM permutation runner used 12 joblib workers. Native
threading was restricted to one thread per worker using:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

The number of joblib workers can be changed through the `N_JOBS`
environment variable in:

```text
scripts/run_lgbm_permutations_local.sh
```

## Computational reproducibility

Exact numerical replication can be affected by hardware, library versions,
parallel execution, and changes in the implementations of machine-learning
libraries. Random seeds and the principal software versions used in the
production experiments are documented where they were preserved.
