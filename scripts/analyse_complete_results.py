#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

# Full experiment outputs are not tracked in Git.
# Extract the archived Zenodo output bundle here when reproducing
# the complete analysis.
RESULTS_DIR = REPO_ROOT / "results_complete"

# Newly reproduced summary tables are written separately from the
# publication-ready tables committed under results/.
OUTPUT_DIR = REPO_ROOT / "results_reproduced"
EXPECTED_PERMUTATIONS = 90

PROBLEMS = ["FP", "FPs"]
CLASSIFIERS = ["ada", "lgbm", "rf", "xgb"]

CLASSIFIER_DISPLAY = {
    "ada": "AdaBoost",
    "lgbm": "LightGBM",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_true_summary(problem, classifier):
    folder = RESULTS_DIR / problem / classifier / "true"

    files = sorted(
        folder.glob(
            f"summary_{problem}_{classifier}_true_seed*.json"
        )
    )

    if len(files) != 1:
        raise RuntimeError(
            f"{problem}/{classifier}: expected exactly 1 true-label "
            f"summary, found {len(files)}"
        )

    return load_json(files[0]), files[0]


def get_permutation_summaries(problem, classifier):
    folder = RESULTS_DIR / problem / classifier / "perm"

    files = sorted(
        folder.glob(
            f"summary_{problem}_{classifier}_perm_seed*.json"
        )
    )

    if len(files) != EXPECTED_PERMUTATIONS:
        raise RuntimeError(
            f"{problem}/{classifier}: expected "
            f"{EXPECTED_PERMUTATIONS} permutation summaries, "
            f"found {len(files)}"
        )

    rows = []

    for path in files:
        row = load_json(path)
        row["_file"] = str(path)
        rows.append(row)

    return rows


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_rows = []
    permutation_detail_rows = []

    print("=" * 88)
    print("FINAL COMPLETE VALIDATION ANALYSIS")
    print("=" * 88)

    for problem in PROBLEMS:

        for classifier in CLASSIFIERS:

            print(
                f"\nProcessing "
                f"{problem} / {CLASSIFIER_DISPLAY[classifier]}"
            )

            # ------------------------------------------------
            # True-label result
            # ------------------------------------------------
            true_summary, true_path = get_true_summary(
                problem,
                classifier
            )

            observed_auc = float(
                true_summary["mean_outer_auc"]
            )

            observed_sd = float(
                true_summary["std_outer_auc"]
            )

            pooled_oof_auc = float(
                true_summary["pooled_outer_auc"]
            )

            # ------------------------------------------------
            # Permutation results
            # ------------------------------------------------
            perm_rows = get_permutation_summaries(
                problem,
                classifier
            )

            perm_aucs = np.array(
                [
                    float(row["mean_outer_auc"])
                    for row in perm_rows
                ],
                dtype=float
            )

            perm_pooled_aucs = np.array(
                [
                    float(row["pooled_outer_auc"])
                    for row in perm_rows
                ],
                dtype=float
            )

            # ------------------------------------------------
            # Verify permutation seeds
            # ------------------------------------------------
            perm_seeds = []

            for row in perm_rows:

                seed = row.get("perm_seed")

                if seed is None:
                    seed = row.get("seed")

                perm_seeds.append(int(seed))

            unique_seeds = sorted(
                set(perm_seeds)
            )

            if len(unique_seeds) != EXPECTED_PERMUTATIONS:
                raise RuntimeError(
                    f"{problem}/{classifier}: expected "
                    f"{EXPECTED_PERMUTATIONS} unique permutation seeds, "
                    f"found {len(unique_seeds)}"
                )

            # ------------------------------------------------
            # Null-distribution statistics
            # ------------------------------------------------
            perm_mean = float(
                np.mean(perm_aucs)
            )

            perm_sd = float(
                np.std(
                    perm_aucs,
                    ddof=1
                )
            )

            perm_min = float(
                np.min(perm_aucs)
            )

            perm_max = float(
                np.max(perm_aucs)
            )

            perm_q025 = float(
                np.quantile(
                    perm_aucs,
                    0.025
                )
            )

            perm_q05 = float(
                np.quantile(
                    perm_aucs,
                    0.05
                )
            )

            perm_q50 = float(
                np.quantile(
                    perm_aucs,
                    0.50
                )
            )

            perm_q95 = float(
                np.quantile(
                    perm_aucs,
                    0.95
                )
            )

            perm_q975 = float(
                np.quantile(
                    perm_aucs,
                    0.975
                )
            )

            # ------------------------------------------------
            # Permutation comparison with observed result
            # ------------------------------------------------
            exceedances = int(
                np.sum(
                    perm_aucs >= observed_auc
                )
            )

            empirical_p = float(
                (exceedances + 1)
                /
                (len(perm_aucs) + 1)
            )

            observed_exceeds_all = bool(
                observed_auc > perm_max
            )

            auc_difference_from_null_mean = float(
                observed_auc - perm_mean
            )

            # ------------------------------------------------
            # Save one row per permutation
            # ------------------------------------------------
            for row, auc, pooled_auc, seed in zip(
                perm_rows,
                perm_aucs,
                perm_pooled_aucs,
                perm_seeds,
            ):

                permutation_detail_rows.append(
                    {
                        "problem": problem,
                        "classifier": classifier,
                        "classifier_name":
                            CLASSIFIER_DISPLAY[classifier],
                        "perm_seed": seed,
                        "mean_outer_auc": auc,
                        "pooled_oof_auc": pooled_auc,
                        "summary_file": row["_file"],
                    }
                )

            # ------------------------------------------------
            # Main summary row
            # ------------------------------------------------
            summary_rows.append(
                {
                    "problem": problem,

                    "classifier": classifier,

                    "classifier_name":
                        CLASSIFIER_DISPLAY[classifier],

                    "observed_mean_outer_auc":
                        observed_auc,

                    "observed_outer_auc_sd":
                        observed_sd,

                    "observed_pooled_oof_auc":
                        pooled_oof_auc,

                    "n_permutations":
                        len(perm_aucs),

                    "permutation_mean_auc":
                        perm_mean,

                    "permutation_sd_auc":
                        perm_sd,

                    "permutation_min_auc":
                        perm_min,

                    "permutation_q025_auc":
                        perm_q025,

                    "permutation_q05_auc":
                        perm_q05,

                    "permutation_median_auc":
                        perm_q50,

                    "permutation_q95_auc":
                        perm_q95,

                    "permutation_q975_auc":
                        perm_q975,

                    "permutation_max_auc":
                        perm_max,

                    "observed_minus_permutation_mean":
                        auc_difference_from_null_mean,

                    "permutation_exceedances_of_observed":
                        exceedances,

                    "empirical_p_plus1":
                        empirical_p,

                    "observed_exceeds_all_permutations":
                        observed_exceeds_all,

                    "true_summary_file":
                        str(true_path),
                }
            )

            # ------------------------------------------------
            # Console report
            # ------------------------------------------------
            print(
                f"  Observed mean outer AUC : "
                f"{observed_auc:.4f}"
            )

            print(
                f"  Observed outer-fold SD  : "
                f"{observed_sd:.4f}"
            )

            print(
                f"  Pooled OOF AUC          : "
                f"{pooled_oof_auc:.4f}"
            )

            print(
                f"  Permutation mean        : "
                f"{perm_mean:.4f}"
            )

            print(
                f"  Permutation SD          : "
                f"{perm_sd:.4f}"
            )

            print(
                f"  Permutation range       : "
                f"{perm_min:.4f} -- "
                f"{perm_max:.4f}"
            )

            print(
                f"  Exceedances             : "
                f"{exceedances}/{len(perm_aucs)}"
            )

            print(
                f"  Empirical p (+1)        : "
                f"{empirical_p:.4f}"
            )

            print(
                f"  Observed > all perms    : "
                f"{observed_exceeds_all}"
            )

    # ========================================================
    # CREATE DATAFRAMES
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    permutation_df = pd.DataFrame(
        permutation_detail_rows
    )

    # Keep desired ordering
    summary_df["problem"] = pd.Categorical(
        summary_df["problem"],
        categories=PROBLEMS,
        ordered=True
    )

    summary_df["classifier"] = pd.Categorical(
        summary_df["classifier"],
        categories=CLASSIFIERS,
        ordered=True
    )

    summary_df = summary_df.sort_values(
        [
            "problem",
            "classifier"
        ]
    ).reset_index(drop=True)

    permutation_df["problem"] = pd.Categorical(
        permutation_df["problem"],
        categories=PROBLEMS,
        ordered=True
    )

    permutation_df["classifier"] = pd.Categorical(
        permutation_df["classifier"],
        categories=CLASSIFIERS,
        ordered=True
    )

    permutation_df = permutation_df.sort_values(
        [
            "problem",
            "classifier",
            "perm_seed"
        ]
    ).reset_index(drop=True)

    # ========================================================
    # SAVE COMPLETE OUTPUTS
    # ========================================================

    summary_path = (
        OUTPUT_DIR
        / "complete_validation_summary.csv"
    )

    permutation_path = (
        OUTPUT_DIR
        / "all_720_permutation_auc_values.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    permutation_df.to_csv(
        permutation_path,
        index=False
    )

    # ========================================================
    # CREATE LEAN PAPER TABLE
    # ========================================================

    paper_table = summary_df[
        [
            "problem",
            "classifier_name",
            "observed_mean_outer_auc",
            "observed_outer_auc_sd",
            "permutation_mean_auc",
            "permutation_sd_auc",
            "permutation_max_auc",
            "permutation_exceedances_of_observed",
            "empirical_p_plus1",
        ]
    ].copy()

    paper_table = paper_table.rename(
        columns={
            "problem":
                "Problem",

            "classifier_name":
                "Classifier",

            "observed_mean_outer_auc":
                "Observed mean outer AUC",

            "observed_outer_auc_sd":
                "Observed SD",

            "permutation_mean_auc":
                "Permutation mean AUC",

            "permutation_sd_auc":
                "Permutation SD",

            "permutation_max_auc":
                "Maximum permutation AUC",

            "permutation_exceedances_of_observed":
                "Exceedances",

            "empirical_p_plus1":
                "Empirical p",
        }
    )

    paper_table_path = (
        OUTPUT_DIR
        / "paper_results_table.csv"
    )

    paper_table.to_csv(
        paper_table_path,
        index=False
    )

    # ========================================================
    # IDENTIFY BEST OBSERVED RESULT
    # ========================================================

    best_idx = summary_df[
        "observed_mean_outer_auc"
    ].idxmax()

    best = summary_df.loc[
        best_idx
    ]

    best_problem = str(
        best["problem"]
    )

    best_classifier = str(
        best["classifier"]
    )

    best_classifier_name = str(
        best["classifier_name"]
    )

    best_auc = float(
        best["observed_mean_outer_auc"]
    )

    # ========================================================
    # FINAL CONSOLE TABLE
    # ========================================================

    print("\n")
    print("=" * 88)
    print("FINAL PAPER RESULTS TABLE")
    print("=" * 88)

    display_columns = [
        "problem",
        "classifier_name",
        "observed_mean_outer_auc",
        "observed_outer_auc_sd",
        "permutation_mean_auc",
        "permutation_sd_auc",
        "permutation_max_auc",
        "permutation_exceedances_of_observed",
        "empirical_p_plus1",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n")
    print("=" * 88)
    print("BEST OBSERVED MEAN OUTER-FOLD AUC")
    print("=" * 88)

    print(
        f"Problem     : {best_problem}"
    )

    print(
        f"Classifier  : {best_classifier_name}"
    )

    print(
        f"Mean AUC    : {best_auc:.4f}"
    )

    print("\n")
    print("=" * 88)
    print("OUTPUT FILES")
    print("=" * 88)

    print(
        f"Complete summary:\n"
        f"  {summary_path}"
    )

    print(
        f"\nAll 720 permutation AUCs:\n"
        f"  {permutation_path}"
    )

    print(
        f"\nLean paper table:\n"
        f"  {paper_table_path}"
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
