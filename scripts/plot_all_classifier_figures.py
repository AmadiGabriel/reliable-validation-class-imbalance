from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, roc_auc_score


# ============================================================
# GLOBAL STYLE
# ============================================================

font = {
    "family": "Times New Roman",
    "size": 14,
}

mpl.rc("font", **font)
mpl.rc("xtick", labelsize=14)
mpl.rc("ytick", labelsize=14)


# ============================================================
# USER SETTINGS
# ============================================================

# Histogram bin width.
#
# Smaller value -> narrower bars / more bins
# Larger value  -> wider bars / fewer bins
#
# Good starting values:
# 0.020
# 0.025
# 0.030
HIST_BIN_WIDTH = 0.038

# Main ROC line width
MAIN_ROC_LW = 1.8

# Chance/reference line width
REFERENCE_LW = 1.2

# Grid line width
GRID_LW = 0.7

# True-AUC marker sizes
FP_MARKER_SIZE = 18
FPS_MARKER_SIZE = 18

# Output directory
REPO_ROOT = Path(__file__).resolve().parents[1]

# Reproduced figures are kept separate from the publication-ready
# figures committed under figures/.
OUTPUT_DIR = REPO_ROOT / "figures_reproduced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RESULT LOCATIONS
# ============================================================

# The complete fold-level and permutation outputs are distributed
# separately from the Git repository. To reproduce the figures,
# extract the archived results so that they appear under
# <repository>/results_complete/.
RESULTS_ROOT = REPO_ROOT / "results_complete"

RESULTS_ROOT_BY_CLASSIFIER = {
    "lgbm": RESULTS_ROOT,
    "ada": RESULTS_ROOT,
    "rf": RESULTS_ROOT,
    "xgb": RESULTS_ROOT,
}

CLASSIFIER_DISPLAY = {
    "lgbm": "LightGBM",
    "ada": "AdaBoost",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}

PROBLEMS = ["FP", "FPs"]


# ============================================================
# VISUAL STYLE
# ============================================================

HIST_STYLE = {
    "FP": {
        "color": "0.55",
        "alpha": 0.65,
        "hatch": "//",
        "label": "FP permutations",
    },
    "FPs": {
        "color": "0.82",
        "alpha": 0.65,
        "hatch": "\\\\",
        "label": "FPs permutations",
    },
}

ROC_STYLE = {
    "FP": {
        "color": "black",
        "linestyle": "-",
        "label": "FP",
    },
    "FPs": {
        "color": "0.35",
        "linestyle": "--",
        "label": "FPs",
    },
}


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_single(pattern):
    """
    Find exactly one matching file.
    """
    matches = sorted(Path().glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file matched:\n{pattern}"
        )

    if len(matches) > 1:
        print(
            f"Warning: multiple files matched:\n"
            f"{pattern}\n"
            f"Using:\n{matches[0]}"
        )

    return matches[0]


def load_true_summary(results_root, problem, classifier):
    """
    Load the true-label summary JSON.
    """
    pattern = (
        f"{results_root}/{problem}/{classifier}/true/"
        f"summary_{problem}_{classifier}_true_seed*.json"
    )

    path = find_single(pattern)
    summary = load_json(path)

    return summary, path


def load_perm_summaries(results_root, problem, classifier):
    """
    Load all permutation summary JSON files.
    """
    pattern = (
        f"{results_root}/{problem}/{classifier}/perm/"
        f"summary_{problem}_{classifier}_perm_seed*.json"
    )

    paths = sorted(Path().glob(pattern))

    if len(paths) == 0:
        raise FileNotFoundError(
            f"No permutation summaries found for:\n"
            f"problem={problem}, classifier={classifier}"
        )

    rows = [load_json(path) for path in paths]

    return pd.DataFrame(rows), paths


def resolve_predictions_path(summary_path, summary_data):
    """
    Resolve the out-of-fold prediction CSV robustly.

    The summary JSON may contain a path from the machine on which the
    experiment was originally run (e.g. 'results/...'). After results are
    copied or merged, the prediction CSV should be looked for in the same
    directory as the summary JSON.
    """

    pred_value = summary_data.get("oof_predictions_file")

    if pred_value is None:
        raise KeyError(
            f"'oof_predictions_file' not found in:\n{summary_path}"
        )

    pred_path = Path(pred_value)

    # --------------------------------------------------------
    # 1. Absolute path, if it still exists
    # --------------------------------------------------------
    if pred_path.is_absolute() and pred_path.exists():
        return pred_path

    # --------------------------------------------------------
    # 2. Original relative path, if it still exists
    # --------------------------------------------------------
    if pred_path.exists():
        return pred_path.resolve()

    # --------------------------------------------------------
    # 3. Most important for merged results:
    #    prediction file beside the summary JSON
    # --------------------------------------------------------
    candidate = summary_path.parent / pred_path.name

    if candidate.exists():
        return candidate.resolve()

    # --------------------------------------------------------
    # 4. Final diagnostic
    # --------------------------------------------------------
    raise FileNotFoundError(
        "\nCould not resolve prediction file.\n"
        f"Stored path in JSON:\n  {pred_value}\n"
        f"Summary file:\n  {summary_path}\n"
        f"Also tried:\n  {candidate}\n"
    )


def classifier_results_available(results_root, classifier):
    """
    Check whether both FP and FPs have:
      - true-label results
      - at least one permutation result
    """
    try:
        for problem in PROBLEMS:
            load_true_summary(
                results_root,
                problem,
                classifier,
            )

            load_perm_summaries(
                results_root,
                problem,
                classifier,
            )

        return True

    except Exception:
        return False


# ============================================================
# HISTOGRAM
# ============================================================

def make_histogram(
    results_root,
    classifier,
    output_dir,
):
    classifier_name = CLASSIFIER_DISPLAY.get(
        classifier,
        classifier,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    # --------------------------------------------------------
    # Common histogram bins from 0 to 1
    # --------------------------------------------------------
    common_bins = np.arange(
        0.0,
        1.0 + HIST_BIN_WIDTH,
        HIST_BIN_WIDTH,
    )

    true_auc_positions = {}

    # --------------------------------------------------------
    # Plot FP and FPs permutation distributions
    # --------------------------------------------------------
    for problem in PROBLEMS:

        true_summary, _ = load_true_summary(
            results_root,
            problem,
            classifier,
        )

        perm_df, perm_paths = load_perm_summaries(
            results_root,
            problem,
            classifier,
        )

        perm_values = (
            perm_df["mean_outer_auc"]
            .astype(float)
            .to_numpy()
        )

        true_auc = float(
            true_summary["mean_outer_auc"]
        )

        true_auc_positions[problem] = true_auc

        style = HIST_STYLE[problem]

        print(
            f"{classifier_name} {problem}: "
            f"{len(perm_values)} permutation runs"
        )

        print(
            f"{classifier_name} {problem}: "
            f"true mean outer-fold AUC = "
            f"{true_auc:.6f}"
        )

        print(
            f"{classifier_name} {problem}: "
            f"permutation mean = "
            f"{np.mean(perm_values):.6f}"
        )

        print(
            f"{classifier_name} {problem}: "
            f"permutation SD = "
            f"{np.std(perm_values, ddof=1):.6f}"
        )

        print(
            f"{classifier_name} {problem}: "
            f"permutation min = "
            f"{np.min(perm_values):.6f}"
        )

        print(
            f"{classifier_name} {problem}: "
            f"permutation max = "
            f"{np.max(perm_values):.6f}"
        )

        ax.hist(
            perm_values,
            bins=common_bins,
            alpha=style["alpha"],
            edgecolor="black",
            color=style["color"],
            hatch=style["hatch"],
            linewidth=1.0,
            label=style["label"],
        )

    # --------------------------------------------------------
    # Axes formatting
    # --------------------------------------------------------
    ax.set_xlabel(
        "Mean outer-fold AUC"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_title(
        f"{classifier_name} permutation distributions: FP vs FPs"
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_xticks(
        np.arange(
            0,
            1.01,
            0.1,
        )
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=GRID_LW,
        alpha=0.9,
    )

    # Keep histogram y-axis starting exactly at zero
    ymax = ax.get_ylim()[1]

    ax.set_ylim(
        bottom=0,
        top=ymax,
    )

    # --------------------------------------------------------
    # True AUC markers on x-axis
    # --------------------------------------------------------

    # FP true AUC -> *
    ax.plot(
        [true_auc_positions["FP"]],
        [0],
        marker=r"$*$",
        color="black",
        markersize=FP_MARKER_SIZE,
        linestyle="None",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        zorder=10,
        label="_nolegend_",
    )

    # FPs true AUC -> +
    ax.plot(
        [true_auc_positions["FPs"]],
        [0],
        marker=r"$+$",
        color="black",
        markersize=FPS_MARKER_SIZE,
        linestyle="None",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        zorder=10,
        label="_nolegend_",
    )

    # --------------------------------------------------------
    # Histogram legend
    # --------------------------------------------------------

    hist_legend_handles = [

        Patch(
            facecolor=HIST_STYLE["FP"]["color"],
            edgecolor="black",
            alpha=HIST_STYLE["FP"]["alpha"],
            hatch=HIST_STYLE["FP"]["hatch"],
            label="FP permutations",
        ),

        Patch(
            facecolor=HIST_STYLE["FPs"]["color"],
            edgecolor="black",
            alpha=HIST_STYLE["FPs"]["alpha"],
            hatch=HIST_STYLE["FPs"]["hatch"],
            label="FPs permutations",
        ),

        Line2D(
            [0],
            [0],
            marker=r"$*$",
            color="black",
            linestyle="None",
            markersize=14,
            label="FP true",
        ),

        Line2D(
            [0],
            [0],
            marker=r"$+$",
            color="black",
            linestyle="None",
            markersize=14,
            label="FPs true",
        ),
    ]

    ax.legend(
        handles=hist_legend_handles,
        loc="best",
        frameon=True,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{classifier}_histogram_FP_vs_FPs_greyscale.png"
    )

    fig.savefig(
        output_path,
        dpi=500,
        bbox_inches="tight",
        transparent=True,
    )

    print(
        f"\nSaved histogram:\n"
        f"{output_path}"
    )

    print(
        f"Histogram bin width = "
        f"{HIST_BIN_WIDTH}"
    )

    print(
        f"Number of bins = "
        f"{len(common_bins) - 1}"
    )

    plt.show()


# ============================================================
# ROC
# ============================================================

def make_roc_plot(
    results_root,
    classifier,
    output_dir,
):
    classifier_name = CLASSIFIER_DISPLAY.get(
        classifier,
        classifier,
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    for problem in PROBLEMS:

        true_summary, true_summary_path = (
            load_true_summary(
                results_root,
                problem,
                classifier,
            )
        )

        pred_path = resolve_predictions_path(
            true_summary_path,
            true_summary,
        )

        pred_df = pd.read_csv(
            pred_path
        )

        fpr, tpr, _ = roc_curve(
            pred_df["y_true"],
            pred_df["y_prob"],
        )

        pooled_auc = roc_auc_score(
            pred_df["y_true"],
            pred_df["y_prob"],
        )

        style = ROC_STYLE[problem]

        ax.plot(
            fpr,
            tpr,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=MAIN_ROC_LW,
            label=(
                f"{style['label']} "
                f"(pooled OOF AUC = "
                f"{pooled_auc:.3f})"
            ),
        )

    # --------------------------------------------------------
    # Chance line
    # --------------------------------------------------------
    ax.plot(
        [0, 1],
        [0, 1],
        color="0.55",
        linestyle=":",
        linewidth=REFERENCE_LW,
        label="Chance",
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------
    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        f"{classifier_name} ROC curves: FP vs FPs"
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_xticks(
        np.arange(
            0,
            1.01,
            0.2,
        )
    )

    ax.set_yticks(
        np.arange(
            0,
            1.01,
            0.2,
        )
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=GRID_LW,
        alpha=0.9,
    )

    ax.legend(
        loc="lower right",
        frameon=True,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{classifier}_roc_FP_vs_FPs_greyscale.png"
    )

    fig.savefig(
        output_path,
        dpi=500,
        bbox_inches="tight",
        transparent=True,
    )

    print(
        f"\nSaved ROC:\n"
        f"{output_path}"
    )

    plt.show()


# ============================================================
# WRAPPER
# ============================================================

def make_plots_for_classifier(
    classifier,
    output_dir=OUTPUT_DIR,
):
    results_root = (
        RESULTS_ROOT_BY_CLASSIFIER[
            classifier
        ]
    )

    if not classifier_results_available(
        results_root,
        classifier,
    ):

        print(
            "\n=============================================="
        )

        print(
            f"Skipping {classifier}"
        )

        print(
            "Required FP/FPs true or permutation "
            f"results are not yet available under:\n"
            f"{results_root}"
        )

        print(
            "=============================================="
        )

        return

    print(
        "\n=============================================="
    )

    print(
        f"Generating figures for "
        f"{CLASSIFIER_DISPLAY[classifier]}"
    )

    print(
        f"Results directory: "
        f"{results_root}"
    )

    print(
        "=============================================="
    )

    make_histogram(
        results_root,
        classifier,
        output_dir,
    )

    make_roc_plot(
        results_root,
        classifier,
        output_dir,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    classifiers = [
        "lgbm",
        "ada",
        "rf",
        "xgb",
    ]

    for classifier in classifiers:

        make_plots_for_classifier(
            classifier,
            output_dir=OUTPUT_DIR,
        )


if __name__ == "__main__":
    main()
