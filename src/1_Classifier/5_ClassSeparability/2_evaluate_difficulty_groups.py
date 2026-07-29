"""
Evaluate validation performance by fixed class-difficulty group.

This script consumes predictions already produced by a 102-class model
evaluator. It does not run a neural network and does not perform coarse-label
mapping. Class-difficulty definitions must be created from validation analysis
and frozen by 2_prepare_separability_data.py before this script is run.

The output contains five rows:

1. cluster_hard: the 35 classes in selected visual-confusion clusters;
2. non_cluster_total: all 67 classes outside cluster_hard;
3. easy: the easy subset of non_cluster_total;
4. diffuse_hard: the diffuse-hard subset of non_cluster_total;
5. uncertain: the uncertain subset of non_cluster_total.

non_cluster_total overlaps with easy, diffuse_hard, and uncertain. It is
included as a direct comparison group for cluster_hard and must not be added
to the three non-cluster subgroup rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


NUM_CLASSES = 102
SPLITS = ("train", "val", "test")
FINE_GROUPS = (
    "easy",
    "cluster_hard",
    "diffuse_hard",
    "uncertain",
)
REPORT_GROUP_ORDER = (
    "cluster_hard",
    "non_cluster_total",
    "easy",
    "diffuse_hard",
    "uncertain",
)


def parse_arguments() -> argparse.Namespace:
    """Read prediction, definition, split, and output options."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate original 102-class predictions by fixed "
            "class-difficulty group."
        ),
    )
    parser.add_argument(
        "model_output_dir",
        type=Path,
        help=(
            "Model output directory containing "
            "<split>_predictions.csv."
        ),
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="val",
        help="Dataset split to evaluate (default: val).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "COMP9444_Group root. By default, parents[3] of this "
            "script is used."
        ),
    )
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing class_difficulty_groups.csv. "
            "Default: outputs/classifier/class_separability."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Prediction CSV override. Default: "
            "<model_output_dir>/<split>_predictions.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Result directory. Default: "
            "<model_output_dir>/difficulty_group_analysis."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path, project_root: Path) -> Path:
    """Resolve a path relative to the project root when necessary."""
    path = path.expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def resolve_project_root(requested: Path | None) -> Path:
    """Resolve and validate the COMP9444_Group project root."""
    if requested is None:
        project_root = Path(__file__).resolve().parents[3]
    else:
        project_root = requested.expanduser().resolve()

    classifier_output = project_root / "outputs" / "classifier"
    if not classifier_output.is_dir():
        raise NotADirectoryError(
            f"Invalid COMP9444_Group project root: {project_root}"
        )
    return project_root


def resolve_paths(
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Path]:
    """Resolve and validate all input and output paths."""
    model_output_dir = resolve_path(
        args.model_output_dir,
        project_root,
    )
    if not model_output_dir.is_dir():
        raise NotADirectoryError(
            f"Model output directory not found: {model_output_dir}"
        )

    definitions_dir = (
        project_root
        / "outputs"
        / "classifier"
        / "class_separability"
        if args.definitions_dir is None
        else resolve_path(args.definitions_dir, project_root)
    )
    predictions = (
        model_output_dir / f"{args.split}_predictions.csv"
        if args.predictions is None
        else resolve_path(args.predictions, project_root)
    )
    groups = definitions_dir / "class_difficulty_groups.csv"
    output_dir = (
        model_output_dir / "difficulty_group_analysis"
        if args.output_dir is None
        else resolve_path(args.output_dir, project_root)
    )

    missing = [
        path
        for path in (predictions, groups)
        if not path.is_file()
    ]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required input files are missing:\n"
            f"{missing_text}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "model_output_dir": model_output_dir,
        "predictions": predictions,
        "groups": groups,
        "output_dir": output_dir,
        "summary": (
            output_dir
            / f"{args.split}_difficulty_group_summary.csv"
        ),
    }


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate one original 102-class prediction table."""
    dataframe = pd.read_csv(path)
    required = {"true_label", "predicted_label"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV is missing columns: {sorted(missing)}"
        )
    if dataframe.empty:
        raise ValueError(f"Prediction CSV is empty: {path}")

    for column in ("true_label", "predicted_label"):
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        ).astype(int)
        invalid = ~dataframe[column].between(0, NUM_CLASSES - 1)
        if invalid.any():
            raise ValueError(
                f"{column} contains labels outside 0-101."
            )
    return dataframe


def load_groups(path: Path) -> pd.DataFrame:
    """Load and validate the frozen validation-derived group definitions."""
    dataframe = pd.read_csv(path)
    required = {"label", "class_name", "difficulty_group"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Difficulty-group CSV is missing columns: {sorted(missing)}"
        )

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="raise",
    ).astype(int)
    dataframe["difficulty_group"] = (
        dataframe["difficulty_group"].astype(str).str.strip()
    )

    if set(dataframe["label"]) != set(range(NUM_CLASSES)):
        raise ValueError(
            "Difficulty-group CSV must contain labels 0-101 exactly."
        )
    if dataframe["label"].duplicated().any():
        raise ValueError(
            "Difficulty-group CSV contains duplicate labels."
        )

    unknown_groups = set(
        dataframe["difficulty_group"]
    ) - set(FINE_GROUPS)
    if unknown_groups:
        raise ValueError(
            f"Unknown difficulty groups: {sorted(unknown_groups)}"
        )
    return dataframe.sort_values("label").reset_index(drop=True)


def calculate_full_per_class_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate metrics for all 102 original classes.

    Precision is calculated from the complete split so false positives coming
    from classes outside a reported group remain included.
    """
    labels = list(range(NUM_CLASSES))
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            predictions["true_label"],
            predictions["predicted_label"],
            labels=labels,
            zero_division=0,
        )
    )
    return pd.DataFrame({
        "label": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })


def build_group_summary(
    predictions: pd.DataFrame,
    groups: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Create cluster, non-cluster, and subgroup validation summaries."""
    class_metrics = calculate_full_per_class_metrics(predictions)
    label_to_group = groups.set_index("label")[
        "difficulty_group"
    ].to_dict()
    sample_groups = predictions["true_label"].map(label_to_group)

    if sample_groups.isna().any():
        raise ValueError(
            "Some prediction labels have no difficulty-group definition."
        )

    total_samples = len(predictions)
    total_errors = int(
        (
            predictions["true_label"]
            != predictions["predicted_label"]
        ).sum()
    )

    cluster_labels = set(
        groups.loc[
            groups["difficulty_group"] == "cluster_hard",
            "label",
        ]
    )
    report_labels: dict[str, list[int]] = {
        "cluster_hard": sorted(cluster_labels),
        "non_cluster_total": sorted(
            set(range(NUM_CLASSES)) - cluster_labels
        ),
        "easy": groups.loc[
            groups["difficulty_group"] == "easy",
            "label",
        ].tolist(),
        "diffuse_hard": groups.loc[
            groups["difficulty_group"] == "diffuse_hard",
            "label",
        ].tolist(),
        "uncertain": groups.loc[
            groups["difficulty_group"] == "uncertain",
            "label",
        ].tolist(),
    }

    rows: list[dict[str, int | float | str | bool]] = []
    for group_order, group_name in enumerate(
        REPORT_GROUP_ORDER,
        start=1,
    ):
        labels = report_labels[group_name]
        if group_name == "non_cluster_total":
            mask = sample_groups != "cluster_hard"
        else:
            mask = sample_groups == group_name

        selected = predictions.loc[mask]
        number_samples = len(selected)
        number_correct = int(
            (
                selected["true_label"]
                == selected["predicted_label"]
            ).sum()
        )
        number_errors = number_samples - number_correct
        selected_class_metrics = class_metrics[
            class_metrics["label"].isin(labels)
        ]

        rows.append({
            "group_order": group_order,
            "split": split,
            "difficulty_group": group_name,
            "is_aggregate": group_name == "non_cluster_total",
            "number_classes": len(labels),
            "number_samples": number_samples,
            "sample_share": (
                number_samples / total_samples
                if total_samples
                else 0.0
            ),
            "number_correct": number_correct,
            "number_errors": number_errors,
            "conditional_accuracy": (
                number_correct / number_samples
                if number_samples
                else 0.0
            ),
            "conditional_error_rate": (
                number_errors / number_samples
                if number_samples
                else 0.0
            ),
            "error_share": (
                number_errors / total_errors
                if total_errors
                else 0.0
            ),
            "macro_precision": float(
                selected_class_metrics["precision"].mean()
            ),
            "macro_recall": float(
                selected_class_metrics["recall"].mean()
            ),
            "macro_f1": float(
                selected_class_metrics["f1"].mean()
            ),
        })

    return pd.DataFrame(rows)


def main() -> None:
    """Load definitions and predictions, evaluate groups, and save CSV."""
    args = parse_arguments()
    project_root = resolve_project_root(args.project_root)
    paths = resolve_paths(args, project_root)

    predictions = load_predictions(paths["predictions"])
    groups = load_groups(paths["groups"])
    summary = build_group_summary(
        predictions,
        groups,
        args.split,
    )
    summary.to_csv(
        paths["summary"],
        index=False,
        float_format="%.6f",
    )

    print("Difficulty-group evaluation completed successfully.")
    print("Split:", args.split)
    print("Predictions:", paths["predictions"])
    print("Group definitions:", paths["groups"])
    print("Summary:", paths["summary"])


if __name__ == "__main__":
    main()
