"""
Evaluate original fine-label and merged coarse-label performance.

This script consumes prediction CSV files already produced by a model
evaluator. It does not run or retrain a neural network.

The coarse-label mapping must first be prepared and frozen by:

    2_evaluate_difficulty_groups.py
        -> selected_clusters.csv
    3_prepare_separability_data.py
        -> coarse_label_mapping.csv

For test-set evaluation, both definitions must be derived from validation data
only. This prevents test-set information from influencing which classes are
merged.

The script produces:

1. <split>_original_vs_coarse_summary.csv
   Compares the original 102-class metrics with the merged coarse metrics.

2. <split>_coarse_classification_report.csv
   Reports precision, recall, F1-score, support, and error rate for every
   coarse class.

3. <split>_coarse_confusion_matrix.csv
   Stores the confusion matrix after fine labels are mapped to coarse labels.

4. <split>_coarse_recovered_errors.csv
   Lists fine-label mistakes that become correct after both labels are mapped
   to the same valid coarse class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


NUM_FINE_CLASSES = 102
SPLITS = ("train", "val", "test")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a model at the original 102-class level and at "
            "the frozen merged coarse-label level."
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
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Project root directory. By default, parents[3] of this "
            "script is used."
        ),
    )
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=None,
        help=(
            "Directory produced by 3_prepare_separability_data.py. "
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
            "<model_output_dir>/separability_analysis."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path, project_root: Path) -> Path:
    """Resolve an absolute path or a path relative to the project root."""
    path = path.expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def resolve_project_root(requested: Path | None) -> Path:
    """Resolve and validate the project root."""
    if requested is None:
        project_root = Path(__file__).resolve().parents[3]
    else:
        project_root = requested.expanduser().resolve()

    if not (project_root / "outputs" / "classifier").is_dir():
        raise NotADirectoryError(
            f"Invalid project root: {project_root}\n"
            "Expected an outputs/classifier directory."
        )
    return project_root


def resolve_paths(
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Path]:
    """Resolve input and output paths and check required inputs."""
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
    output_dir = (
        model_output_dir / "separability_analysis"
        if args.output_dir is None
        else resolve_path(args.output_dir, project_root)
    )

    paths = {
        "mapping": definitions_dir / "coarse_label_mapping.csv",
        "predictions": predictions,
        "output_dir": output_dir,
    }
    missing = [
        path
        for key, path in paths.items()
        if key != "output_dir" and not path.is_file()
    ]
    if missing:
        text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required input files are missing:\n"
            f"{text}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return paths


def load_mapping(path: Path) -> pd.DataFrame:
    """Load and validate the frozen fine-to-coarse label mapping."""
    dataframe = pd.read_csv(path)
    required = {
        "fine_label",
        "class_name",
        "coarse_label",
        "coarse_name",
        "is_merged",
        "cluster_name",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Coarse mapping is missing columns: {sorted(missing)}"
        )

    dataframe["fine_label"] = pd.to_numeric(
        dataframe["fine_label"],
        errors="raise",
    ).astype(int)
    dataframe["coarse_label"] = pd.to_numeric(
        dataframe["coarse_label"],
        errors="raise",
    ).astype(int)
    dataframe = dataframe.sort_values("fine_label").reset_index(
        drop=True
    )

    if dataframe["fine_label"].tolist() != list(
        range(NUM_FINE_CLASSES)
    ):
        raise ValueError(
            "Coarse mapping must contain fine labels 0-101 exactly."
        )

    if dataframe["class_name"].isna().any():
        raise ValueError("Coarse mapping contains missing class names.")
    if dataframe["coarse_name"].isna().any():
        raise ValueError("Coarse mapping contains missing coarse names.")

    fine_name_counts = dataframe.groupby(
        "fine_label"
    )["class_name"].nunique()
    if (fine_name_counts != 1).any():
        raise ValueError(
            "One fine label maps to multiple class names."
        )

    coarse_pairs = (
        dataframe[["coarse_label", "coarse_name"]]
        .drop_duplicates()
        .sort_values("coarse_label")
        .reset_index(drop=True)
    )
    if coarse_pairs["coarse_label"].tolist() != list(
        range(len(coarse_pairs))
    ):
        raise ValueError(
            "Coarse labels must be contiguous and start at zero."
        )
    if coarse_pairs["coarse_label"].duplicated().any():
        raise ValueError(
            "One coarse label maps to multiple coarse names."
        )

    return dataframe


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate the saved fine-label predictions."""
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
        invalid = ~dataframe[column].between(
            0,
            NUM_FINE_CLASSES - 1,
        )
        if invalid.any():
            invalid_values = sorted(
                dataframe.loc[invalid, column].unique().tolist()
            )
            raise ValueError(
                f"{column} contains labels outside 0-101: "
                f"{invalid_values[:10]}"
            )

    if "image_name" not in dataframe.columns:
        dataframe.insert(
            0,
            "image_name",
            [
                f"row_{index:08d}"
                for index in range(len(dataframe))
            ],
        )

    dataframe["image_name"] = dataframe["image_name"].astype(str)
    if dataframe["image_name"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["image_name"].duplicated(),
            "image_name",
        ].head(5)
        raise ValueError(
            "Prediction CSV contains duplicate image names: "
            f"{duplicates.tolist()}"
        )

    return dataframe


def geometric_mean_from_recalls(recalls: np.ndarray) -> float:
    """Calculate the geometric mean of per-class recalls."""
    recalls = np.asarray(recalls, dtype=np.float64)
    if len(recalls) == 0 or np.any(recalls <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def calculate_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    labels: list[int],
) -> dict[str, float | int]:
    """Calculate overall and macro-averaged metrics."""
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=labels,
            zero_division=0,
        )
    )
    return {
        "number_samples": int(len(true_labels)),
        "number_correct": int(
            np.sum(true_labels == predicted_labels)
        ),
        "number_errors": int(
            np.sum(true_labels != predicted_labels)
        ),
        "accuracy": float(
            accuracy_score(true_labels, predicted_labels)
        ),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "g_mean": geometric_mean_from_recalls(recall),
    }


def calculate_per_class_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    labels: list[int],
) -> pd.DataFrame:
    """Calculate precision, recall, F1-score, and support per class."""
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
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


def add_names_and_coarse_labels(
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Add fine-class names and mapped coarse labels to predictions."""
    fine_to_name = mapping.set_index("fine_label")[
        "class_name"
    ].to_dict()
    fine_to_coarse = mapping.set_index("fine_label")[
        "coarse_label"
    ].to_dict()
    coarse_to_name = (
        mapping[["coarse_label", "coarse_name"]]
        .drop_duplicates()
        .set_index("coarse_label")["coarse_name"]
        .to_dict()
    )

    result = predictions.copy()
    result["true_class"] = result["true_label"].map(
        fine_to_name
    )
    result["predicted_class"] = result["predicted_label"].map(
        fine_to_name
    )
    result["true_coarse_label"] = result["true_label"].map(
        fine_to_coarse
    )
    result["predicted_coarse_label"] = (
        result["predicted_label"].map(fine_to_coarse)
    )
    result["true_coarse_class"] = (
        result["true_coarse_label"].map(coarse_to_name)
    )
    result["predicted_coarse_class"] = (
        result["predicted_coarse_label"].map(coarse_to_name)
    )

    mapped_columns = [
        "true_class",
        "predicted_class",
        "true_coarse_label",
        "predicted_coarse_label",
        "true_coarse_class",
        "predicted_coarse_class",
    ]
    if result[mapped_columns].isna().any().any():
        raise ValueError(
            "At least one prediction label is absent from the "
            "coarse mapping."
        )

    result["true_coarse_label"] = result[
        "true_coarse_label"
    ].astype(int)
    result["predicted_coarse_label"] = result[
        "predicted_coarse_label"
    ].astype(int)
    return result


def build_original_vs_coarse_summary(
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Compare the original fine evaluation with the coarse evaluation."""
    fine_labels = list(range(NUM_FINE_CLASSES))
    coarse_labels = sorted(
        mapping["coarse_label"].unique().tolist()
    )

    original = calculate_metrics(
        predictions["true_label"].to_numpy(),
        predictions["predicted_label"].to_numpy(),
        fine_labels,
    )
    coarse = calculate_metrics(
        predictions["true_coarse_label"].to_numpy(),
        predictions["predicted_coarse_label"].to_numpy(),
        coarse_labels,
    )

    original_errors = int(original["number_errors"])
    recovered = (
        original_errors - int(coarse["number_errors"])
    )
    recovery_rate = (
        recovered / original_errors
        if original_errors > 0
        else 0.0
    )

    rows = [
        {
            "evaluation_order": 1,
            "split": split,
            "evaluation_level": "original_fine",
            "number_classes": len(fine_labels),
            **original,
            "errors_recovered_by_merging": 0,
            "error_recovery_rate": 0.0,
        },
        {
            "evaluation_order": 2,
            "split": split,
            "evaluation_level": "merged_coarse",
            "number_classes": len(coarse_labels),
            **coarse,
            "errors_recovered_by_merging": recovered,
            "error_recovery_rate": recovery_rate,
        },
    ]
    return pd.DataFrame(rows)


def build_coarse_report(
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Build a per-coarse-class classification report."""
    coarse_names = (
        mapping[["coarse_label", "coarse_name"]]
        .drop_duplicates()
        .sort_values("coarse_label")
        .reset_index(drop=True)
    )
    labels = coarse_names["coarse_label"].tolist()
    report = calculate_per_class_metrics(
        predictions["true_coarse_label"].to_numpy(),
        predictions["predicted_coarse_label"].to_numpy(),
        labels,
    )
    report = report.rename(columns={"label": "coarse_label"})
    report.insert(0, "split", split)
    report = report.merge(
        coarse_names,
        on="coarse_label",
        how="left",
        validate="one_to_one",
    )

    report["number_correct"] = np.rint(
        report["recall"] * report["support"]
    ).astype(int)
    report["number_errors"] = (
        report["support"] - report["number_correct"]
    )
    report["error_rate"] = np.divide(
        report["number_errors"],
        report["support"],
        out=np.zeros(len(report), dtype=np.float64),
        where=report["support"].to_numpy() > 0,
    )

    ordered_columns = [
        "split",
        "coarse_label",
        "coarse_name",
        "precision",
        "recall",
        "f1",
        "support",
        "number_correct",
        "number_errors",
        "error_rate",
    ]
    return report[ordered_columns].sort_values(
        ["f1", "recall", "support"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_recovered_errors(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """List fine mistakes absorbed by a valid coarse-label merge."""
    original_wrong = (
        predictions["true_label"]
        != predictions["predicted_label"]
    )
    coarse_correct = (
        predictions["true_coarse_label"]
        == predictions["predicted_coarse_label"]
    )
    recovered = predictions[
        original_wrong & coarse_correct
    ].copy()
    recovered.insert(
        0,
        "recovered_error_rank",
        range(1, len(recovered) + 1),
    )

    ordered_columns = [
        "recovered_error_rank",
        "image_name",
        "true_label",
        "true_class",
        "predicted_label",
        "predicted_class",
        "true_coarse_label",
        "true_coarse_class",
        "predicted_coarse_label",
        "predicted_coarse_class",
    ]
    extra_columns = [
        column
        for column in recovered.columns
        if column not in ordered_columns
    ]
    return recovered[ordered_columns + extra_columns]


def save_coarse_confusion_matrix(
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
    path: Path,
) -> None:
    """Save a named coarse-label confusion matrix."""
    coarse_names = (
        mapping[["coarse_label", "coarse_name"]]
        .drop_duplicates()
        .sort_values("coarse_label")
    )
    labels = coarse_names["coarse_label"].tolist()
    names = coarse_names["coarse_name"].tolist()
    matrix = confusion_matrix(
        predictions["true_coarse_label"],
        predictions["predicted_coarse_label"],
        labels=labels,
    )
    matrix_dataframe = pd.DataFrame(
        matrix,
        index=names,
        columns=names,
    )
    matrix_dataframe.index.name = "true_coarse_class"
    matrix_dataframe.to_csv(path, index=True)


def main() -> None:
    args = parse_arguments()
    project_root = resolve_project_root(args.project_root)
    paths = resolve_paths(args, project_root)

    mapping = load_mapping(paths["mapping"])
    predictions = load_predictions(paths["predictions"])
    predictions = add_names_and_coarse_labels(
        predictions,
        mapping,
    )

    summary = build_original_vs_coarse_summary(
        predictions,
        mapping,
        args.split,
    )
    coarse_report = build_coarse_report(
        predictions,
        mapping,
        args.split,
    )
    recovered_errors = build_recovered_errors(predictions)

    prefix = args.split
    output_paths = {
        "summary": (
            paths["output_dir"]
            / f"{prefix}_original_vs_coarse_summary.csv"
        ),
        "coarse_report": (
            paths["output_dir"]
            / f"{prefix}_coarse_classification_report.csv"
        ),
        "coarse_confusion": (
            paths["output_dir"]
            / f"{prefix}_coarse_confusion_matrix.csv"
        ),
        "recovered": (
            paths["output_dir"]
            / f"{prefix}_coarse_recovered_errors.csv"
        ),
    }

    summary.to_csv(
        output_paths["summary"],
        index=False,
        float_format="%.6f",
    )
    coarse_report.to_csv(
        output_paths["coarse_report"],
        index=False,
        float_format="%.6f",
    )
    recovered_errors.to_csv(
        output_paths["recovered"],
        index=False,
    )
    save_coarse_confusion_matrix(
        predictions,
        mapping,
        output_paths["coarse_confusion"],
    )

    original_row = summary.iloc[0]
    coarse_row = summary.iloc[1]

    print("Coarse-label evaluation completed successfully.")
    print("Split:", args.split)
    print("Predictions:", paths["predictions"])
    print("Coarse mapping:", paths["mapping"])
    print(
        "Original 102-class accuracy / Macro-F1:",
        f"{original_row['accuracy']:.4f} / "
        f"{original_row['macro_f1']:.4f}",
    )
    print(
        "Merged coarse accuracy / Macro-F1:",
        f"{coarse_row['accuracy']:.4f} / "
        f"{coarse_row['macro_f1']:.4f}",
    )
    print(
        "Fine errors absorbed by valid coarse merges:",
        int(coarse_row["errors_recovered_by_merging"]),
    )
    print("Output directory:", paths["output_dir"])
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()