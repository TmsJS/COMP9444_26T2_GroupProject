"""
Evaluate original, merged-coarse, and difficulty-group performance.

This script consumes prediction CSV files already produced by a model
evaluator. It does not run a neural network. The validation-derived mapping
created by 1_prepare_separability_data.py must be frozen before evaluating
the test split.
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
GROUP_ORDER = {
    "easy": 1,
    "cluster_hard": 2,
    "diffuse_hard": 3,
    "uncertain": 4,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a model at the original 102-class level, at the "
            "merged coarse level, and by fixed class-difficulty group."
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
            "COMP9444_Group root. By default, parents[3] of this "
            "script is used."
        ),
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Directory produced by 1_prepare_separability_data.py. "
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
            "Result directory. Default: <analysis-dir>/<model-folder>."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path, project_root: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def resolve_project_root(requested: Path | None) -> Path:
    if requested is None:
        project_root = Path(__file__).resolve().parents[3]
    else:
        project_root = requested.expanduser().resolve()

    if not (project_root / "outputs" / "classifier").is_dir():
        raise NotADirectoryError(
            f"Invalid COMP9444_Group project root: {project_root}"
        )
    return project_root


def resolve_paths(
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Path]:
    model_output_dir = resolve_path(
        args.model_output_dir,
        project_root,
    )
    if not model_output_dir.is_dir():
        raise NotADirectoryError(
            f"Model output directory not found: {model_output_dir}"
        )

    analysis_dir = (
        project_root
        / "outputs"
        / "classifier"
        / "class_separability"
        if args.analysis_dir is None
        else resolve_path(args.analysis_dir, project_root)
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
        "mapping": analysis_dir / "coarse_label_mapping.csv",
        "groups": analysis_dir / "class_difficulty_groups.csv",
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

    coarse_pairs = (
        dataframe[["coarse_label", "coarse_name"]]
        .drop_duplicates()
        .sort_values("coarse_label")
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


def load_groups(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    required = {
        "label",
        "class_name",
        "difficulty_group",
        "val_support",
        "val_recall",
        "val_f1",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Difficulty-group CSV is missing columns: {sorted(missing)}"
        )

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="raise",
    ).astype(int)
    if set(dataframe["label"]) != set(range(NUM_FINE_CLASSES)):
        raise ValueError(
            "Difficulty-group CSV must contain labels 0-101 exactly."
        )
    unknown_groups = (
        set(dataframe["difficulty_group"]) - set(GROUP_ORDER)
    )
    if unknown_groups:
        raise ValueError(
            f"Unknown difficulty groups: {sorted(unknown_groups)}"
        )
    return dataframe.sort_values("label").reset_index(drop=True)


def load_predictions(path: Path) -> pd.DataFrame:
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
            raise ValueError(
                f"{column} contains labels outside 0-101."
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
    recalls = np.asarray(recalls, dtype=np.float64)
    if len(recalls) == 0 or np.any(recalls <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def calculate_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    labels: list[int],
) -> dict[str, float | int]:
    precision, recall, f1, support = (
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
    return result


def build_original_vs_coarse_summary(
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    fine_labels = list(range(NUM_FINE_CLASSES))
    coarse_labels = sorted(mapping["coarse_label"].unique())

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
            "error_recovery_rate": (
                recovered / original_errors
                if original_errors > 0
                else 0.0
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_coarse_report(
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
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
    report["number_errors"] = (
        report["support"]
        - np.rint(report["recall"] * report["support"]).astype(int)
    )
    report["error_rate"] = np.divide(
        report["number_errors"],
        report["support"],
        out=np.zeros(len(report), dtype=np.float64),
        where=report["support"].to_numpy() > 0,
    )
    return report.sort_values(
        ["f1", "recall", "support"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_group_reports(
    predictions: pd.DataFrame,
    groups: pd.DataFrame,
    mapping: pd.DataFrame,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_metrics = calculate_per_class_metrics(
        predictions["true_label"].to_numpy(),
        predictions["predicted_label"].to_numpy(),
        list(range(NUM_FINE_CLASSES)),
    )
    details = groups.merge(
        class_metrics,
        on="label",
        how="left",
        validate="one_to_one",
        suffixes=("_definition", f"_{split}"),
    )
    details = details.merge(
        mapping[
            [
                "fine_label",
                "coarse_label",
                "coarse_name",
                "is_merged",
                "cluster_name",
            ]
        ],
        left_on="label",
        right_on="fine_label",
        how="left",
        validate="one_to_one",
        suffixes=("", "_mapping"),
    )
    details.insert(0, "split", split)
    details["number_correct"] = np.rint(
        details["recall"] * details["support"]
    ).astype(int)
    details["number_errors"] = (
        details["support"] - details["number_correct"]
    )
    details["class_error_rate"] = np.divide(
        details["number_errors"],
        details["support"],
        out=np.zeros(len(details), dtype=np.float64),
        where=details["support"].to_numpy() > 0,
    )
    details["difficulty_group_order"] = details[
        "difficulty_group"
    ].map(GROUP_ORDER)
    details = details.sort_values(
        ["difficulty_group_order", "f1", "recall", "label"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    total_samples = len(predictions)
    total_errors = int(
        np.sum(
            predictions["true_label"].to_numpy()
            != predictions["predicted_label"].to_numpy()
        )
    )
    label_to_group = groups.set_index("label")[
        "difficulty_group"
    ].to_dict()
    sample_groups = predictions["true_label"].map(label_to_group)
    summary_rows: list[dict[str, int | float | str]] = []

    for group_name, group_order in GROUP_ORDER.items():
        group_labels = groups.loc[
            groups["difficulty_group"] == group_name,
            "label",
        ].tolist()
        mask = sample_groups == group_name
        true_group = predictions.loc[mask, "true_label"].to_numpy()
        pred_group = predictions.loc[
            mask,
            "predicted_label",
        ].to_numpy()
        number_samples = int(mask.sum())
        number_correct = int(np.sum(true_group == pred_group))
        number_errors = number_samples - number_correct
        group_class_metrics = class_metrics[
            class_metrics["label"].isin(group_labels)
        ]

        summary_rows.append({
            "group_order": group_order,
            "split": split,
            "difficulty_group": group_name,
            "number_classes": len(group_labels),
            "number_samples": number_samples,
            "sample_share": (
                number_samples / total_samples
                if total_samples > 0
                else 0.0
            ),
            "number_correct": number_correct,
            "number_errors": number_errors,
            "conditional_accuracy": (
                number_correct / number_samples
                if number_samples > 0
                else 0.0
            ),
            "conditional_error_rate": (
                number_errors / number_samples
                if number_samples > 0
                else 0.0
            ),
            "error_share": (
                number_errors / total_errors
                if total_errors > 0
                else 0.0
            ),
            "macro_precision": float(
                group_class_metrics["precision"].mean()
            ),
            "macro_recall": float(
                group_class_metrics["recall"].mean()
            ),
            "macro_f1": float(
                group_class_metrics["f1"].mean()
            ),
        })

    return pd.DataFrame(summary_rows), details


def save_coarse_confusion_matrix(
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
    path: Path,
) -> None:
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
    pd.DataFrame(
        matrix,
        index=names,
        columns=names,
    ).to_csv(path, index=True)


def main() -> None:
    args = parse_arguments()
    project_root = resolve_project_root(args.project_root)
    paths = resolve_paths(args, project_root)

    mapping = load_mapping(paths["mapping"])
    groups = load_groups(paths["groups"])
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
    group_summary, group_details = build_group_reports(
        predictions,
        groups,
        mapping,
        args.split,
    )

    original_wrong = (
        predictions["true_label"]
        != predictions["predicted_label"]
    )
    coarse_correct = (
        predictions["true_coarse_label"]
        == predictions["predicted_coarse_label"]
    )
    recovered_errors = predictions[
        original_wrong & coarse_correct
    ].copy()
    recovered_errors.insert(
        0,
        "recovered_error_rank",
        range(1, len(recovered_errors) + 1),
    )

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
        "group_summary": (
            paths["output_dir"]
            / f"{prefix}_difficulty_group_summary.csv"
        ),
        "group_details": (
            paths["output_dir"]
            / f"{prefix}_difficulty_class_details.csv"
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
    group_summary.to_csv(
        output_paths["group_summary"],
        index=False,
        float_format="%.6f",
    )
    group_details.to_csv(
        output_paths["group_details"],
        index=False,
        float_format="%.6f",
    )
    save_coarse_confusion_matrix(
        predictions,
        mapping,
        output_paths["coarse_confusion"],
    )

    original_row = summary.iloc[0]
    coarse_row = summary.iloc[1]
    print("Coarse and difficulty-group evaluation complete.")
    print("Split:", args.split)
    print("Predictions:", paths["predictions"])
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