"""Create detailed train/validation/test confusion-matrix analysis tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse train, validation, or test confusion matrices produced "
            "by a classifier evaluation script."
        )
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Existing model output directory containing files such as "
            "train_confusion_matrix.csv."
        ),
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Directory used for generated analysis tables. By default, "
            "<output_dir>/analyze_confusion_matrix is created automatically."
        ),
    )
    parser.add_argument(
        "--split",
        choices=(*SPLITS, "all"),
        default="all",
        help="Dataset split to analyse (default: all).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of wrong destinations stored for each class (default: 5).",
    )
    parser.add_argument(
        "--min-pair-count",
        type=int,
        default=1,
        help=(
            "Only save directed confusion pairs with at least this many "
            "mistakes (default: 1)."
        ),
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")
    if args.min_pair_count <= 0:
        parser.error("--min-pair-count must be positive.")

    return args


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    """Divide arrays while returning zero when the denominator is zero."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def load_confusion_matrix(
    csv_path: Path,
) -> tuple[np.ndarray, list[str]]:
    """Load and validate one labelled square confusion matrix."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Confusion matrix not found: {csv_path}"
        )

    dataframe = pd.read_csv(csv_path, index_col=0)

    if dataframe.empty:
        raise ValueError(f"Confusion matrix is empty: {csv_path}")

    dataframe.index = dataframe.index.astype(str).str.strip()
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    if dataframe.shape[0] != dataframe.shape[1]:
        raise ValueError(
            f"Confusion matrix must be square, got {dataframe.shape}."
        )

    row_names = dataframe.index.tolist()
    column_names = dataframe.columns.tolist()

    if row_names != column_names:
        raise ValueError(
            "Confusion-matrix row and column class names or ordering differ."
        )

    try:
        matrix = dataframe.to_numpy(dtype=np.float64)
    except ValueError as error:
        raise ValueError(
            f"Confusion matrix contains non-numeric values: {csv_path}"
        ) from error

    if not np.isfinite(matrix).all():
        raise ValueError("Confusion matrix contains NaN or infinite values.")
    if (matrix < 0).any():
        raise ValueError("Confusion matrix contains negative values.")

    return matrix, row_names


def calculate_metrics(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate class-level counts and classification metrics."""
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted_total = matrix.sum(axis=0)
    false_negative = support - true_positive
    false_positive = predicted_total - true_positive

    precision = safe_divide(true_positive, predicted_total)
    recall = safe_divide(true_positive, support)
    f1_score = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "support": support,
        "predicted_total": predicted_total,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "error_rate": 1.0 - recall,
    }


def create_per_class_table(
    matrix: np.ndarray,
    class_names: list[str],
    split: str,
    top_k: int,
) -> pd.DataFrame:
    """
    Create one row per true class.

    The top-N fields show where images from that true class were sent.
    wrong_rate is relative to all true samples in the class, while
    share_of_class_errors is relative only to its incorrectly predicted images.
    """
    metrics = calculate_metrics(matrix)
    rows: list[dict[str, int | float | str]] = []

    for true_index, true_name in enumerate(class_names):
        row: dict[str, int | float | str] = {
            "split": split,
            "label": true_index,
            "insect_name": true_name,
            "support": int(metrics["support"][true_index]),
            "true_positive": int(metrics["true_positive"][true_index]),
            "false_negative": int(metrics["false_negative"][true_index]),
            "predicted_total": int(metrics["predicted_total"][true_index]),
            "false_positive": int(metrics["false_positive"][true_index]),
            "precision": metrics["precision"][true_index],
            "recall": metrics["recall"][true_index],
            "f1_score": metrics["f1_score"][true_index],
            "error_rate": metrics["error_rate"][true_index],
        }

        wrong_predictions = [
            (predicted_index, matrix[true_index, predicted_index])
            for predicted_index in range(len(class_names))
            if (
                predicted_index != true_index
                and matrix[true_index, predicted_index] > 0
            )
        ]
        wrong_predictions.sort(key=lambda item: item[1], reverse=True)

        for rank in range(1, top_k + 1):
            if rank <= len(wrong_predictions):
                predicted_index, count = wrong_predictions[rank - 1]
                support = metrics["support"][true_index]
                all_errors = metrics["false_negative"][true_index]

                row[f"top{rank}_wrong_label"] = predicted_index
                row[f"top{rank}_wrong_class"] = class_names[predicted_index]
                row[f"top{rank}_wrong_count"] = int(count)
                row[f"top{rank}_wrong_rate"] = (
                    count / support if support else 0.0
                )
                row[f"top{rank}_share_of_class_errors"] = (
                    count / all_errors if all_errors else 0.0
                )
            else:
                row[f"top{rank}_wrong_label"] = ""
                row[f"top{rank}_wrong_class"] = ""
                row[f"top{rank}_wrong_count"] = 0
                row[f"top{rank}_wrong_rate"] = 0.0
                row[f"top{rank}_share_of_class_errors"] = 0.0

        rows.append(row)

    # Priority rule:
    # 1. Lowest F1 first.
    # 2. Lowest recall first when F1 is equal.
    # 3. Larger support first when both metrics are equal, because the
    #    estimate is based on more evidence.
    result = (
        pd.DataFrame(rows)
        .sort_values(
            by=["f1_score", "recall", "support"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )
    result.insert(
        0,
        "difficulty_rank",
        np.arange(1, len(result) + 1),
    )
    return result


def create_directed_pairs_table(
    matrix: np.ndarray,
    class_names: list[str],
    split: str,
    min_pair_count: int,
) -> pd.DataFrame:
    """Create one row for every true-class to predicted-class error."""
    metrics = calculate_metrics(matrix)
    rows: list[dict[str, int | float | str]] = []

    for true_index, true_name in enumerate(class_names):
        for predicted_index, predicted_name in enumerate(class_names):
            if true_index == predicted_index:
                continue

            count = matrix[true_index, predicted_index]
            if count < min_pair_count:
                continue

            support = metrics["support"][true_index]
            all_errors = metrics["false_negative"][true_index]
            predicted_total = metrics["predicted_total"][predicted_index]

            rows.append(
                {
                    "split": split,
                    "true_label": true_index,
                    "true_class": true_name,
                    "predicted_label": predicted_index,
                    "predicted_class": predicted_name,
                    "wrong_count": int(count),
                    "true_class_support": int(support),
                    "wrong_rate_of_true_class": (
                        count / support if support else 0.0
                    ),
                    "share_of_true_class_errors": (
                        count / all_errors if all_errors else 0.0
                    ),
                    "predicted_class_total": int(predicted_total),
                    "share_of_predicted_class": (
                        count / predicted_total
                        if predicted_total
                        else 0.0
                    ),
                }
            )

    columns = [
        "split",
        "true_label",
        "true_class",
        "predicted_label",
        "predicted_class",
        "wrong_count",
        "true_class_support",
        "wrong_rate_of_true_class",
        "share_of_true_class_errors",
        "predicted_class_total",
        "share_of_predicted_class",
    ]

    # Priority rule:
    # 1. Largest number of wrong predictions first.
    # 2. Largest proportion of the true class first when counts are equal.
    # 3. Largest share of that class's errors first as the final tie-breaker.
    result = (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            by=[
                "wrong_count",
                "wrong_rate_of_true_class",
                "share_of_true_class_errors",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    result.insert(
        0,
        "confusion_rank",
        np.arange(1, len(result) + 1),
    )
    return result


def create_mutual_pairs_table(
    matrix: np.ndarray,
    class_names: list[str],
    split: str,
    min_pair_count: int,
) -> pd.DataFrame:
    """Create one row for every pair confused in both directions."""
    support = matrix.sum(axis=1)
    rows: list[dict[str, int | float | str]] = []

    for class_a in range(len(class_names)):
        for class_b in range(class_a + 1, len(class_names)):
            a_to_b = matrix[class_a, class_b]
            b_to_a = matrix[class_b, class_a]
            combined_count = a_to_b + b_to_a

            if (
                a_to_b < min_pair_count
                or b_to_a < min_pair_count
            ):
                continue

            combined_support = support[class_a] + support[class_b]

            rows.append(
                {
                    "split": split,
                    "class_a_label": class_a,
                    "class_a": class_names[class_a],
                    "class_a_support": int(support[class_a]),
                    "class_b_label": class_b,
                    "class_b": class_names[class_b],
                    "class_b_support": int(support[class_b]),
                    "a_to_b_count": int(a_to_b),
                    "a_to_b_rate": (
                        a_to_b / support[class_a]
                        if support[class_a]
                        else 0.0
                    ),
                    "b_to_a_count": int(b_to_a),
                    "b_to_a_rate": (
                        b_to_a / support[class_b]
                        if support[class_b]
                        else 0.0
                    ),
                    "combined_wrong_count": int(combined_count),
                    "combined_confusion_rate": (
                        combined_count / combined_support
                        if combined_support
                        else 0.0
                    ),
                }
            )

    columns = [
        "split",
        "class_a_label",
        "class_a",
        "class_a_support",
        "class_b_label",
        "class_b",
        "class_b_support",
        "a_to_b_count",
        "a_to_b_rate",
        "b_to_a_count",
        "b_to_a_rate",
        "combined_wrong_count",
        "combined_confusion_rate",
    ]

    # Priority rule:
    # 1. Largest combined A-to-B and B-to-A error count first.
    # 2. Largest combined confusion rate first when counts are equal.
    # 3. Larger combined class support first as the final tie-breaker.
    result = pd.DataFrame(rows, columns=columns)
    result["combined_support"] = (
        result["class_a_support"]
        + result["class_b_support"]
    )
    result = (
        result
        .sort_values(
            by=[
                "combined_wrong_count",
                "combined_confusion_rate",
                "combined_support",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    result.insert(
        0,
        "mutual_confusion_rank",
        np.arange(1, len(result) + 1),
    )
    return result


def create_summary(
    matrix: np.ndarray,
    per_class: pd.DataFrame,
    split: str,
) -> dict[str, int | float | str]:
    """Create a compact machine-readable split summary."""
    total = matrix.sum()
    correct = np.diag(matrix).sum()

    return {
        "split": split,
        "number_classes": len(matrix),
        "number_samples": int(total),
        "number_correct": int(correct),
        "number_wrong": int(total - correct),
        "accuracy": correct / total if total else 0.0,
        "macro_precision": float(per_class["precision"].mean()),
        "macro_recall": float(per_class["recall"].mean()),
        "macro_f1": float(per_class["f1_score"].mean()),
    }


def analyse_split(
    model_output_dir: Path,
    analysis_dir: Path,
    split: str,
    top_k: int,
    min_pair_count: int,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Analyse one split and save all result tables."""
    matrix_path = (
        model_output_dir / f"{split}_confusion_matrix.csv"
    )
    matrix, class_names = load_confusion_matrix(matrix_path)

    per_class = create_per_class_table(
        matrix,
        class_names,
        split,
        top_k,
    )
    directed_pairs = create_directed_pairs_table(
        matrix,
        class_names,
        split,
        min_pair_count,
    )
    mutual_pairs = create_mutual_pairs_table(
        matrix,
        class_names,
        split,
        min_pair_count,
    )
    summary = create_summary(matrix, per_class, split)

    per_class_path = (
        analysis_dir / f"{split}_confusion_per_class.csv"
    )
    directed_pairs_path = (
        analysis_dir / f"{split}_confusion_pairs.csv"
    )
    mutual_pairs_path = (
        analysis_dir / f"{split}_mutual_confusions.csv"
    )
    per_class.to_csv(
        per_class_path,
        index=False,
        float_format="%.6f",
    )
    directed_pairs.to_csv(
        directed_pairs_path,
        index=False,
        float_format="%.6f",
    )
    mutual_pairs.to_csv(
        mutual_pairs_path,
        index=False,
        float_format="%.6f",
    )
    print(f"\n[{split.upper()}]")
    print("Input:", matrix_path)
    print(
        f"Accuracy={summary['accuracy']:.4f}, "
        f"Macro-F1={summary['macro_f1']:.4f}, "
        f"wrong={summary['number_wrong']}/{summary['number_samples']}"
    )
    print("Per-class analysis:", per_class_path)
    print("Directed confusion pairs:", directed_pairs_path)
    print("Mutual confusion pairs:", mutual_pairs_path)
    return per_class, summary


def create_split_comparison(
    split_tables: dict[str, pd.DataFrame],
    analysis_dir: Path,
) -> Path:
    """Combine class metrics from all available splits into one table."""
    comparison: pd.DataFrame | None = None

    for split in SPLITS:
        if split not in split_tables:
            continue

        table = split_tables[split][
            [
                "label",
                "insect_name",
                "support",
                "precision",
                "recall",
                "f1_score",
                "error_rate",
            ]
        ].rename(
            columns={
                column: f"{split}_{column}"
                for column in (
                    "support",
                    "precision",
                    "recall",
                    "f1_score",
                    "error_rate",
                )
            }
        )

        if comparison is None:
            comparison = table
        else:
            comparison = comparison.merge(
                table,
                on=["label", "insect_name"],
                how="outer",
                validate="one_to_one",
            )

    if comparison is None:
        raise ValueError("No split tables were available for comparison.")

    if {"train_f1_score", "val_f1_score"} <= set(comparison.columns):
        comparison["train_val_f1_gap"] = (
            comparison["train_f1_score"]
            - comparison["val_f1_score"]
        )
        comparison["train_val_recall_gap"] = (
            comparison["train_recall"]
            - comparison["val_recall"]
        )

    if {"train_f1_score", "test_f1_score"} <= set(comparison.columns):
        comparison["train_test_f1_gap"] = (
            comparison["train_f1_score"]
            - comparison["test_f1_score"]
        )

    if {"val_f1_score", "test_f1_score"} <= set(comparison.columns):
        comparison["val_test_f1_difference"] = (
            comparison["val_f1_score"]
            - comparison["test_f1_score"]
        )

    # Priority rule for the cross-split table:
    # 1. When train and validation are present, put the largest
    #    train-to-validation F1 generalisation gap first.
    # 2. Otherwise, put the lowest-F1 class in the available split first.
    if "train_val_f1_gap" in comparison.columns:
        comparison = comparison.sort_values(
            by=[
                "train_val_f1_gap",
                "val_f1_score",
                "val_support",
            ],
            ascending=[False, True, False],
        )
    else:
        available_f1_columns = [
            f"{split}_f1_score"
            for split in ("val", "test", "train")
            if f"{split}_f1_score" in comparison.columns
        ]
        comparison = comparison.sort_values(
            by=[available_f1_columns[0], "label"],
            ascending=[True, True],
        )

    comparison = comparison.reset_index(drop=True)
    comparison.insert(
        0,
        "comparison_rank",
        np.arange(1, len(comparison) + 1),
    )

    path = analysis_dir / "confusion_split_comparison.csv"
    comparison.to_csv(path, index=False, float_format="%.6f")
    return path


def main() -> None:
    args = parse_arguments()
    model_output_dir = args.output_dir.expanduser().resolve()

    if not model_output_dir.is_dir():
        raise NotADirectoryError(
            f"Model output directory not found: {model_output_dir}"
        )

    if args.analysis_dir is None:
        analysis_dir = (
            model_output_dir / "analyze_confusion_matrix"
        )
    else:
        analysis_dir = args.analysis_dir.expanduser().resolve()

    analysis_dir.mkdir(parents=True, exist_ok=True)

    requested_splits = (
        SPLITS
        if args.split == "all"
        else (args.split,)
    )

    missing = [
        model_output_dir / f"{split}_confusion_matrix.csv"
        for split in requested_splits
        if not (
            model_output_dir / f"{split}_confusion_matrix.csv"
        ).is_file()
    ]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required confusion matrices were not found:\n"
            f"{missing_text}\n"
            "Evaluate the corresponding splits before running this analysis."
        )

    split_tables: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, int | float | str]] = {}

    for split in requested_splits:
        table, summary = analyse_split(
            model_output_dir=model_output_dir,
            analysis_dir=analysis_dir,
            split=split,
            top_k=args.top_k,
            min_pair_count=args.min_pair_count,
        )
        split_tables[split] = table
        summaries[split] = summary

    comparison_path = create_split_comparison(
        split_tables,
        analysis_dir,
    )

    # Summary rows always follow the fixed train -> val -> test order.
    summary_dataframe = pd.DataFrame(
        [
            summaries[split]
            for split in SPLITS
            if split in summaries
        ]
    )
    summary_dataframe.insert(
        0,
        "summary_order",
        np.arange(1, len(summary_dataframe) + 1),
    )
    summary_path = (
        analysis_dir / "confusion_analysis_summary.csv"
    )
    summary_dataframe.to_csv(
        summary_path,
        index=False,
        float_format="%.6f",
    )

    print("\nAnalysis output directory:", analysis_dir)
    print("Cross-split comparison:", comparison_path)
    print("Combined CSV summary:", summary_path)


if __name__ == "__main__":
    main()