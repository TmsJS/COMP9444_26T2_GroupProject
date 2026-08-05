"""
Define and evaluate IP102 class-difficulty groups using validation data only.

This script is the second step of the class-separability workflow:

1. Read the automatically discovered validation clusters and retain clusters
   whose visual_confusion_score meets the configured threshold.
2. Freeze those clusters in selected_clusters.csv.
3. Divide all 102 original classes into cluster_hard, easy, diffuse_hard,
   and uncertain using validation predictions only.
4. Save the per-class definitions and a validation difficulty-group summary.

The summary also contains non_cluster_total, which is the union of easy,
diffuse_hard, and uncertain and serves as the direct comparison group for
cluster_hard. It overlaps with those three subgroup rows and must not be added
to them.

This script does not run a neural network and does not create coarse labels.
The selected clusters and class-difficulty definitions must be frozen before
any test-set evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)


NUM_CLASSES = 102
DEFAULT_MINIMUM_VISUAL_SCORE = 50.0
DEFAULT_EASY_MIN_SUPPORT = 20
DEFAULT_EASY_MIN_F1 = 0.70
DEFAULT_EASY_MIN_RECALL = 0.70
DEFAULT_EASY_MAX_DOMINANT_CONFUSION = 0.15
DEFAULT_HARD_MAX_F1 = 0.60
DEFAULT_HARD_MAX_RECALL = 0.60

DIFFICULTY_GROUP_ORDER = {
    "easy": 1,
    "cluster_hard": 2,
    "diffuse_hard": 3,
    "uncertain": 4,
}
SUMMARY_GROUP_ORDER = (
    "cluster_hard",
    "non_cluster_total",
    "easy",
    "diffuse_hard",
    "uncertain",
)


def parse_arguments() -> argparse.Namespace:
    """Read validation inputs, thresholds, and output locations."""
    parser = argparse.ArgumentParser(
        description=(
            "Define and evaluate validation-derived IP102 "
            "class-difficulty groups."
        ),
    )
    parser.add_argument(
        "model_output_dir",
        type=Path,
        help=(
            "Baseline model output directory containing "
            "val_predictions.csv."
        ),
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
        "--cluster-csv",
        type=Path,
        default=None,
        help=(
            "Automatically discovered validation-cluster CSV. Default: "
            "<model_output_dir>/analyze_confusion_matrix/"
            "val_discovered_confusion_clusters.csv."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Validation prediction CSV override. Default: "
            "<model_output_dir>/val_predictions.csv."
        ),
    )
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=None,
        help=(
            "Directory for selected_clusters.csv and "
            "class_difficulty_groups.csv. Default: "
            "outputs/classifier/class_separability."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for val_difficulty_group_summary.csv. Default: "
            "<model_output_dir>/difficulty_group_analysis."
        ),
    )
    parser.add_argument(
        "--minimum-visual-score",
        type=float,
        default=DEFAULT_MINIMUM_VISUAL_SCORE,
    )
    parser.add_argument(
        "--easy-min-support",
        type=int,
        default=DEFAULT_EASY_MIN_SUPPORT,
    )
    parser.add_argument(
        "--easy-min-f1",
        type=float,
        default=DEFAULT_EASY_MIN_F1,
    )
    parser.add_argument(
        "--easy-min-recall",
        type=float,
        default=DEFAULT_EASY_MIN_RECALL,
    )
    parser.add_argument(
        "--easy-max-dominant-confusion",
        type=float,
        default=DEFAULT_EASY_MAX_DOMINANT_CONFUSION,
    )
    parser.add_argument(
        "--hard-max-f1",
        type=float,
        default=DEFAULT_HARD_MAX_F1,
    )
    parser.add_argument(
        "--hard-max-recall",
        type=float,
        default=DEFAULT_HARD_MAX_RECALL,
    )

    args = parser.parse_args()
    probability_arguments = (
        args.easy_min_f1,
        args.easy_min_recall,
        args.easy_max_dominant_confusion,
        args.hard_max_f1,
        args.hard_max_recall,
    )

    if args.minimum_visual_score < 0:
        parser.error("--minimum-visual-score cannot be negative.")
    if args.easy_min_support < 0:
        parser.error("--easy-min-support cannot be negative.")
    if any(value < 0 or value > 1 for value in probability_arguments):
        parser.error(
            "F1, recall, and confusion thresholds must be in [0, 1]."
        )
    return args


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

    required = (
        project_root
        / "datasets"
        / "raw"
        / "Classification"
        / "classes.txt"
    )
    if not required.is_file():
        raise FileNotFoundError(
            f"Invalid project root or missing classes.txt: {project_root}"
        )
    return project_root


def resolve_paths(
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Path]:
    """Resolve and validate all validation inputs and outputs."""
    model_output_dir = resolve_path(
        args.model_output_dir,
        project_root,
    )
    if not model_output_dir.is_dir():
        raise NotADirectoryError(
            f"Model output directory not found: {model_output_dir}"
        )

    cluster_csv = (
        model_output_dir
        / "analyze_confusion_matrix"
        / "val_discovered_confusion_clusters.csv"
        if args.cluster_csv is None
        else resolve_path(args.cluster_csv, project_root)
    )
    predictions = (
        model_output_dir / "val_predictions.csv"
        if args.predictions is None
        else resolve_path(args.predictions, project_root)
    )
    definitions_dir = (
        project_root
        / "outputs"
        / "classifier"
        / "class_separability"
        if args.definitions_dir is None
        else resolve_path(args.definitions_dir, project_root)
    )
    output_dir = (
        model_output_dir / "difficulty_group_analysis"
        if args.output_dir is None
        else resolve_path(args.output_dir, project_root)
    )

    missing = [
        path
        for path in (cluster_csv, predictions)
        if not path.is_file()
    ]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required validation inputs are missing:\n"
            f"{missing_text}"
        )

    definitions_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "classes": (
            project_root
            / "datasets"
            / "raw"
            / "Classification"
            / "classes.txt"
        ),
        "cluster_csv": cluster_csv,
        "predictions": predictions,
        "definitions_dir": definitions_dir,
        "output_dir": output_dir,
        "selected_clusters": definitions_dir / "selected_clusters.csv",
        "difficulty_groups": (
            definitions_dir / "class_difficulty_groups.csv"
        ),
        "summary": output_dir / "val_difficulty_group_summary.csv",
    }


def load_class_names(classes_path: Path) -> list[str]:
    """Load classes.txt and convert its one-based IDs to zero-based order."""
    label_to_name: dict[int, str] = {}

    with classes_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                class_number, class_name = stripped.split(maxsplit=1)
            except ValueError as error:
                raise ValueError(
                    f"Invalid classes.txt row {line_number}: {line!r}"
                ) from error

            label = int(class_number) - 1
            if label in label_to_name:
                raise ValueError(
                    f"Duplicate class number: {class_number}"
                )
            label_to_name[label] = class_name.strip()

    expected = set(range(NUM_CLASSES))
    observed = set(label_to_name)
    if observed != expected:
        raise ValueError(
            "classes.txt must contain labels 1-102 exactly. "
            f"Missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    class_names = [
        label_to_name[label]
        for label in range(NUM_CLASSES)
    ]
    if len(set(class_names)) != NUM_CLASSES:
        raise ValueError("classes.txt contains duplicate class names.")
    return class_names


def parse_members(value: object) -> list[str]:
    """Parse a pipe-separated cluster-member field."""
    if pd.isna(value):
        return []
    return [
        member.strip()
        for member in str(value).split("|")
        if member.strip()
    ]


def select_validation_clusters(
    csv_path: Path,
    class_names: list[str],
    minimum_visual_score: float,
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Select and validate candidate clusters using validation scores only."""
    dataframe = pd.read_csv(csv_path)
    required = {
        "cluster_name",
        "class_members",
        "visual_confusion_score",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Cluster CSV is missing columns: {sorted(missing)}"
        )

    dataframe["visual_confusion_score"] = pd.to_numeric(
        dataframe["visual_confusion_score"],
        errors="raise",
    )
    if "split" in dataframe.columns:
        dataframe = dataframe[
            dataframe["split"].astype(str).str.lower() == "val"
        ].copy()
        if dataframe.empty:
            raise ValueError("Cluster CSV contains no validation rows.")

    selected = dataframe[
        dataframe["visual_confusion_score"] >= minimum_visual_score
    ].copy()
    if selected.empty:
        raise ValueError(
            "No validation clusters meet the visual-score threshold."
        )

    sort_columns = (
        ["visual_cluster_rank", "visual_confusion_score"]
        if "visual_cluster_rank" in selected.columns
        else ["visual_confusion_score"]
    )
    ascending = (
        [True, False]
        if "visual_cluster_rank" in selected.columns
        else [False]
    )
    selected = (
        selected
        .sort_values(sort_columns, ascending=ascending)
        .reset_index(drop=True)
    )
    if "split" not in selected.columns:
        selected.insert(0, "split", "val")

    name_to_label = {
        name: label
        for label, name in enumerate(class_names)
    }
    clusters: dict[str, list[int]] = {}
    used_labels: dict[int, str] = {}

    for row_number, row in selected.iterrows():
        cluster_name = str(row["cluster_name"]).strip()
        member_names = parse_members(row["class_members"])

        if not cluster_name:
            raise ValueError(
                f"Empty cluster name at row {row_number + 1}."
            )
        if cluster_name in clusters:
            raise ValueError(
                f"Duplicate cluster name: {cluster_name}"
            )
        if len(member_names) < 2:
            raise ValueError(
                f"Cluster {cluster_name!r} needs at least two classes."
            )

        unknown = [
            name
            for name in member_names
            if name not in name_to_label
        ]
        if unknown:
            raise ValueError(
                f"Unknown classes in {cluster_name!r}: {unknown}"
            )

        labels = [name_to_label[name] for name in member_names]
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"Duplicate member inside {cluster_name!r}."
            )
        for label in labels:
            if label in used_labels:
                raise ValueError(
                    f"{class_names[label]!r} belongs to both "
                    f"{used_labels[label]!r} and {cluster_name!r}."
                )
            used_labels[label] = cluster_name

        clusters[cluster_name] = labels

    return selected, clusters


def load_validation_predictions(path: Path) -> pd.DataFrame:
    """Load and validate original 102-class validation predictions."""
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
        if (~dataframe[column].between(0, NUM_CLASSES - 1)).any():
            raise ValueError(
                f"{column} contains labels outside 0-101."
            )
    return dataframe


def calculate_per_class_metrics(
    predictions: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    """Calculate complete-split metrics for all 102 original classes."""
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
        "class_name": class_names,
        "support": support.astype(int),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


def build_difficulty_groups(
    predictions: pd.DataFrame,
    per_class: pd.DataFrame,
    class_names: list[str],
    clusters: dict[str, list[int]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Assign every fine class to one frozen validation-derived group."""
    labels = list(range(NUM_CLASSES))
    confusion = confusion_matrix(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=labels,
    )
    label_to_cluster = {
        label: cluster_name
        for cluster_name, members in clusters.items()
        for label in members
    }
    rows: list[dict[str, int | float | str | bool]] = []

    for row in per_class.itertuples(index=False):
        label = int(row.label)
        support = int(confusion[label].sum())
        if support != int(row.support):
            raise RuntimeError(
                f"Support mismatch for label {label}: "
                f"metrics={row.support}, confusion={support}."
            )

        off_diagonal = confusion[label].copy()
        off_diagonal[label] = 0
        dominant_count = int(off_diagonal.max())
        dominant_label = int(off_diagonal.argmax())
        dominant_rate = (
            dominant_count / support
            if support
            else 0.0
        )

        cluster_name = label_to_cluster.get(label, "")
        if cluster_name:
            difficulty_group = "cluster_hard"
            reason = "member_of_selected_visual_cluster"
        elif (
            support >= args.easy_min_support
            and float(row.f1) >= args.easy_min_f1
            and float(row.recall) >= args.easy_min_recall
            and dominant_rate < args.easy_max_dominant_confusion
        ):
            difficulty_group = "easy"
            reason = "stable_high_validation_performance"
        elif (
            float(row.f1) < args.hard_max_f1
            or float(row.recall) < args.hard_max_recall
        ):
            difficulty_group = "diffuse_hard"
            reason = "low_validation_metric_without_selected_cluster"
        else:
            difficulty_group = "uncertain"
            reason = "intermediate_or_low_support"

        rows.append({
            "difficulty_group_order": DIFFICULTY_GROUP_ORDER[
                difficulty_group
            ],
            "label": label,
            "class_number": label + 1,
            "class_name": row.class_name,
            "cluster_name": cluster_name,
            "is_selected_cluster_member": bool(cluster_name),
            "val_support": support,
            "val_precision": float(row.precision),
            "val_recall": float(row.recall),
            "val_f1": float(row.f1),
            "dominant_wrong_label": (
                dominant_label if dominant_count else -1
            ),
            "dominant_wrong_class": (
                class_names[dominant_label]
                if dominant_count
                else ""
            ),
            "dominant_wrong_count": dominant_count,
            "dominant_confusion_rate": dominant_rate,
            "difficulty_group": difficulty_group,
            "group_reason": reason,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["difficulty_group_order", "val_f1", "label"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )


def build_group_summary(
    predictions: pd.DataFrame,
    per_class: pd.DataFrame,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise validation performance for cluster and comparison groups."""
    label_to_group = groups.set_index("label")[
        "difficulty_group"
    ].to_dict()
    sample_groups = predictions["true_label"].map(label_to_group)
    if sample_groups.isna().any():
        raise ValueError(
            "Some validation labels have no difficulty-group definition."
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
    report_labels = {
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
        SUMMARY_GROUP_ORDER,
        start=1,
    ):
        labels = report_labels[group_name]
        mask = (
            sample_groups != "cluster_hard"
            if group_name == "non_cluster_total"
            else sample_groups == group_name
        )
        selected_predictions = predictions.loc[mask]
        number_samples = len(selected_predictions)
        number_correct = int(
            (
                selected_predictions["true_label"]
                == selected_predictions["predicted_label"]
            ).sum()
        )
        number_errors = number_samples - number_correct
        selected_metrics = per_class[
            per_class["label"].isin(labels)
        ]

        rows.append({
            "group_order": group_order,
            "split": "val",
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
                selected_metrics["precision"].mean()
            ),
            "macro_recall": float(
                selected_metrics["recall"].mean()
            ),
            "macro_f1": float(
                selected_metrics["f1"].mean()
            ),
        })

    return pd.DataFrame(rows)


def main() -> None:
    """Select clusters, define groups, evaluate validation, and save CSVs."""
    args = parse_arguments()
    project_root = resolve_project_root(args.project_root)
    paths = resolve_paths(args, project_root)

    class_names = load_class_names(paths["classes"])
    selected_clusters, clusters = select_validation_clusters(
        paths["cluster_csv"],
        class_names,
        args.minimum_visual_score,
    )
    predictions = load_validation_predictions(paths["predictions"])
    per_class = calculate_per_class_metrics(
        predictions,
        class_names,
    )
    difficulty_groups = build_difficulty_groups(
        predictions,
        per_class,
        class_names,
        clusters,
        args,
    )
    group_summary = build_group_summary(
        predictions,
        per_class,
        difficulty_groups,
    )

    selected_clusters.to_csv(
        paths["selected_clusters"],
        index=False,
        float_format="%.6f",
    )
    difficulty_groups.to_csv(
        paths["difficulty_groups"],
        index=False,
        float_format="%.6f",
    )
    group_summary.to_csv(
        paths["summary"],
        index=False,
        float_format="%.6f",
    )

    counts = (
        difficulty_groups["difficulty_group"]
        .value_counts()
        .to_dict()
    )
    print("Validation difficulty-group analysis completed successfully.")
    print("Selected clusters:", len(clusters))
    print(
        "Clustered classes:",
        sum(len(members) for members in clusters.values()),
    )
    print("Difficulty groups:", counts)
    print("Selected clusters:", paths["selected_clusters"])
    print("Difficulty definitions:", paths["difficulty_groups"])
    print("Validation summary:", paths["summary"])


if __name__ == "__main__":
    main()
