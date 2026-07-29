"""Plot per-class IP102 metrics produced by evaluate_resnet.py."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Read the model output directory and plotting options."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-class classification metrics "
            "for an IP102 model."
        ),
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Model output directory containing a split-specific "
            "classification-report CSV."
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "test"),
        default="test",
        help=(
            "Dataset split whose classification metrics should "
            "be plotted: train, val, or test (default: test)."
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Display name used in chart titles. By default, the output "
            "directory name is used."
        ),
    )
    parser.add_argument(
        "--lowest-count",
        type=int,
        default=20,
        help="Number of lowest-F1 classes to plot (default: 20).",
    )

    args = parser.parse_args()

    if args.lowest_count <= 0:
        parser.error("--lowest-count must be positive.")

    return args


def create_paths(
    output_dir: Path,
    split: str,
) -> dict[str, Path]:
    """Create split-specific input and output paths."""

    output_dir = (
        output_dir
        .expanduser()
        .resolve()
    )

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"Model output directory not found: {output_dir}"
        )

    return {
        "output_dir": output_dir,
        "classification_report": (
            output_dir
            / f"{split}_classification_report.csv"
        ),
        "all_class_metrics": (
            output_dir
            / f"{split}_class_precision_recall_f1.png"
        ),
        "lowest_f1": (
            output_dir
            / f"{split}_lowest_f1_classes.png"
        ),
        "class_support": (
            output_dir
            / f"{split}_class_support.png"
        ),
    }

def create_default_model_name(output_dir: Path) -> str:
    """Convert a directory name into a readable chart title."""
    known_names = {
        "resnet50": "ResNet50",
        "resnet50_imbalance": "Imbalance-Aware ResNet50",
    }

    return known_names.get(
        output_dir.name,
        output_dir.name.replace("_", " ").title(),
    )


def load_classification_report(csv_path: Path) -> pd.DataFrame:
    """Load and validate per-class classification metrics."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Classification report not found: {csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    required_columns = {
        "label",
        "insect_name",
        "precision",
        "recall",
        "f1-score",
        "support",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Classification-report CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("Classification-report CSV is empty.")

    numeric_columns = [
        "label",
        "precision",
        "recall",
        "f1-score",
        "support",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    dataframe["insect_name"] = (
        dataframe["insect_name"]
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe.sort_values(
        by="label",
    ).reset_index(drop=True)

    return dataframe


def plot_all_class_metrics(
    report: pd.DataFrame,
    output_path: Path,
    model_name: str,
    split_name: str,
) -> None:
    """Plot precision, recall, and F1 for all classes."""
    labels = report["label"].to_numpy()
    precision = report["precision"].to_numpy()
    recall = report["recall"].to_numpy()
    f1_scores = report["f1-score"].to_numpy()

    figure, axis = plt.subplots(figsize=(24, 8))

    axis.plot(
        labels,
        precision,
        label="Precision",
        linewidth=1.2,
    )
    axis.plot(
        labels,
        recall,
        label="Recall",
        linewidth=1.2,
    )
    axis.plot(
        labels,
        f1_scores,
        label="F1-score",
        linewidth=1.5,
    )

    axis.set_title(f"{model_name} {split_name} Metrics by IP102 Class")
    axis.set_xlabel("Class label")
    axis.set_ylabel("Metric value")
    axis.set_xlim(labels.min(), labels.max())
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(
        np.arange(
            labels.min(),
            labels.max() + 1,
            5,
        )
    )
    axis.grid(visible=True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_lowest_f1_classes(
    report: pd.DataFrame,
    output_path: Path,
    model_name: str,
    split_name: str,
    number_classes: int,
) -> None:
    """Plot the classes with the lowest F1 scores."""
    number_classes = min(number_classes, len(report))

    lowest_classes = (
        report
        .nsmallest(number_classes, "f1-score")
        .sort_values(by="f1-score", ascending=False)
    )

    display_names = [
        f"{int(row.label)}: {row.insect_name}"
        for row in lowest_classes.itertuples()
    ]

    figure, axis = plt.subplots(figsize=(12, 10))

    axis.barh(
        display_names,
        lowest_classes["f1-score"],
    )

    axis.set_title(
        f"{model_name} {split_name} Lowest "
        f"{len(lowest_classes)} Per-Class F1-Scores"
    )
    axis.set_xlabel("F1-score")
    axis.set_ylabel("Class")
    axis.set_xlim(0.0, 1.0)
    axis.grid(
        visible=True,
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_class_support(
    report: pd.DataFrame,
    output_path: Path,
    split_name: str,
) -> None:
    """Plot the number of images in every class for one split."""
    labels = report["label"].to_numpy()
    support = report["support"].to_numpy()

    figure, axis = plt.subplots(figsize=(24, 8))

    axis.bar(labels, support)
    axis.set_title(f"IP102 {split_name} Support by Class")
    axis.set_xlabel("Class label")
    axis.set_ylabel(f"Number of {split_name.lower()} images")
    axis.set_xticks(
        np.arange(
            labels.min(),
            labels.max() + 1,
            5,
        )
    )
    axis.grid(
        visible=True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def print_metric_summary(
    report: pd.DataFrame,
    split_name: str,
) -> None:
    """Print the five best and worst classes by F1 score."""

    best_classes = report.nlargest(
        5,
        "f1-score",
    )

    worst_classes = report.nsmallest(
        5,
        "f1-score",
    )

    print(
        f"\nFive highest-F1 classes "
        f"on the {split_name.lower()} set:"
    )

    for _, row in best_classes.iterrows():
        print(
            f"  {int(row['label']):3d} "
            f"{row['insect_name']}: "
            f"{row['f1-score']:.4f}"
        )

    print(
        f"\nFive lowest-F1 classes "
        f"on the {split_name.lower()} set:"
    )

    for _, row in worst_classes.iterrows():
        print(
            f"  {int(row['label']):3d} "
            f"{row['insect_name']}: "
            f"{row['f1-score']:.4f}"
        )


def main() -> None:
    args = parse_arguments()

    paths = create_paths(
        output_dir=args.output_dir,
        split=args.split,
    )

    if args.model_name is None:
        model_name = create_default_model_name(
            paths["output_dir"]
        )
    else:
        model_name = args.model_name.strip()

        if not model_name:
            raise ValueError(
                "--model-name cannot be empty."
            )

    split_name = {
        "train": "Training",
        "val": "Validation",
        "test": "Test",
    }[args.split]

    report = load_classification_report(
        paths["classification_report"]
    )

    plot_all_class_metrics(
        report=report,
        output_path=paths["all_class_metrics"],
        model_name=model_name,
        split_name=split_name,
    )

    plot_lowest_f1_classes(
        report=report,
        output_path=paths["lowest_f1"],
        model_name=model_name,
        split_name=split_name,
        number_classes=args.lowest_count,
    )

    plot_class_support(
        report=report,
        output_path=paths["class_support"],
        split_name=split_name,
    )

    print_metric_summary(
        report=report,
        split_name=split_name,
    )

    print("\nModel:", model_name)
    print("Dataset split:", args.split)
    print(
        "Classification report:",
        paths["classification_report"],
    )
    print(
        "All-class metrics saved to:",
        paths["all_class_metrics"],
    )
    print(
        "Lowest-F1 chart saved to:",
        paths["lowest_f1"],
    )
    print(
        "Class-support chart saved to:",
        paths["class_support"],
    )

if __name__ == "__main__":
    main()