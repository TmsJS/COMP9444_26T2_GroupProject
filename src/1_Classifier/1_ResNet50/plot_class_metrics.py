from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classifier"
    / "resnet50"
)

CLASSIFICATION_REPORT_PATH = (
    OUTPUT_DIR
    / "test_classification_report.csv"
)

ALL_CLASS_METRICS_PATH = (
    OUTPUT_DIR
    / "class_precision_recall_f1.png"
)

LOWEST_F1_PATH = (
    OUTPUT_DIR
    / "lowest_f1_classes.png"
)

CLASS_SUPPORT_PATH = (
    OUTPUT_DIR
    / "class_support.png"
)


# 2. Load classification report

def load_classification_report(
    csv_path: Path,
) -> pd.DataFrame:
    """Load and validate per-class classification metrics."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Classification report not found:\n{csv_path}"
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

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Classification-report CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Classification-report CSV is empty."
        )

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
        by="label"
    ).reset_index(drop=True)

    return dataframe


# 3. Plot all class metrics

def plot_all_class_metrics(
    report: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot precision, recall and F1 for all classes.

    Numeric class labels are used because 102 insect names
    would make the horizontal axis unreadable.
    """

    labels = report["label"].to_numpy()
    precision = report["precision"].to_numpy()
    recall = report["recall"].to_numpy()
    f1_scores = report["f1-score"].to_numpy()

    figure, axis = plt.subplots(
        figsize=(24, 8)
    )

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

    axis.set_title(
        "ResNet50 Test Metrics by IP102 Class"
    )

    axis.set_xlabel("Class label")
    axis.set_ylabel("Metric value")

    axis.set_xlim(
        labels.min(),
        labels.max(),
    )

    axis.set_ylim(0.0, 1.0)

    axis.set_xticks(
        np.arange(
            labels.min(),
            labels.max() + 1,
            5,
        )
    )

    axis.grid(
        visible=True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# 4. Plot lowest-F1 classes

def plot_lowest_f1_classes(
    report: pd.DataFrame,
    output_path: Path,
    number_classes: int = 20,
) -> None:
    """Plot the classes with the lowest F1-scores."""

    lowest_classes = (
        report
        .nsmallest(
            number_classes,
            "f1-score",
        )
        .sort_values(
            by="f1-score",
            ascending=True,
        )
    )

    display_names = [
        f"{int(row.label)}: {row.insect_name}"
        for row in lowest_classes.itertuples()
    ]

    figure, axis = plt.subplots(
        figsize=(12, 10)
    )

    axis.barh(
        display_names,
        lowest_classes["f1-score"],
    )

    axis.set_title(
        f"ResNet50 Lowest {len(lowest_classes)} "
        "Per-Class F1-Scores"
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


# 5. Plot class support

def plot_class_support(
    report: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the number of test images in every class."""

    labels = report["label"].to_numpy()
    support = report["support"].to_numpy()

    figure, axis = plt.subplots(
        figsize=(24, 8)
    )

    axis.bar(
        labels,
        support,
    )

    axis.set_title(
        "IP102 Test Support by Class"
    )

    axis.set_xlabel("Class label")
    axis.set_ylabel("Number of test images")

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


# 6. Print best and worst classes

def print_metric_summary(
    report: pd.DataFrame,
) -> None:
    """Print the five best and five worst classes by F1-score."""

    best_classes = report.nlargest(
        5,
        "f1-score",
    )

    worst_classes = report.nsmallest(
        5,
        "f1-score",
    )

    print("\nFive highest-F1 classes:")

    for _, row in best_classes.iterrows():
        print(
            f"  {int(row['label']):3d} "
            f"{row['insect_name']}: "
            f"{row['f1-score']:.4f}"
        )

    print("\nFive lowest-F1 classes:")

    for _, row in worst_classes.iterrows():
        print(
            f"  {int(row['label']):3d} "
            f"{row['insect_name']}: "
            f"{row['f1-score']:.4f}"
        )

# 7. Main

def main() -> None:
    report = load_classification_report(
        CLASSIFICATION_REPORT_PATH
    )

    plot_all_class_metrics(
        report=report,
        output_path=ALL_CLASS_METRICS_PATH,
    )

    plot_lowest_f1_classes(
        report=report,
        output_path=LOWEST_F1_PATH,
        number_classes=20,
    )

    plot_class_support(
        report=report,
        output_path=CLASS_SUPPORT_PATH,
    )

    print_metric_summary(report)

    print(
        "\nAll-class metrics saved to:",
        ALL_CLASS_METRICS_PATH,
    )

    print(
        "Lowest-F1 chart saved to:",
        LOWEST_F1_PATH,
    )

    print(
        "Class-support chart saved to:",
        CLASS_SUPPORT_PATH,
    )


if __name__ == "__main__":
    main()