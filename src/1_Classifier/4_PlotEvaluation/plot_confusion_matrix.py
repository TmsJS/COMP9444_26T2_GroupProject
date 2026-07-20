"""Plot raw and normalized confusion matrices from a model output folder."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Read the model output directory and optional display name."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw and normalized confusion matrices for a ResNet50 model."
        ),
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Existing model output directory containing "
            "test_confusion_matrix.csv."
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

    return parser.parse_args()


def create_paths(output_dir: Path) -> dict[str, Path]:
    """Create all file paths directly inside one model output directory."""
    output_dir = output_dir.expanduser().resolve()

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"Model output directory not found: {output_dir}"
        )

    return {
        "output_dir": output_dir,
        "confusion_matrix": output_dir / "test_confusion_matrix.csv",
        "raw_figure": output_dir / "test_confusion_matrix.png",
        "normalized_figure": (
            output_dir / "test_confusion_matrix_normalized.png"
        ),
    }


def create_default_model_name(output_dir: Path) -> str:
    """Convert a known output-directory name into a chart display name."""
    known_names = {
        "resnet50": "ResNet50",
        "resnet50_imbalance": "Imbalance-Aware ResNet50",
    }

    return known_names.get(
        output_dir.name,
        output_dir.name.replace("_", " ").title(),
    )


def load_confusion_matrix(
    csv_path: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load and validate a square confusion-matrix CSV.

    The first row contains predicted-class names, and the first column
    contains true-class names.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Confusion matrix CSV not found: {csv_path}"
        )

    dataframe = pd.read_csv(
        csv_path,
        index_col=0,
    )

    if dataframe.empty:
        raise ValueError(
            f"Confusion matrix CSV is empty: {csv_path}"
        )

    dataframe.index = (
        dataframe.index
        .astype(str)
        .str.strip()
    )
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    try:
        matrix = dataframe.to_numpy(dtype=np.float64)
    except ValueError as error:
        raise ValueError(
            "The confusion matrix contains non-numeric values."
        ) from error

    if matrix.ndim != 2:
        raise ValueError(
            "The confusion matrix must be two-dimensional."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "The confusion matrix must be square, "
            f"but its shape is {matrix.shape}."
        )

    row_names = dataframe.index.tolist()
    column_names = dataframe.columns.tolist()

    if len(row_names) != matrix.shape[0]:
        raise ValueError(
            "The number of row names does not match "
            "the number of matrix rows."
        )

    if len(column_names) != matrix.shape[1]:
        raise ValueError(
            "The number of column names does not match "
            "the number of matrix columns."
        )

    if row_names != column_names:
        print(
            "Warning: row class names and column class names "
            "are not exactly identical or are in a different order."
        )

    print("Loaded confusion matrix:", matrix.shape)

    return matrix, row_names, column_names


def normalize_by_true_class(matrix: np.ndarray) -> np.ndarray:
    """Normalize every row by its number of true-class samples."""
    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=row_totals != 0,
    )


def plot_matrix(
    matrix: np.ndarray,
    row_names: list[str],
    column_names: list[str],
    output_path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    """Plot and save one confusion-matrix heatmap."""
    number_classes = matrix.shape[0]

    figure, axis = plt.subplots(figsize=(32, 28))

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto",
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(
        colorbar_label,
        fontsize=14,
    )

    axis.set_title(
        title,
        fontsize=20,
        pad=20,
    )
    axis.set_xlabel(
        "Predicted class",
        fontsize=16,
        labelpad=15,
    )
    axis.set_ylabel(
        "True class",
        fontsize=16,
        labelpad=15,
    )

    class_positions = np.arange(number_classes)

    axis.set_xticks(class_positions)
    axis.set_yticks(class_positions)
    axis.set_xticklabels(
        column_names,
        rotation=90,
        fontsize=4.5,
    )
    axis.set_yticklabels(
        row_names,
        fontsize=4.5,
    )

    axis.set_xticks(
        np.arange(-0.5, number_classes, 1),
        minor=True,
    )
    axis.set_yticks(
        np.arange(-0.5, number_classes, 1),
        minor=True,
    )
    axis.grid(
        which="minor",
        linewidth=0.1,
    )
    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    paths = create_paths(args.output_dir)

    if args.model_name is None:
        model_name = create_default_model_name(paths["output_dir"])
    else:
        model_name = args.model_name.strip()

        if not model_name:
            raise ValueError("--model-name cannot be empty.")

    matrix, row_names, column_names = load_confusion_matrix(
        paths["confusion_matrix"]
    )
    normalized_matrix = normalize_by_true_class(matrix)

    plot_matrix(
        matrix=matrix,
        row_names=row_names,
        column_names=column_names,
        output_path=paths["raw_figure"],
        title=f"{model_name} Test Confusion Matrix",
        colorbar_label="Number of test images",
    )
    plot_matrix(
        matrix=normalized_matrix,
        row_names=row_names,
        column_names=column_names,
        output_path=paths["normalized_figure"],
        title=(
            f"{model_name} Test Confusion Matrix "
            "(Normalized by True Class)"
        ),
        colorbar_label="Proportion of true-class images",
    )

    print("Model:", model_name)
    print("Confusion-matrix CSV:", paths["confusion_matrix"])
    print("Raw confusion-matrix figure:", paths["raw_figure"])
    print(
        "Normalized confusion-matrix figure:",
        paths["normalized_figure"],
    )


if __name__ == "__main__":
    main()