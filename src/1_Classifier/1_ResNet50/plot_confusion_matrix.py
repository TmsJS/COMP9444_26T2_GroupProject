from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. Project paths

# Current file:
# COMP9444_Group/src/1_Classifier/1_ResNet50/
# plot_confusion_matrix.py
#
# parents[3] points to:
# COMP9444_Group/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classifier"
    / "resnet50"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR
    / "test_confusion_matrix.csv"
)

RAW_FIGURE_PATH = (
    OUTPUT_DIR
    / "test_confusion_matrix.png"
)

NORMALIZED_FIGURE_PATH = (
    OUTPUT_DIR
    / "test_confusion_matrix_normalized.png"
)


# 2. Load confusion matrix

def load_confusion_matrix(
    csv_path: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load a confusion matrix CSV.

    Expected format:

        ,class_a,class_b,class_c
        class_a,10,2,1
        class_b,3,20,4
        class_c,0,1,15

    The first row contains predicted class names.
    The first column contains true class names.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Confusion matrix CSV not found:\n{csv_path}"
        )

    # The first column contains the true-class names.
    dataframe = pd.read_csv(
        csv_path,
        index_col=0,
    )

    if dataframe.empty:
        raise ValueError(
            f"Confusion matrix CSV is empty:\n{csv_path}"
        )

    # Remove accidental whitespace around class names.
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
        matrix = dataframe.to_numpy(
            dtype=np.float64,
        )
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

    print(
        "Loaded confusion matrix:",
        matrix.shape,
    )

    return matrix, row_names, column_names


# 3. Normalize confusion matrix

def normalize_by_true_class(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Normalize each row by its total number of true samples.

    Each row therefore sums to approximately 1.0.
    """

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=row_totals != 0,
    )

    return normalized_matrix


# 4. Plot confusion matrix

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

    figure, axis = plt.subplots(
        figsize=(32, 28),
    )

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

    # Draw cell boundaries.
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# 5. Main

def main() -> None:
    matrix, row_names, column_names = load_confusion_matrix(
        CONFUSION_MATRIX_PATH
    )

    normalized_matrix = normalize_by_true_class(
        matrix
    )

    plot_matrix(
        matrix=matrix,
        row_names=row_names,
        column_names=column_names,
        output_path=RAW_FIGURE_PATH,
        title="ResNet50 Test Confusion Matrix",
        colorbar_label="Number of test images",
    )

    plot_matrix(
        matrix=normalized_matrix,
        row_names=row_names,
        column_names=column_names,
        output_path=NORMALIZED_FIGURE_PATH,
        title=(
            "ResNet50 Test Confusion Matrix "
            "(Normalized by True Class)"
        ),
        colorbar_label="Proportion of true-class images",
    )

    print(
        "Raw confusion matrix (used for checking actual wrong predictions per class) saved to:",
        RAW_FIGURE_PATH,
    )

    print(
        "Normalized confusion matrix (used for checking recall rate per class) saved to:",
        NORMALIZED_FIGURE_PATH,
    )


if __name__ == "__main__":
    main()