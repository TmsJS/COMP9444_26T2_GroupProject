"""Plot ResNet50 training history from a model output folder."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Read the model output directory and optional display name."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot training and validation curves for a ResNet50 model."
        ),
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Existing model output directory containing "
            "training_history.csv."
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
        "training_history": output_dir / "training_history.csv",
        "loss_curve": output_dir / "training_loss_curve.png",
        "accuracy_curve": output_dir / "training_accuracy_curve.png",
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


def load_training_history(csv_path: Path) -> pd.DataFrame:
    """Load and validate the training-history CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Training history CSV not found: {csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    required_columns = {
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Training-history CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Training-history CSV contains no epochs."
        )

    numeric_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    dataframe = dataframe.sort_values(
        by="epoch",
    ).reset_index(drop=True)

    return dataframe


def plot_loss_curves(
    history: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    """Plot training and validation loss."""
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Training loss",
    )
    axis.plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        label="Validation loss",
    )

    axis.set_title(
        f"{model_name} Training and Validation Loss"
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_xticks(history["epoch"])
    axis.grid(visible=True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_accuracy_curves(
    history: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    """Plot training and validation accuracy."""
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        history["epoch"],
        history["train_accuracy"],
        marker="o",
        label="Training accuracy",
    )
    axis.plot(
        history["epoch"],
        history["val_accuracy"],
        marker="o",
        label="Validation accuracy",
    )

    axis.set_title(
        f"{model_name} Training and Validation Accuracy"
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(history["epoch"])
    axis.set_ylim(0.0, 1.0)
    axis.grid(visible=True, alpha=0.3)
    axis.legend()

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

    history = load_training_history(paths["training_history"])

    plot_loss_curves(
        history=history,
        output_path=paths["loss_curve"],
        model_name=model_name,
    )
    plot_accuracy_curves(
        history=history,
        output_path=paths["accuracy_curve"],
        model_name=model_name,
    )

    print("Model:", model_name)
    print("Training history:", paths["training_history"])
    print("Loss curve saved to:", paths["loss_curve"])
    print("Accuracy curve saved to:", paths["accuracy_curve"])


if __name__ == "__main__":
    main()