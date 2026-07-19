from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# 1. Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classifier"
    / "resnet50"
)

TRAINING_HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history.csv"
)

LOSS_CURVE_PATH = (
    OUTPUT_DIR
    / "training_loss_curve.png"
)

ACCURACY_CURVE_PATH = (
    OUTPUT_DIR
    / "training_accuracy_curve.png"
)


# 2. Load training history

def load_training_history(
    csv_path: Path,
) -> pd.DataFrame:
    """Load and validate the training-history CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training history CSV not found:\n{csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    required_columns = {
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training-history CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Training-history CSV contains no epochs."
        )

    dataframe = dataframe.sort_values(
        by="epoch"
    ).reset_index(drop=True)

    return dataframe


# 3. Plot loss

def plot_loss_curves(
    history: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot training and validation loss."""

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

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
        "ResNet50 Training and Validation Loss"
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")

    axis.set_xticks(history["epoch"])

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


# 4. Plot accuracy

def plot_accuracy_curves(
    history: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot training and validation accuracy."""

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

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
        "ResNet50 Training and Validation Accuracy"
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")

    axis.set_xticks(history["epoch"])
    axis.set_ylim(0.0, 1.0)

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


# 5. Main

def main() -> None:
    history = load_training_history(
        TRAINING_HISTORY_PATH
    )

    plot_loss_curves(
        history=history,
        output_path=LOSS_CURVE_PATH,
    )

    plot_accuracy_curves(
        history=history,
        output_path=ACCURACY_CURVE_PATH,
    )

    print(
        "Loss curve saved to:",
        LOSS_CURVE_PATH,
    )

    print(
        "Accuracy curve saved to:",
        ACCURACY_CURVE_PATH,
    )


if __name__ == "__main__":
    main()