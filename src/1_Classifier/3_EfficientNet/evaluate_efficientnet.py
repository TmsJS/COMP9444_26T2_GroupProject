"""Evaluate an EfficientNet-B3 checkpoint on an IP102 dataset split."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


# Expected location:
# COMP9444_Group/src/1_Classifier/3_EfficientNet/evaluate_efficientnet.py
# parents[3] therefore points to COMP9444_Group.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ip102_dataset import IP102Dataset


NUM_CLASSES = 102
MODEL_ARCHITECTURE = "efficientnet_b3"
MODEL_DROPOUT = 0.3
IMAGE_SIZE = 300

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "Classification"
    / "ip102_v1.1"
)
IMAGES_DIR = DATA_DIR / "images"
CLASSES_PATH = DATA_DIR.parent / "classes.txt"


def parse_arguments() -> argparse.Namespace:
    """Read the checkpoint and evaluation options from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate an EfficientNet-B3 checkpoint on IP102.",
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the EfficientNet-B3 checkpoint (.pth file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for evaluation results. By default, results are "
            "saved directly beside the checkpoint."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "test"),
        default="test",
        help=(
            "Dataset split to evaluate: train, val, or test "
            "(default: test)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4).",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    return args


def select_device() -> torch.device:
    """Use CUDA first, then Apple MPS, and otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def create_output_paths(
    model_path: Path,
    requested_output_dir: Path | None,
    split: str,
) -> dict[str, Path]:
    """Create split-specific paths in the model output directory."""
    if requested_output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = requested_output_dir.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "summary": output_dir / f"{split}_summary.txt",
        "report": output_dir / f"{split}_classification_report.csv",
        "predictions": output_dir / f"{split}_predictions.csv",
        "probabilities": output_dir / f"{split}_probabilities.npz",
        "confusion_matrix": output_dir / f"{split}_confusion_matrix.csv",
    }


def load_insect_names(classes_path: Path) -> dict[int, str]:
    """Load insect names and convert labels from 1-based to 0-based."""
    if not classes_path.is_file():
        raise FileNotFoundError(
            f"Class-name file not found: {classes_path}"
        )

    insect_names = {}

    with classes_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            class_id, insect_name = line.split(maxsplit=1)
            insect_names[int(class_id) - 1] = insect_name

    missing_labels = [
        label
        for label in range(NUM_CLASSES)
        if label not in insect_names
    ]

    if missing_labels:
        raise ValueError(
            "classes.txt is missing zero-based labels: "
            f"{missing_labels}"
        )

    return insect_names


def create_evaluation_loader(
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[IP102Dataset, DataLoader]:
    """Create a deterministic DataLoader for one IP102 split."""
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Unsupported evaluation split: {split}"
        )

    evaluation_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    evaluation_dataset = IP102Dataset(
        images_dir=IMAGES_DIR,
        annotation_file=DATA_DIR / f"{split}.txt",
        transform=evaluation_transform,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    return evaluation_dataset, evaluation_loader


def create_efficientnet_b3_model(device: torch.device) -> nn.Module:
    """Recreate exactly the EfficientNet-B3 architecture used in training."""
    model = efficientnet_b3(weights=None)
    number_of_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=MODEL_DROPOUT),
        nn.Linear(number_of_features, NUM_CLASSES),
    )

    return model.to(device)


def load_checkpoint(
    model_path: Path,
    model: nn.Module,
    device: torch.device,
) -> dict:
    """Load a training checkpoint and restore the model parameters."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint must be a dictionary.")

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain 'model_state_dict'."
        )

    checkpoint_architecture = checkpoint.get("architecture")

    if (
        checkpoint_architecture is not None
        and checkpoint_architecture != MODEL_ARCHITECTURE
    ):
        raise ValueError(
            "Checkpoint architecture mismatch: "
            f"expected {MODEL_ARCHITECTURE}, "
            f"found {checkpoint_architecture}."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return checkpoint


def evaluate(
    model: nn.Module,
    evaluation_loader: DataLoader,
    device: torch.device,
    split: str,
) -> tuple[float, list[int], list[int], np.ndarray]:
    """Run inference once and return labels, predictions, and probabilities."""
    criterion = nn.CrossEntropyLoss()
    all_predictions: list[int] = []
    all_labels: list[int] = []
    probability_batches: list[np.ndarray] = []
    total_loss = 0.0
    number_samples = 0

    with torch.inference_mode():
        progress_bar = tqdm(
            evaluation_loader,
            desc=f"Evaluating {split}",
        )

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            number_samples += labels.size(0)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            probability_batches.append(
                probabilities.cpu().numpy()
            )

    if number_samples == 0:
        raise ValueError(
            f"The {split} dataset is empty."
        )

    evaluation_loss = total_loss / number_samples
    all_probabilities = np.concatenate(
        probability_batches,
        axis=0,
    ).astype(np.float32, copy=False)

    expected_shape = (number_samples, NUM_CLASSES)
    if all_probabilities.shape != expected_shape:
        raise RuntimeError(
            "Unexpected probability-array shape: "
            f"expected {expected_shape}, "
            f"received {all_probabilities.shape}."
        )

    return (
        evaluation_loss,
        all_labels,
        all_predictions,
        all_probabilities,
    )


def format_checkpoint_value(checkpoint: dict, key: str) -> str:
    """Format optional checkpoint metadata without raising KeyError."""
    value = checkpoint.get(key)

    if value is None:
        return "Not available"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def build_summary(
    checkpoint: dict,
    model_path: Path,
    output_dir: Path,
    device: torch.device,
    split: str,
    evaluation_loss: float,
    all_labels: list[int],
    all_predictions: list[int],
) -> str:
    """Calculate aggregate metrics and format the text summary."""
    evaluation_accuracy = accuracy_score(all_labels, all_predictions)
    evaluation_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    evaluation_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    evaluation_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    evaluation_g_mean = float(
        geometric_mean_score(
            all_labels,
            all_predictions,
            average="multiclass",
        )
    )

    split_name = {
        "train": "Training",
        "val": "Validation",
        "test": "Test",
    }[split]

    return "\n".join([
        f"Checkpoint: {model_path}",
        f"Evaluation output: {output_dir}",
        f"Dataset split: {split}",
        f"Device: {device}",
        "Model: EfficientNet-B3",
        f"Architecture: {MODEL_ARCHITECTURE}",
        "",
        (
            "Best checkpoint epoch: "
            f"{format_checkpoint_value(checkpoint, 'epoch')}"
        ),
        (
            "Best validation accuracy: "
            f"{format_checkpoint_value(checkpoint, 'val_accuracy')}"
        ),
        (
            "Best validation Macro-F1: "
            f"{format_checkpoint_value(checkpoint, 'val_macro_f1')}"
        ),
        "",
        f"{split_name} loss: {evaluation_loss:.4f}",
        f"{split_name} accuracy: {evaluation_accuracy:.4f}",
        f"{split_name} macro precision: {evaluation_precision:.4f}",
        f"{split_name} macro recall: {evaluation_recall:.4f}",
        f"{split_name} macro-F1: {evaluation_macro_f1:.4f}",
        f"{split_name} GM: {evaluation_g_mean:.4f}",
    ])


def save_per_class_report(
    output_path: Path,
    insect_names: dict[int, str],
    all_labels: list[int],
    all_predictions: list[int],
) -> None:
    """Save precision, recall, F1, and support for every class."""
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        output_dict=True,
        zero_division=0,
    )
    report_rows = []

    for label in range(NUM_CLASSES):
        class_metrics = report[str(label)]
        report_rows.append({
            "label": label,
            "insect_name": insect_names[label],
            "precision": class_metrics["precision"],
            "recall": class_metrics["recall"],
            "f1-score": class_metrics["f1-score"],
            "support": int(class_metrics["support"]),
        })

    pd.DataFrame(report_rows).to_csv(
        output_path,
        index=False,
        float_format="%.4f",
    )


def save_predictions(
    output_path: Path,
    evaluation_dataset: IP102Dataset,
    all_labels: list[int],
    all_predictions: list[int],
) -> None:
    """Save one row for every evaluated image."""
    if not hasattr(evaluation_dataset, "samples"):
        raise AttributeError(
            "IP102Dataset must expose a 'samples' attribute to save filenames."
        )

    image_names = [
        image_name
        for image_name, _ in evaluation_dataset.samples
    ]

    if len(image_names) != len(all_labels):
        raise ValueError(
            "The number of image names does not match the predictions."
        )

    pd.DataFrame({
        "image_name": image_names,
        "true_label": all_labels,
        "predicted_label": all_predictions,
    }).to_csv(output_path, index=False)


def save_probabilities(
    output_path: Path,
    evaluation_dataset: IP102Dataset,
    all_labels: list[int],
    all_probabilities: np.ndarray,
) -> None:
    """Save image names, labels, and N-by-102 class probabilities."""
    if not hasattr(evaluation_dataset, "samples"):
        raise AttributeError(
            "IP102Dataset must expose a 'samples' attribute to save "
            "probabilities."
        )

    image_names = np.asarray(
        [
            str(image_name)
            for image_name, _ in evaluation_dataset.samples
        ],
        dtype=str,
    )
    true_labels = np.asarray(
        all_labels,
        dtype=np.int64,
    )
    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float32,
    )

    number_samples = len(true_labels)
    expected_shape = (number_samples, NUM_CLASSES)

    if len(image_names) != number_samples:
        raise ValueError(
            "The number of image names does not match the labels: "
            f"{len(image_names)} != {number_samples}."
        )

    if probabilities.shape != expected_shape:
        raise ValueError(
            "The probability array has an unexpected shape: "
            f"expected {expected_shape}, "
            f"received {probabilities.shape}."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "The probability array contains NaN or infinite values."
        )

    if np.any(probabilities < 0):
        raise ValueError(
            "The probability array contains negative values."
        )

    row_sums = probabilities.sum(axis=1)
    if not np.allclose(
        row_sums,
        np.ones(number_samples, dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            "Each probability row must sum to one."
        )

    np.savez_compressed(
        output_path,
        image_names=image_names,
        true_labels=true_labels,
        probabilities=probabilities,
    )


def save_confusion_matrix(
    output_path: Path,
    insect_names: dict[int, str],
    all_labels: list[int],
    all_predictions: list[int],
) -> None:
    """Save the 102-by-102 confusion matrix with insect names."""
    confusion = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )
    class_names = [
        insect_names[label]
        for label in range(NUM_CLASSES)
    ]

    pd.DataFrame(
        confusion,
        index=class_names,
        columns=class_names,
    ).to_csv(
        output_path,
        index=True,
        index_label="true_class",
    )


def main() -> None:
    """Load one checkpoint, evaluate it, and save all result files."""
    args = parse_arguments()
    model_path = args.model_path.expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    paths = create_output_paths(
        model_path=model_path,
        requested_output_dir=args.output_dir,
        split=args.split,
    )
    device = select_device()
    insect_names = load_insect_names(CLASSES_PATH)

    evaluation_dataset, evaluation_loader = create_evaluation_loader(
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model = create_efficientnet_b3_model(device)
    checkpoint = load_checkpoint(model_path, model, device)

    (
        evaluation_loss,
        all_labels,
        all_predictions,
        all_probabilities,
    ) = evaluate(
        model=model,
        evaluation_loader=evaluation_loader,
        device=device,
        split=args.split,
    )

    summary = build_summary(
        checkpoint=checkpoint,
        model_path=model_path,
        output_dir=paths["output_dir"],
        device=device,
        split=args.split,
        evaluation_loss=evaluation_loss,
        all_labels=all_labels,
        all_predictions=all_predictions,
    )

    print()
    print(summary)
    paths["summary"].write_text(summary + "\n", encoding="utf-8")

    save_per_class_report(
        paths["report"],
        insect_names,
        all_labels,
        all_predictions,
    )
    save_predictions(
        paths["predictions"],
        evaluation_dataset,
        all_labels,
        all_predictions,
    )
    save_probabilities(
        paths["probabilities"],
        evaluation_dataset,
        all_labels,
        all_probabilities,
    )
    save_confusion_matrix(
        paths["confusion_matrix"],
        insect_names,
        all_labels,
        all_predictions,
    )

    print()
    print(f"{args.split.capitalize()} summary saved to:", paths["summary"])
    print("Classification report saved to:", paths["report"])
    print("Predictions saved to:", paths["predictions"])
    print("Probabilities saved to:", paths["probabilities"])
    print("Confusion matrix saved to:", paths["confusion_matrix"])


if __name__ == "__main__":
    main()
