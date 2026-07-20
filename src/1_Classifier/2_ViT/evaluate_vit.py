"""Evaluate a trained DeiT-III Base/16 checkpoint on the IP102 test set."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import timm
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
from tqdm import tqdm


# Expected location:
# COMP9444_Group/src/1_Classifier/3_deit/evaluate_vit.py
# parents[3] therefore points to COMP9444_Group.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ip102_dataset import IP102Dataset


NUM_CLASSES = 102
MODEL_ARCHITECTURE = "deit3_base_patch16_224.fb_in1k"
MODEL_DROPOUT = 0.1

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
        description="Evaluate a DeiT-III Base/16 checkpoint on IP102.",
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the DeiT-III checkpoint (.pth file).",
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
        "--batch-size",
        type=int,
        default=16,
        help="Test batch size (default: 16).",
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
) -> dict[str, Path]:
    """Save evaluation results in the checkpoint directory by default."""
    if requested_output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = requested_output_dir.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "summary": output_dir / "test_summary.txt",
        "report": output_dir / "test_classification_report.csv",
        "predictions": output_dir / "test_predictions.csv",
        "confusion_matrix": output_dir / "test_confusion_matrix.csv",
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


def create_test_loader(
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[IP102Dataset, DataLoader]:
    """Create the unchanged IP102 test dataset and sequential DataLoader."""
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = IP102Dataset(
        images_dir=IMAGES_DIR,
        annotation_file=DATA_DIR / "test.txt",
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    return test_dataset, test_loader


def create_deit3_model(device: torch.device) -> nn.Module:
    """Recreate exactly the DeiT-III architecture used during training."""
    model = timm.create_model(
        MODEL_ARCHITECTURE,
        pretrained=False,
        num_classes=NUM_CLASSES,
        drop_rate=MODEL_DROPOUT,
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
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Run inference once and return loss, labels, and predictions."""
    criterion = nn.CrossEntropyLoss()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    number_samples = 0

    with torch.inference_mode():
        progress_bar = tqdm(test_loader, desc="Testing")

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            number_samples += labels.size(0)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if number_samples == 0:
        raise ValueError("The test dataset is empty.")

    return total_loss / number_samples, all_labels, all_predictions


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
    test_loss: float,
    all_labels: list[int],
    all_predictions: list[int],
) -> str:
    """Calculate aggregate metrics and format the text summary."""
    test_accuracy = accuracy_score(all_labels, all_predictions)
    test_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    test_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    test_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    test_g_mean = float(
        geometric_mean_score(
            all_labels,
            all_predictions,
            average="multiclass",
        )
    )

    return "\n".join([
        f"Checkpoint: {model_path}",
        f"Evaluation output: {output_dir}",
        f"Device: {device}",
        "Model: DeiT-III Base/16",
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
        f"Test loss: {test_loss:.4f}",
        f"Test accuracy: {test_accuracy:.4f}",
        f"Test macro precision: {test_precision:.4f}",
        f"Test macro recall: {test_recall:.4f}",
        f"Test macro-F1: {test_macro_f1:.4f}",
        f"Test GM: {test_g_mean:.4f}",
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
    test_dataset: IP102Dataset,
    all_labels: list[int],
    all_predictions: list[int],
) -> None:
    """Save one row for every test image."""
    if not hasattr(test_dataset, "samples"):
        raise AttributeError(
            "IP102Dataset must expose a 'samples' attribute to save filenames."
        )

    image_names = [
        image_name
        for image_name, _ in test_dataset.samples
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

    paths = create_output_paths(model_path, args.output_dir)
    device = select_device()
    insect_names = load_insect_names(CLASSES_PATH)
    test_dataset, test_loader = create_test_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    model = create_deit3_model(device)
    checkpoint = load_checkpoint(model_path, model, device)
    test_loss, all_labels, all_predictions = evaluate(
        model,
        test_loader,
        device,
    )

    summary = build_summary(
        checkpoint=checkpoint,
        model_path=model_path,
        output_dir=paths["output_dir"],
        device=device,
        test_loss=test_loss,
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
        test_dataset,
        all_labels,
        all_predictions,
    )
    save_confusion_matrix(
        paths["confusion_matrix"],
        insect_names,
        all_labels,
        all_predictions,
    )

    print()
    print("Test summary saved to:", paths["summary"])
    print("Classification report saved to:", paths["report"])
    print("Predictions saved to:", paths["predictions"])
    print("Confusion matrix saved to:", paths["confusion_matrix"])


if __name__ == "__main__":
    main()