"""Final training script for ImageNet-pretrained EfficientNet-B3 on IP102."""

import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


# Expected location:
# COMP9444_Group/src/1_Classifier/2_efficientnet/train_efficientnet.py
# parents[3] therefore points to COMP9444_Group.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ip102_dataset import IP102Dataset


NUM_CLASSES = 102
BATCH_SIZE = 16
MAX_EPOCHS = 40
LEARNING_RATE = 0.01
NUM_WORKERS = 4
LR_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 6
RANDOM_SEED = 42

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "Classification"
    / "ip102_v1.1"
)
IMAGES_DIR = DATA_DIR / "images"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classifier" / "efficientnet_b3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "efficientnet_b3_best_model.pth"
TRAINING_HISTORY_PATH = OUTPUT_DIR / "training_history.csv"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def create_logger() -> tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"efficientnet_b3_trainlog_{timestamp}.txt"

    logger = logging.getLogger("efficientnet_b3_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_path


def create_data_loaders(
    device: torch.device,
) -> tuple[IP102Dataset, IP102Dataset, DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.Resize(
            (300, 300),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(
            (300, 300),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = IP102Dataset(
        images_dir=IMAGES_DIR,
        annotation_file=DATA_DIR / "train.txt",
        transform=train_transform,
    )
    val_dataset = IP102Dataset(
        images_dir=IMAGES_DIR,
        annotation_file=DATA_DIR / "val.txt",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def create_model(device: torch.device) -> nn.Module:
    model = efficientnet_b3(
        weights=EfficientNet_B3_Weights.IMAGENET1K_V1,
    )
    number_of_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(number_of_features, NUM_CLASSES),
    )

    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    number_correct = 0
    number_samples = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        number_correct += (predictions == labels).sum().item()
        number_samples += labels.size(0)
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / number_samples, number_correct / number_samples


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    number_correct = 0
    number_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            number_correct += (predictions == labels).sum().item()
            number_samples += labels.size(0)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    g_mean = float(
        geometric_mean_score(
            all_labels,
            all_predictions,
            average="multiclass",
        )
    )

    return (
        total_loss / number_samples,
        number_correct / number_samples,
        macro_f1,
        g_mean,
    )


def main() -> None:
    set_random_seed(RANDOM_SEED)
    device = select_device()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dataset, val_dataset, train_loader, val_loader = (
        create_data_loaders(device)
    )
    model = create_model(device)
    criterion = nn.CrossEntropyLoss()

    # Keep the baseline ResNet50 optimizer settings for a direct comparison.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=LR_PATIENCE,
        min_lr=1e-6,
    )

    logger, train_log_path = create_logger()
    logger.info("Model: EfficientNet-B3")
    logger.info(f"Device: {device}")
    logger.info(f"Training images: {len(train_dataset)}")
    logger.info(f"Validation images: {len(val_dataset)}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Maximum epochs: {MAX_EPOCHS}")
    logger.info(f"Best model path: {BEST_MODEL_PATH}")
    logger.info(f"Training log path: {train_log_path}")

    best_macro_f1 = 0.0
    epochs_without_improvement = 0
    training_history = []

    for epoch in range(MAX_EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_accuracy, val_macro_f1, val_g_mean = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        current_learning_rate = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch [{epoch + 1}/{MAX_EPOCHS}] | "
            f"LR: {current_learning_rate:.6f} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f} | "
            f"Val macro-F1: {val_macro_f1:.4f} | "
            f"Val GM: {val_g_mean:.4f}"
        )

        training_history.append({
            "epoch": epoch + 1,
            "learning_rate": current_learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_g_mean": val_g_mean,
        })
        pd.DataFrame(training_history).to_csv(
            TRAINING_HISTORY_PATH,
            index=False,
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "architecture": "efficientnet_b3",
                    "weights": "IMAGENET1K_V1",
                    "image_size": 300,
                    "num_classes": NUM_CLASSES,
                    "dropout": 0.3,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_macro_f1": val_macro_f1,
                    "val_g_mean": val_g_mean,
                },
                BEST_MODEL_PATH,
            )
            logger.info(
                "Saved new best model "
                f"(Macro-F1: {best_macro_f1:.4f})"
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "Epochs without Macro-F1 improvement: "
                f"{epochs_without_improvement}"
            )

        previous_learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(val_macro_f1)
        new_learning_rate = optimizer.param_groups[0]["lr"]

        if new_learning_rate < previous_learning_rate:
            logger.info(
                "Learning rate reduced: "
                f"{previous_learning_rate:.6f} -> {new_learning_rate:.6f}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            logger.info(
                "Early stopping: validation Macro-F1 did not improve for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    logger.info("Training complete")
    logger.info(f"Best validation Macro-F1: {best_macro_f1:.4f}")
    logger.info(f"Best model path: {BEST_MODEL_PATH}")
    logger.info(f"Training log path: {train_log_path}")
    logger.info(f"Training history path: {TRAINING_HISTORY_PATH}")


if __name__ == "__main__":
    main()