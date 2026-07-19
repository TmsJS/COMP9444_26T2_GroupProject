'''
The imbalance-aware model is trained on the IP102 training set with explicit imbalance-specific processing.
The original class distribution is modified only during training through two-sided resampling.
1.(Oversampling)
Minority Classes with fewer than 150 training images are oversampled to approximately 150 samples per epoch.
2.(Undersampling)
Majority Classes with more than 2500 training images are undersampled to approximately 2500 samples per epoch.
Majority-class undersampling is performed dynamically during training.
3.(Normalsampling)
Classes containing between 150 and 2500 training images preserve their original sampling frequency.
4.(Class Weights)
Still keep applyingthe standard CrossEntropyLoss same as in (base)train_resnet.py, 
but Class-weighted CrossEntropyLoss maybe considered in future experiments, 
with Inverse-square-root class weights are computed from the original training class frequencies.
5.(Custom Sampler)
A custom class-aware sampler replaces ordinary random mini-batch sampling.
6.(Data Augmentation)
Minority classes(before oversampling) receive stronger online data augmentation, 
including random horizontal flip, small-angle rotation, and mild colour jitter,
random resized crop are not adopted, cause it may cut antenna, wings).
Non-Minority Classes with at least 150 training images retain the baseline augmentation: random horizontal flip.
All augmented images are generated online during training and are not saved to local storage.
7.(Validation Sets)
Validation sets remain unchanged and are evaluated on the original data distribution.
No oversampling, undersampling, or random augmentation is applied to validation test data.
'''

import logging
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


# This file is intended to be located at:
# COMP9444_Group/src/1_Classifier/1_resnet50/imbalance_train_resnet.py
# parents[3] therefore points to COMP9444_Group.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ip102_dataset import IP102Dataset


# 1. Training configuration

NUM_CLASSES = 102
BATCH_SIZE = 64
MAX_EPOCHS = 40
LEARNING_RATE = 0.01
NUM_WORKERS = 4
LR_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 6

# Two-sided resampling thresholds.
MIN_SAMPLES_PER_CLASS = 150
MAX_SAMPLES_PER_CLASS = 2500

# Class-weight limits after normalization.
MIN_CLASS_WEIGHT = 0.5
MAX_CLASS_WEIGHT = 3.0

# Reproducibility.
RANDOM_SEED = 42


# 2. Project and dataset paths

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "Classification"
    / "ip102_v1.1"
)

IMAGES_DIR = DATA_DIR / "images"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classifier"
    / "resnet50_imbalance"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "resnet50_imbalance_best_model.pth"
TRAINING_HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
RESAMPLING_SUMMARY_PATH = OUTPUT_DIR / "resampling_summary.csv"
CLASS_WEIGHTS_PATH = OUTPUT_DIR / "class_weights.csv"


def set_random_seed(seed: int) -> None:
    """Set random seeds used by Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set the seed before datasets, DataLoaders, and model are created.
set_random_seed(RANDOM_SEED)

def create_logger() -> tuple[logging.Logger, Path]:
    """Create a logger that writes to the terminal and a timestamped file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"resnet50_imbalance_trainlog_{timestamp}.txt"

    logger = logging.getLogger("resnet50_imbalance_training")
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


# 3. Image preprocessing and online augmentation

# Baseline-style augmentation for classes with at least 150 training images.
standard_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Stronger online augmentation for minority classes with fewer than 150 images.
minority_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15,
        hue=0.02,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Validation data remains unchanged and receives no random augmentation.
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# 4. Dataset wrappers and label extraction

def extract_labels(dataset: Dataset) -> list[int]:
    """
    Extract class labels without assuming one exact IP102Dataset attribute name.

    The function first checks common label attributes. If none exist, it falls
    back to reading labels through __getitem__. The fallback may be slower
    because it may load each image once.
    """
    possible_attributes = (
        "labels",
        "targets",
        "y",
    )

    for attribute_name in possible_attributes:
        if hasattr(dataset, attribute_name):
            values = getattr(dataset, attribute_name)
            return [int(value) for value in values]

    if hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")
        return [int(sample[1]) for sample in samples]

    labels = []

    for index in tqdm(
        range(len(dataset)),
        desc="Reading training labels",
        leave=False,
    ):
        _, label = dataset[index]
        labels.append(int(label))

    return labels


class ClassAwareTransformDataset(Dataset):
    """
    Apply stronger online augmentation only to minority-class samples.

    The underlying dataset is expected to return an untransformed PIL image
    and its integer class label.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        original_class_counts: dict[int, int],
        minority_threshold: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.original_class_counts = original_class_counts
        self.minority_threshold = minority_threshold

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        label = int(label)

        if self.original_class_counts[label] < self.minority_threshold:
            image = minority_train_transform(image)
        else:
            image = standard_train_transform(image)

        return image, label


# 5. Exact two-sided resampling

class TwoSidedClassSampler(Sampler[int]):
    """
    Construct one resampled epoch with exact per-class target counts.

    Rules:
    original count < 150:
        sample with replacement until the class contributes 150 samples

    150 <= original count <= 2500:
        keep every original sample once

    original count > 2500:
        sample 2500 distinct examples without replacement

    The sampled index order is shuffled at the end of every epoch.
    """

    def __init__(
        self,
        labels: list[int],
        minimum_count: int,
        maximum_count: int,
        seed: int,
    ) -> None:
        if minimum_count <= 0:
            raise ValueError("minimum_count must be positive.")

        if maximum_count < minimum_count:
            raise ValueError(
                "maximum_count must be greater than or equal to minimum_count."
            )

        self.minimum_count = minimum_count
        self.maximum_count = maximum_count
        self.seed = seed
        self.epoch = 0

        indices_by_class: dict[int, list[int]] = defaultdict(list)

        for index, label in enumerate(labels):
            indices_by_class[int(label)].append(index)

        self.indices_by_class = dict(indices_by_class)

        self.target_counts = {
            label: min(
                max(len(indices), self.minimum_count),
                self.maximum_count,
            )
            for label, indices in self.indices_by_class.items()
        }

        self.epoch_length = sum(self.target_counts.values())

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number so each epoch receives a different sample."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.epoch_length

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        sampled_indices: list[int] = []

        for label in sorted(self.indices_by_class):
            class_indices = self.indices_by_class[label]
            original_count = len(class_indices)
            target_count = self.target_counts[label]

            class_index_tensor = torch.tensor(
                class_indices,
                dtype=torch.long,
            )

            if target_count < original_count:
                # Majority-class undersampling without replacement.
                selected_positions = torch.randperm(
                    original_count,
                    generator=generator,
                )[:target_count]

                selected_indices = class_index_tensor[
                    selected_positions
                ].tolist()

            elif target_count > original_count:
                # Keep every original image once, then sample the remainder
                # with replacement. This avoids accidentally omitting rare
                # original samples from an epoch.
                additional_count = target_count - original_count

                additional_positions = torch.randint(
                    low=0,
                    high=original_count,
                    size=(additional_count,),
                    generator=generator,
                )

                selected_indices = (
                    class_indices
                    + class_index_tensor[
                        additional_positions
                    ].tolist()
                )

            else:
                selected_indices = list(class_indices)

            sampled_indices.extend(selected_indices)

        final_order = torch.randperm(
            len(sampled_indices),
            generator=generator,
        ).tolist()

        shuffled_indices = [
            sampled_indices[position]
            for position in final_order
        ]

        return iter(shuffled_indices)


# 6. Create datasets, sampler, and DataLoaders

# Training images are deliberately loaded without a transform first.
# ClassAwareTransformDataset chooses the transform after reading the label.
raw_train_dataset = IP102Dataset(
    images_dir=IMAGES_DIR,
    annotation_file=DATA_DIR / "train.txt",
    transform=None,
)

training_labels = extract_labels(raw_train_dataset)

original_class_counts = {
    label: training_labels.count(label)
    for label in sorted(set(training_labels))
}

if len(original_class_counts) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} training classes, "
        f"but found {len(original_class_counts)}."
    )

train_dataset = ClassAwareTransformDataset(
    base_dataset=raw_train_dataset,
    original_class_counts=original_class_counts,
    minority_threshold=MIN_SAMPLES_PER_CLASS,
)

train_sampler = TwoSidedClassSampler(
    labels=training_labels,
    minimum_count=MIN_SAMPLES_PER_CLASS,
    maximum_count=MAX_SAMPLES_PER_CLASS,
    seed=RANDOM_SEED,
)

val_dataset = IP102Dataset(
    images_dir=IMAGES_DIR,
    annotation_file=DATA_DIR / "val.txt",
    transform=val_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# 7. Class-weighted loss

def create_class_weights(
    class_counts: dict[int, int],
    number_of_classes: int,
) -> torch.Tensor:
    """
    Create inverse-square-root class weights from original train frequencies.

    The raw weights are normalized to mean 1 and then clipped to [0.5, 3.0].
    The final mean may differ slightly from 1 after clipping.
    """
    counts = torch.tensor(
        [
            class_counts[class_id]
            for class_id in range(number_of_classes)
        ],
        dtype=torch.float32,
    )

    weights = torch.rsqrt(counts)
    weights = weights / weights.mean()
    weights = torch.clamp(
        weights,
        min=MIN_CLASS_WEIGHT,
        max=MAX_CLASS_WEIGHT,
    )

    # Re-normalize after clipping so that the average loss scale remains
    # approximately comparable with ordinary CrossEntropyLoss.
    weights = weights / weights.mean()

    return weights

class_weights_cpu = create_class_weights(
    original_class_counts,
    NUM_CLASSES,
)


# 8. Save resampling and weighting configuration

def save_imbalance_configuration() -> None:
    """Save original counts, effective counts, and loss weights."""
    rows = []

    for class_id in range(NUM_CLASSES):
        original_count = original_class_counts[class_id]
        effective_count = train_sampler.target_counts[class_id]

        if original_count < MIN_SAMPLES_PER_CLASS:
            sampling_action = "oversample"
            augmentation = "minority"
        elif original_count > MAX_SAMPLES_PER_CLASS:
            sampling_action = "undersample"
            augmentation = "standard"
        else:
            sampling_action = "keep"
            augmentation = "standard"

        rows.append(
            {
                "label": class_id,
                "original_train_count": original_count,
                "effective_count_per_epoch": effective_count,
                "sampling_action": sampling_action,
                "augmentation": augmentation,
                "class_weight": float(class_weights_cpu[class_id]),
            }
        )

    configuration = pd.DataFrame(rows)

    configuration.to_csv(
        RESAMPLING_SUMMARY_PATH,
        index=False,
    )

    configuration[
        [
            "label",
            "original_train_count",
            "class_weight",
        ]
    ].to_csv(
        CLASS_WEIGHTS_PATH,
        index=False,
    )


# 9. Model, loss, optimizer, and scheduler

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

model = resnet50(weights=ResNet50_Weights.DEFAULT)

number_of_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(number_of_features, NUM_CLASSES),
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

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


# 10. Training and validation functions

def train_one_epoch():
    """Train the model once on one resampled training epoch."""
    model.train()

    total_loss = 0.0
    number_correct = 0
    number_samples = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=False,
    )

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

    average_loss = total_loss / number_samples
    accuracy = number_correct / number_samples

    return average_loss, accuracy


def validate():
    """Evaluate the model on the unchanged validation distribution."""
    model.eval()

    total_loss = 0.0
    number_correct = 0
    number_samples = 0

    all_predictions = []
    all_labels = []

    # Report validation loss using ordinary unweighted cross-entropy so it is
    # easier to compare with the baseline validation loss.
    validation_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        progress_bar = tqdm(
            val_loader,
            desc="Validation",
            leave=False,
        )

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = validation_criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            number_correct += (predictions == labels).sum().item()
            number_samples += labels.size(0)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / number_samples
    accuracy = number_correct / number_samples

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

    return average_loss, accuracy, macro_f1, g_mean


# 11. Complete training loop

def main():
    """
    Train the imbalance-aware ResNet50 model.

    Imbalance handling:
    1. Classes below 150 samples are oversampled to 150 per epoch.
    2. Classes above 2500 samples are undersampled to 2500 per epoch.
    3. Classes below 150 samples receive mild online augmentation.
    4. Training uses standard unweighted cross-entropy.
    5. Validation remains unchanged.
    """
    save_imbalance_configuration()

    logger, train_log_path = create_logger()

    logger.info(f"Device: {device}")
    logger.info(f"Original training images: {len(raw_train_dataset)}")
    logger.info(f"Effective images per epoch: {len(train_sampler)}")
    logger.info(f"Validation images: {len(val_dataset)}")
    logger.info(f"Minority threshold: < {MIN_SAMPLES_PER_CLASS}")
    logger.info(f"Majority cap: > {MAX_SAMPLES_PER_CLASS}")
    logger.info(
        "Class-weight range: "
        f"{class_weights_cpu.min().item():.4f} "
        f"to {class_weights_cpu.max().item():.4f}"
    )
    logger.info(f"Maximum epochs: {MAX_EPOCHS}")
    logger.info(f"Best model path: {BEST_MODEL_PATH}")
    logger.info(f"Training log path: {train_log_path}")
    logger.info(f"Resampling summary: {RESAMPLING_SUMMARY_PATH}")
    logger.info(f"Class weights: {CLASS_WEIGHTS_PATH}")

    best_macro_f1 = 0.0
    epochs_without_improvement = 0
    training_history = []

    for epoch in range(MAX_EPOCHS):
        train_sampler.set_epoch(epoch)

        train_loss, train_accuracy = train_one_epoch()
        val_loss, val_accuracy, val_macro_f1, val_g_mean = validate()

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

        training_history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": current_learning_rate,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "val_g_mean": val_g_mean,
            }
        )

        pd.DataFrame(training_history).to_csv(
            TRAINING_HISTORY_PATH,
            index=False,
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "val_g_mean": val_g_mean,
                "minimum_samples_per_class": MIN_SAMPLES_PER_CLASS,
                "maximum_samples_per_class": MAX_SAMPLES_PER_CLASS,
                "class_weights": class_weights_cpu,
            }

            torch.save(
                checkpoint,
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
                f"{previous_learning_rate:.6f} "
                f"-> {new_learning_rate:.6f}"
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            logger.info(
                "Early stopping: validation Macro-F1 "
                f"did not improve for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    logger.info("Training complete")
    logger.info(f"Best validation Macro-F1: {best_macro_f1}")
    logger.info(f"Best model path: {BEST_MODEL_PATH}")
    logger.info(f"Training log path: {train_log_path}")
    logger.info(f"Training history path: {TRAINING_HISTORY_PATH}")


if __name__ == "__main__":
    main()
