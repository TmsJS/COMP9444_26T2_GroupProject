import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm
import sys
from pathlib import Path
from imblearn.metrics import geometric_mean_score

# This file is located at:
# COMP9444_Group/src/1_Classifier/1_resnet50/train.py
# parents[3] therefore points to COMP9444_Group.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Add the project root to Python's module search path.
# This allows imports beginning with "src".
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# This import must appear after PROJECT_ROOT is added to sys.path.
from src.data.ip102_dataset import IP102Dataset


# 1. Training configuration

# Number of insect classes in IP102.
NUM_CLASSES = 102

# Paper setting: mini-batch size of 64.
# Reduce this to 32 or 16 if the GPU runs out of memory.
BATCH_SIZE = 64

# Maximum number of training epochs.
# Early stopping will normally finish training sooner.
MAX_EPOCHS = 40

# Initial learning rate.
LEARNING_RATE = 0.01

# Number of worker processes used by each DataLoader.
NUM_WORKERS = 4

# Reduce the learning rate after this many epochs
# without validation Macro-F1 improvement.
LR_PATIENCE = 3

# Stop training after this many epochs
# without validation Macro-F1 improvement.
EARLY_STOPPING_PATIENCE = 6


# 2. Project and dataset paths

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "Classification"
    / "ip102_v1.1"
)
IMAGES_DIR = DATA_DIR / "images"

# Directory used to store trained model checkpoints.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classifier" / "resnet50"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pth"


# 3. Image preprocessing

# The paper fixes the input image size to 224 x 224.
# These normalization values are used by models pretrained on the ImageNet dataset.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Validation images should not use random augmentation.
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# 4. Create datasets and DataLoaders

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
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# 5. Create the pretrained ResNet50 model

# Use the GPU if CUDA is available or Mac mps.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# Turn on CuDNN benchmark if cuda 
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
# Load ResNet50 weights pretrained on ImageNet.
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Save the number of features produced by ResNet50.
# For ResNet50, this value is 2048.
number_of_features = model.fc.in_features

# Replace the original 1000-class ImageNet classifier.
#
# The paper uses dropout=0.3 and changes the final output
# layer to match the number of IP102 classes.
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(number_of_features, NUM_CLASSES),
)

model = model.to(device)


# 6. Loss function, optimizer and learning-rate scheduler

# CrossEntropyLoss is used for multi-class classification.
criterion = nn.CrossEntropyLoss()

# Paper settings:
#   learning rate = 0.01
#   momentum = 0.9
#   weight decay = 0.0005
#
# model.parameters() means that all ResNet50 layers are fine-tuned.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=0.0005,
)

# Reduce the learning rate when validation Macro-F1
# stops improving.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.1,
    patience=LR_PATIENCE,
    min_lr=1e-6,
)

# 7. Train the model for one epoch

def train_one_epoch():
    """Train the model once on the complete training dataset."""

    # Enable training behavior such as Dropout.
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
        # Move the batch to the GPU or CPU.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Remove gradients calculated for the previous batch.
        optimizer.zero_grad(set_to_none=True)

        # Run the images through ResNet50.
        outputs = model(images)

        # Compare the predicted outputs with the correct labels.
        loss = criterion(outputs, labels)

        # Calculate gradients.
        loss.backward()

        # Update all model parameters.
        optimizer.step()

        # Add this batch loss to the epoch loss.
        total_loss += loss.item() * images.size(0)

        # Select the class with the largest output value.
        predictions = outputs.argmax(dim=1)

        number_correct += (predictions == labels).sum().item()
        number_samples += labels.size(0)

        progress_bar.set_postfix(loss=loss.item())

    average_loss = total_loss / number_samples
    accuracy = number_correct / number_samples

    return average_loss, accuracy


# 8. Evaluate the model on the validation dataset

def validate():
    """Evaluate the model without changing its parameters."""

    # Disable training behavior such as Dropout.
    model.eval()

    total_loss = 0.0
    number_correct = 0
    number_samples = 0

    all_predictions = []
    all_labels = []

    # Do not calculate gradients during validation.
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
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            number_correct += (predictions == labels).sum().item()
            number_samples += labels.size(0)

            # Move predictions back to the CPU for scikit-learn.
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / number_samples
    accuracy = number_correct / number_samples

    # Macro-F1 gives every insect class equal importance.
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
        
    )

    # G-mean measures whether the model performs well
    # across both majority and minority classes.
    g_mean = float(geometric_mean_score(
        all_labels,
        all_predictions,
        average="multiclass",
    ))
    
    return average_loss, accuracy, macro_f1, g_mean


# 9. Complete training loop

def main():
    """
    Train ResNet50 from the beginning.

    Validation Macro-F1 is used to:
    1. Save the best model.
    2. Adjust the learning rate.
    3. Apply early stopping.
    """

    print("Device:", device)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Maximum epochs:", MAX_EPOCHS)
    print("Best model path:", BEST_MODEL_PATH)

    # Start a completely new training run.
    best_macro_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        # Train the model for one complete epoch.
        train_loss, train_accuracy = train_one_epoch()

        # Evaluate the current model on the validation dataset.
        val_loss, val_accuracy, val_macro_f1, val_g_mean = validate()

        current_learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{MAX_EPOCHS}] | "
            f"LR: {current_learning_rate:.6f} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f} | "
            f"Val macro-F1: {val_macro_f1:.4f} | "
            f"Val GM: {val_g_mean:.4f}"
        )

        # Save the model when validation Macro-F1 improves.
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
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )

            print(
                "Saved new best model "
                f"(Macro-F1: {best_macro_f1:.4f})"
            )

        else:
            epochs_without_improvement += 1

            print(
                "Epochs without Macro-F1 improvement:",
                epochs_without_improvement,
            )

        # Remember the current learning rate so that we can
        # detect whether the scheduler reduces it.
        previous_learning_rate = optimizer.param_groups[0]["lr"]

        # Update the scheduler using validation Macro-F1.
        scheduler.step(val_macro_f1)

        new_learning_rate = optimizer.param_groups[0]["lr"]

        if new_learning_rate < previous_learning_rate:
            print(
                f"Learning rate reduced: "
                f"{previous_learning_rate:.6f} "
                f"-> {new_learning_rate:.6f}"
            )

        # Stop when Macro-F1 has not improved for too long.
        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "Early stopping: validation Macro-F1 "
                f"did not improve for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    print("Training complete")
    print("Best validation Macro-F1:", best_macro_f1)
    print("Best model path:", BEST_MODEL_PATH)

if __name__ == "__main__":
    main()