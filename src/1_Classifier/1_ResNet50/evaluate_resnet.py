import sys
from pathlib import Path

import pandas as pd
import torch
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm


# 1. Project import path
# parents[3] therefore points to COMP9444_Group.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ip102_dataset import IP102Dataset


# 2. Configuration and paths
NUM_CLASSES = 102
BATCH_SIZE = 64
NUM_WORKERS = 4

DATA_DIR = (
    PROJECT_ROOT / "datasets" / "raw" / "Classification" / "ip102_v1.1"
)

IMAGES_DIR = DATA_DIR / "images"

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "classifier" / "resnet50"
)

MODEL_PATH = OUTPUT_DIR / "best_model.pth"
REPORT_PATH = OUTPUT_DIR / "test_classification_report.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"


# 3. Select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# 4. Test dataset
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
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
    persistent_workers=NUM_WORKERS > 0,
)


# 5. Recreate ResNet50

# Pretrained weights are not needed here because the checkpoint
# will replace all model parameters.
model = resnet50(weights=None)

number_of_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(number_of_features, NUM_CLASSES),
)

model = model.to(device)


# 6. Load the best checkpoint

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# 7. Evaluate the test dataset

criterion = nn.CrossEntropyLoss()

all_predictions = []
all_labels = []

total_loss = 0.0
number_samples = 0

with torch.no_grad():
    progress_bar = tqdm(
        test_loader,
        desc="Testing",
    )

    for images, labels in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        number_samples += labels.size(0)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )


# 8. Calculate final metrics

test_loss = total_loss / number_samples

test_accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

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

test_g_mean = geometric_mean_score(
    all_labels,
    all_predictions,
    average="multiclass",
)


print()
print("Best checkpoint epoch:", checkpoint["epoch"])
print("Best validation accuracy:", checkpoint["val_accuracy"])
print("Best validation Macro-F1:", checkpoint["val_macro_f1"])
print()
print("Test loss:", f"{test_loss:.4f}")
print("Test accuracy:", f"{test_accuracy:.4f}")
print("Test macro precision:", f"{test_precision:.4f}")
print("Test macro recall:", f"{test_recall:.4f}")
print("Test macro-F1:", f"{test_macro_f1:.4f}")
print("Test GM:", f"{test_g_mean:.4f}")


# 9. Save per-class report

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    output_dict=True,
    zero_division=0,
)

report_dataframe = pd.DataFrame(
    report
).transpose()

report_dataframe.to_csv(
    REPORT_PATH,
    index=True,
)


# 10. Save every test prediction

image_names = [
    image_name
    for image_name, _ in test_dataset.samples
]

predictions_dataframe = pd.DataFrame({
    "image_name": image_names,
    "true_label": all_labels,
    "predicted_label": all_predictions,
})

predictions_dataframe.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


print()
print("Classification report saved to:", REPORT_PATH)
print("Predictions saved to:", PREDICTIONS_PATH)