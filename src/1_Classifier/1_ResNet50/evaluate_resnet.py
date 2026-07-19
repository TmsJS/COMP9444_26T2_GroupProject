import sys
from datetime import datetime
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
    confusion_matrix,
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

MODEL_PATH = OUTPUT_DIR / "resnet50_best_model.pth"
REPORT_PATH = OUTPUT_DIR / "test_classification_report.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

SUMMARY_PATH = (
    OUTPUT_DIR
    / f"resnet50_test_summary_{timestamp}.txt"
)

CLASSES_PATH = DATA_DIR.parent / "classes.txt"

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "test_confusion_matrix.csv"
)
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

# Load insect names from classes.txt
def load_insect_names(classes_path: Path) -> dict[int, str]:
    insect_names = {}

    with classes_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            class_id, insect_name = line.split(maxsplit=1)

            # classes.txt uses labels 1–102,
            # while PyTorch uses labels 0–101.
            zero_based_label = int(class_id) - 1

            insect_names[zero_based_label] = insect_name

    return insect_names


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

summary = "\n".join([
    f"Best checkpoint epoch: {checkpoint['epoch']}",
    f"Best validation accuracy: {checkpoint['val_accuracy']:.4f}",
    f"Best validation Macro-F1: {checkpoint['val_macro_f1']:.4f}",
    "",
    f"Test loss: {test_loss:.4f}",
    f"Test accuracy: {test_accuracy:.4f}",
    f"Test macro precision: {test_precision:.4f}",
    f"Test macro recall: {test_recall:.4f}",
    f"Test macro-F1: {test_macro_f1:.4f}",
    f"Test GM: {test_g_mean:.4f}",
])

print()
print(summary)

SUMMARY_PATH.write_text(
    summary + "\n",
    encoding="utf-8",
)


# 9. Save per-class report

# 9. Save per-class report

insect_names = load_insect_names(CLASSES_PATH)

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
        "insect_name": insect_names.get(
            label,
            f"Unknown class {label}",
        ),
        "precision": class_metrics["precision"],
        "recall": class_metrics["recall"],
        "f1-score": class_metrics["f1-score"],
        "support": int(class_metrics["support"]),
    })

report_dataframe = pd.DataFrame(
    report_rows,
    columns=[
        "label",
        "insect_name",
        "precision",
        "recall",
        "f1-score",
        "support",
    ],
)

report_dataframe.to_csv(
    REPORT_PATH,
    index=False,
    float_format="%.4f",
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

# 10. Save confusion matrix

confusion = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
)
confusion_dataframe = pd.DataFrame(
    confusion,
    index=[
        insect_names[label]
        for label in range(NUM_CLASSES)
    ],
    columns=[
        insect_names[label]
        for label in range(NUM_CLASSES)
    ],
)
confusion_dataframe.to_csv(
    CONFUSION_MATRIX_PATH,
    index=True,
)
print()
print("Test summary saved to:", SUMMARY_PATH)
print("Classification report saved to:", REPORT_PATH)
print("Predictions saved to:", PREDICTIONS_PATH)
print("Confusion matrix saved to:",CONFUSION_MATRIX_PATH)