import os
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATA_ROOT = os.path.join(BASE_DIR, "datasets", "raw", "Classification")
CLASSIFICATION_DIR = os.path.join(DATA_ROOT, "ip102_v1.1")
TRAIN_TXT = os.path.join(CLASSIFICATION_DIR, "train.txt")
CLASSES_TXT = os.path.join(DATA_ROOT, "classes.txt")

OUTPUT_DIR = SCRIPT_DIR


# Load train.txt into a DataFrame
def load_labels(txt_path):
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            filename, label = parts
            rows.append({"filename": filename, "label": int(label)})
    return pd.DataFrame(rows)


# Load class names for nicer labels on charts
def load_class_names(txt_path):
    names = {}
    if not os.path.exists(txt_path):
        return names
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            idx, name = parts
            try:
                names[int(idx) - 1] = name.strip()
            except ValueError:
                continue
    return names


# Main analysis
def main():
    df = load_labels(TRAIN_TXT)
    print(f"Total training images: {len(df)}")
    print(f"Number of unique labels found: {df['label'].nunique()}")

    class_names = load_class_names(CLASSES_TXT)

    # Count images per class, sorted descending
    class_counts = df["label"].value_counts().sort_values(ascending=False)

    # Top 3 most / least frequent classes
    top3 = class_counts.head(3)
    bottom3 = class_counts.tail(3)

    print("\nTop 3 MOST frequent labels in training set:")
    for label, count in top3.items():
        name = class_names.get(label, "unknown")
        print(f"  Label {label} ({name}): {count} images")

    print("\nTop 3 LEAST frequent labels in training set:")
    for label, count in bottom3.items():
        name = class_names.get(label, "unknown")
        print(f"  Label {label} ({name}): {count} images")

    # Basic stats
    print(f"\nMean images per class: {class_counts.mean():.1f}")
    print(f"Median images per class: {class_counts.median():.1f}")
    print(f"Imbalance ratio (max/min): {class_counts.max() / class_counts.min():.1f}")

    # Plot: sorted bar chart of images per class
    plt.figure(figsize=(16, 5))
    plt.bar(range(len(class_counts)), class_counts.values, color="teal")
    plt.xlabel("Class label (sorted by frequency, descending)")
    plt.ylabel("Number of training images")
    plt.title("IP102 Training Set: Number of Images per Class (102 classes)")
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "class_distribution_train.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved chart to {plot_path}")
    plt.show()

    # Save full table to CSV for reference
    result_df = class_counts.reset_index()
    result_df.columns = ["label", "count"]
    result_df["class_name"] = result_df["label"].map(class_names)
    csv_path = os.path.join(OUTPUT_DIR, "class_distribution_train.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Saved full table to {csv_path}")

    return class_counts, top3, bottom3


if __name__ == "__main__":
    main()