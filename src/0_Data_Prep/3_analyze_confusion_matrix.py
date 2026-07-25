from pathlib import Path
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFUSION_MATRIX_CSV = PROJECT_ROOT / "outputs" / "classifier" / "resnet50_imbalance" / "test_confusion_matrix.csv"
TOP_N_WORST_CLASSES = 15
TOP_N_CONFUSED_PAIRS = 20
SAVE_REPORT = True


def load_confusion_matrix(csv_path: Path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        classes = header[1:]  # predicted-class column names
        matrix = []
        row_classes = []
        for row in reader:
            row_classes.append(row[0])
            matrix.append([float(x) for x in row[1:]])
    assert row_classes == classes, "row order and column order must match"
    return classes, matrix


def per_class_recall(classes, matrix):
    n = len(classes)
    results = []
    for i in range(n):
        row_sum = sum(matrix[i])
        correct = matrix[i][i]
        recall = correct / row_sum if row_sum > 0 else 0.0
        results.append((classes[i], recall, row_sum))
    results.sort(key=lambda x: x[1])  # worst first
    return results


def top_confused_pairs(classes, matrix, top_n):
    n = len(classes)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > 0:
                pairs.append((matrix[i][j], classes[i], classes[j]))
    pairs.sort(reverse=True)
    return pairs[:top_n]


def mutual_top1_pairs(classes, matrix):
    n = len(classes)
    top1 = {}
    for i in range(n):
        best_j, best_v = None, -1
        for j in range(n):
            if j != i and matrix[i][j] > best_v:
                best_j, best_v = j, matrix[i][j]
        top1[i] = (best_j, best_v)

    mutual = []
    for i in range(n):
        j, v1 = top1[i]
        j2, v2 = top1[j]
        if j2 == i and i < j:
            mutual.append((classes[i], classes[j], v1, v2))
    return mutual


def main():
    classes, matrix = load_confusion_matrix(CONFUSION_MATRIX_CSV)
    n = len(classes)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(n))
    print(f"Loaded {n}x{n} confusion matrix, overall accuracy = {correct/total:.4f}\n")

    print(f"--- 1. Worst {TOP_N_WORST_CLASSES} classes by recall ---")
    recalls = per_class_recall(classes, matrix)
    for name, recall, n_samples in recalls[:TOP_N_WORST_CLASSES]:
        print(f"  {name:35s} recall={recall:.3f}  n={n_samples:.0f}")

    print(f"\n--- 2. Top {TOP_N_CONFUSED_PAIRS} confused (true -> predicted) pairs ---")
    pairs = top_confused_pairs(classes, matrix, TOP_N_CONFUSED_PAIRS)
    for count, true_c, pred_c in pairs:
        print(f"  {count:5.0f}  {true_c:30s} -> {pred_c}")

    print("\n--- 3. Mutual top-1 confusion pairs (A's #1 mistake is B, and vice versa) ---")
    mutual = mutual_top1_pairs(classes, matrix)
    for a, b, v1, v2 in mutual:
        print(f"  {a:30s} <-> {b:30s}   ({v1:.0f} / {v2:.0f})")

    if SAVE_REPORT:
        report = {
            "overall_accuracy": correct / total,
            "worst_recall_classes": [
                {"class": n_, "recall": r, "n_samples": s} for n_, r, s in recalls[:TOP_N_WORST_CLASSES]
            ],
            "top_confused_pairs": [
                {"true_class": t, "predicted_class": p, "count": c} for c, t, p in pairs
            ],
            "mutual_top1_pairs": [
                {"class_a": a, "class_b": b, "a_to_b": v1, "b_to_a": v2} for a, b, v1, v2 in mutual
            ],
        }
        out_path = CONFUSION_MATRIX_CSV.parent / "confusion_analysis_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved full report -> {out_path}")


if __name__ == "__main__":
    main()