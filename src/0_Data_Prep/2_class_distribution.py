# Calculate and visualise the class distribution of the IP102 dataset.
import csv
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

# Paths to the dataset and output directories.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "datasets" / "raw" / "Classification"
SPLIT_ROOT = DATA_ROOT / "ip102_v1.1"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "data_analysis"


# Return a mapping from the zero-based dataset label to its class name.
def load_class_names(path: Path) -> dict[int, str]:
    class_names: dict[int, str] = {}

    with path.open(encoding="utf-8") as file:
        for line in file:
            class_number, class_name = line.strip().split(maxsplit=1)
            class_names[int(class_number) - 1] = class_name

    return class_names


# Count how many image records belong to each label in one split file.
def count_split(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.split()

            if len(fields) != 2:
                raise ValueError(
                    f"Invalid row in {path} at line {line_number}: {line!r}"
                )

            counts[int(fields[1])] += 1

    return counts


# Save exact per-class counts and percentages for later experiments.
def save_csv(
    rows: list[dict[str, int | float | str]],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Save a sorted stacked bar chart that exposes the dataset's long tail.
def save_chart(
    rows: list[dict[str, int | float | str]],
    path: Path,
) -> None:
    ranked = sorted(
        rows,
        key=lambda row: int(row["total_count"]),
        reverse=True,
    )

    ranks = list(range(1, len(ranked) + 1))
    train = [int(row["train_count"]) for row in ranked]
    val = [int(row["val_count"]) for row in ranked]
    test = [int(row["test_count"]) for row in ranked]

    fig, axis = plt.subplots(figsize=(14, 7))

    axis.bar(
        ranks,
        train,
        width=0.85,
        color="#3264A8",
        label="Train",
    )

    axis.bar(
        ranks,
        val,
        width=0.85,
        bottom=train,
        color="#D89B2B",
        label="Validation",
    )

    axis.bar(
        ranks,
        test,
        width=0.85,
        bottom=[
            train_count + val_count
            for train_count, val_count in zip(train, val)
        ],
        color="#E2763E",
        label="Test",
    )

    axis.set_title("IP102 class distribution")
    axis.set_xlabel("Class rank by total image count (largest to smallest)")
    axis.set_ylabel("Number of images")
    axis.set_xlim(0, len(ranked) + 1)
    axis.set_ylim(bottom=0)
    axis.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.8,
        alpha=0.7,
    )
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)

    fig.text(
        0.125,
        0.01,
        "Source: local IP102 train.txt, val.txt, and test.txt split files; "
        "102 classes.",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# Build a report of the most and least frequent classes in one split.
def build_split_frequency_report(
    split_name: str,
    counts: Counter[int],
    class_names: dict[int, str],
    top_n: int = 3,
) -> list[str]:
    ranked = sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    most_frequent = ranked[:top_n]
    least_frequent = sorted(
        ranked[-top_n:],
        key=lambda item: item[1],
    )

    split_total = sum(counts.values())

    largest_label_id, largest_count = ranked[0]
    smallest_label_id, smallest_count = ranked[-1]

    imbalance_ratio = largest_count / smallest_count

    lines = [
        f"{split_name.capitalize()} set summary:",
        f"All images in {split_name} set: {split_total}",
        (
            f"Largest class: {class_names[largest_label_id]} "
            f"({largest_count} images)"
        ),
        (
            f"Smallest class: {class_names[smallest_label_id]} "
            f"({smallest_count} images)"
        ),
        f"Max/min imbalance ratio: {imbalance_ratio:.2f}:1",
        "",
        f"Top {top_n} MOST frequent labels in {split_name} set:",
    ]

    for label_id, count in most_frequent:
        lines.append(
            f"  Label {label_id} ({class_names[label_id]}): "
            f"{count} images"
        )

    lines.extend(
        [
            "",
            f"Top {top_n} LEAST frequent labels in {split_name} set:",
        ]
    )

    for label_id, count in least_frequent:
        lines.append(
            f"  Label {label_id} ({class_names[label_id]}): "
            f"{count} images"
        )

    return lines

# Save the complete terminal report as a text file.
def save_text_report(lines: list[str], path: Path) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Compute counts, validate totals, and write the output files.
def main() -> None:
    class_names = load_class_names(DATA_ROOT / "classes.txt")

    split_counts = {
        split: count_split(SPLIT_ROOT / f"{split}.txt")
        for split in ("train", "val", "test")
    }

    expected_labels = set(class_names)
    observed_labels = set().union(
        *(set(counts) for counts in split_counts.values())
    )

    if observed_labels != expected_labels:
        raise ValueError(
            f"Label mismatch: "
            f"missing={sorted(expected_labels - observed_labels)}, "
            f"unexpected={sorted(observed_labels - expected_labels)}"
        )

    grand_total = sum(
        sum(counts.values())
        for counts in split_counts.values()
    )

    rows: list[dict[str, int | float | str]] = []

    for label_id, class_name in sorted(class_names.items()):
        train_count = split_counts["train"][label_id]
        val_count = split_counts["val"][label_id]
        test_count = split_counts["test"][label_id]
        total_count = train_count + val_count + test_count

        rows.append(
            {
                "label_id": label_id,
                "class_number": label_id + 1,
                "class_name": class_name,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
                "total_count": total_count,
                "dataset_percent": round(
                    100 * total_count / grand_total,
                    4,
                ),
            }
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_ROOT / "ip102_class_distribution.csv"
    chart_path = OUTPUT_ROOT / "ip102_class_distribution.png"
    report_path = OUTPUT_ROOT / "ip102_class_distribution_report.txt"

    save_csv(rows, csv_path)
    save_chart(rows, chart_path)

    ranked = sorted(
        rows,
        key=lambda row: int(row["total_count"]),
        reverse=True,
    )

    split_totals = {
        split: sum(counts.values())
        for split, counts in split_counts.items()
    }

    imbalance_ratio = (
        int(ranked[0]["total_count"])
        / int(ranked[-1]["total_count"])
    )

    report_lines = [
        f"Classes: {len(rows)}",
        f"Split totals: {split_totals}",
        f"All images: {grand_total}",
        (
            f"Largest class: {ranked[0]['class_name']} "
            f"({ranked[0]['total_count']} images)"
        ),
        (
            f"Smallest class: {ranked[-1]['class_name']} "
            f"({ranked[-1]['total_count']} images)"
        ),
        f"Max/min imbalance ratio: {imbalance_ratio:.2f}:1",
        "",
    ]

    for split in ("train", "val", "test"):
        report_lines.extend(
            build_split_frequency_report(
                split_name=split,
                counts=split_counts[split],
                class_names=class_names,
                top_n=3,
            )
        )
        report_lines.append("")

    report_lines.extend(
        [
            f"CSV: {csv_path}",
            f"Chart: {chart_path}",
            f"Text report: {report_path}",
        ]
    )

    save_text_report(report_lines, report_path)

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()