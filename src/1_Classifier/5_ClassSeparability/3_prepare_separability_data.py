"""
Prepare a frozen coarse-label mapping and cluster annotation subsets.

This script is the third step of the IP102 class-separability workflow. It
consumes selected_clusters.csv produced by 2_evaluate_difficulty_groups.py;
it does not select clusters again and does not define or evaluate difficulty
groups.

The cluster count, members, and resulting number of coarse classes are not
hard-coded. They are derived entirely from selected_clusters.csv, which must
be created from the automatically discovered validation clusters and frozen
before this script or any test-set evaluation is run. Every selected cluster
becomes one coarse class; all unselected original classes remain singleton
coarse classes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NUM_CLASSES = 102
SPLITS = ("train", "val", "test")
GENERATED_CLUSTER_FILES = {
    "local_label_mapping.csv",
    *(f"{split}.txt" for split in SPLITS),
}


def parse_arguments() -> argparse.Namespace:
    """Read project, frozen-definition, and output locations."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a frozen 102-to-coarse mapping and annotation "
            "subsets from selected validation clusters."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "COMP9444_Group root. By default, parents[3] of this "
            "script is used."
        ),
    )
    parser.add_argument(
        "--selected-clusters",
        type=Path,
        default=None,
        help=(
            "Frozen selected_clusters.csv produced by "
            "2_evaluate_difficulty_groups.py. Default: "
            "outputs/classifier/class_separability/"
            "selected_clusters.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for coarse_label_mapping.csv and the preparation "
            "summary. Default: outputs/classifier/class_separability."
        ),
    )
    parser.add_argument(
        "--cluster-subsets-dir",
        type=Path,
        default=None,
        help=(
            "Cluster annotation output directory. Default: "
            "datasets/processed/ip102_cluster_subsets."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path, project_root: Path) -> Path:
    """Resolve a path relative to the project root when necessary."""
    path = path.expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def resolve_project_root(requested: Path | None) -> Path:
    """Resolve and validate the COMP9444_Group project root."""
    if requested is None:
        project_root = Path(__file__).resolve().parents[3]
    else:
        project_root = requested.expanduser().resolve()

    classification_root = (
        project_root
        / "datasets"
        / "raw"
        / "Classification"
    )
    required = [
        classification_root / "classes.txt",
        *[
            classification_root
            / "ip102_v1.1"
            / f"{split}.txt"
            for split in SPLITS
        ],
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Invalid project root or missing IP102 files:\n"
            f"{missing_text}"
        )
    return project_root


def resolve_paths(
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Path]:
    """Resolve the frozen input and preparation outputs."""
    classification_root = (
        project_root
        / "datasets"
        / "raw"
        / "Classification"
    )
    output_dir = (
        project_root
        / "outputs"
        / "classifier"
        / "class_separability"
        if args.output_dir is None
        else resolve_path(args.output_dir, project_root)
    )
    selected_clusters = (
        output_dir / "selected_clusters.csv"
        if args.selected_clusters is None
        else resolve_path(args.selected_clusters, project_root)
    )
    cluster_subsets_dir = (
        project_root
        / "datasets"
        / "processed"
        / "ip102_cluster_subsets"
        if args.cluster_subsets_dir is None
        else resolve_path(args.cluster_subsets_dir, project_root)
    )

    if not selected_clusters.is_file():
        raise FileNotFoundError(
            "Frozen selected-cluster file not found:\n"
            f"{selected_clusters}\n"
            "Run 2_evaluate_difficulty_groups.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_subsets_dir.mkdir(parents=True, exist_ok=True)
    return {
        "classes": classification_root / "classes.txt",
        "split_root": classification_root / "ip102_v1.1",
        "selected_clusters": selected_clusters,
        "output_dir": output_dir,
        "cluster_subsets_dir": cluster_subsets_dir,
        "coarse_mapping": output_dir / "coarse_label_mapping.csv",
        "preparation_summary": (
            output_dir / "separability_preparation_summary.csv"
        ),
        "subset_summary": (
            cluster_subsets_dir / "cluster_subset_summary.csv"
        ),
    }


def load_class_names(classes_path: Path) -> list[str]:
    """Load classes.txt and convert one-based IDs to zero-based order."""
    label_to_name: dict[int, str] = {}

    with classes_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                class_number, class_name = stripped.split(maxsplit=1)
            except ValueError as error:
                raise ValueError(
                    f"Invalid classes.txt row {line_number}: {line!r}"
                ) from error

            label = int(class_number) - 1
            if label in label_to_name:
                raise ValueError(
                    f"Duplicate class number: {class_number}"
                )
            label_to_name[label] = class_name.strip()

    expected = set(range(NUM_CLASSES))
    observed = set(label_to_name)
    if observed != expected:
        raise ValueError(
            "classes.txt must contain labels 1-102 exactly. "
            f"Missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    class_names = [
        label_to_name[label]
        for label in range(NUM_CLASSES)
    ]
    if len(set(class_names)) != NUM_CLASSES:
        raise ValueError("classes.txt contains duplicate class names.")
    return class_names


def parse_members(value: object) -> list[str]:
    """Parse a pipe-separated cluster-member field."""
    if pd.isna(value):
        return []
    return [
        member.strip()
        for member in str(value).split("|")
        if member.strip()
    ]


def load_frozen_clusters(
    csv_path: Path,
    class_names: list[str],
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Load selected clusters without selecting or scoring them again."""
    dataframe = pd.read_csv(csv_path)
    required = {"cluster_name", "class_members"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Selected-cluster CSV is missing columns: {sorted(missing)}"
        )
    if dataframe.empty:
        raise ValueError(
            f"Selected-cluster CSV is empty: {csv_path}"
        )
    if "split" in dataframe.columns:
        invalid_splits = (
            dataframe["split"].astype(str).str.lower() != "val"
        )
        if invalid_splits.any():
            raise ValueError(
                "selected_clusters.csv must contain validation rows only."
            )

    if "visual_cluster_rank" in dataframe.columns:
        dataframe = dataframe.sort_values(
            "visual_cluster_rank",
            ascending=True,
        ).reset_index(drop=True)
    else:
        dataframe = dataframe.reset_index(drop=True)

    name_to_label = {
        name: label
        for label, name in enumerate(class_names)
    }
    clusters: dict[str, list[int]] = {}
    used_labels: dict[int, str] = {}

    for row_number, row in dataframe.iterrows():
        cluster_name = str(row["cluster_name"]).strip()
        member_names = parse_members(row["class_members"])

        if not cluster_name:
            raise ValueError(
                f"Empty cluster name at row {row_number + 1}."
            )
        if cluster_name in clusters:
            raise ValueError(
                f"Duplicate cluster name: {cluster_name}"
            )
        if len(member_names) < 2:
            raise ValueError(
                f"Cluster {cluster_name!r} needs at least two classes."
            )

        unknown = [
            name
            for name in member_names
            if name not in name_to_label
        ]
        if unknown:
            raise ValueError(
                f"Unknown classes in {cluster_name!r}: {unknown}"
            )

        labels = [name_to_label[name] for name in member_names]
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"Duplicate member inside {cluster_name!r}."
            )
        for label in labels:
            if label in used_labels:
                raise ValueError(
                    f"{class_names[label]!r} belongs to both "
                    f"{used_labels[label]!r} and {cluster_name!r}."
                )
            used_labels[label] = cluster_name

        clusters[cluster_name] = labels

    return dataframe, clusters


def build_coarse_mapping(
    class_names: list[str],
    clusters: dict[str, list[int]],
) -> pd.DataFrame:
    """Map clustered classes together and retain all others as singletons."""
    fine_to_cluster = {
        label: cluster_name
        for cluster_name, members in clusters.items()
        for label in members
    }
    cluster_to_coarse = {
        cluster_name: coarse_label
        for coarse_label, cluster_name in enumerate(clusters)
    }
    next_coarse_label = len(clusters)
    rows: list[dict[str, int | str | bool]] = []

    for fine_label, class_name in enumerate(class_names):
        if fine_label in fine_to_cluster:
            cluster_name = fine_to_cluster[fine_label]
            coarse_label = cluster_to_coarse[cluster_name]
            coarse_name = cluster_name
            is_merged = True
        else:
            cluster_name = ""
            coarse_label = next_coarse_label
            coarse_name = class_name
            is_merged = False
            next_coarse_label += 1

        rows.append({
            "fine_label": fine_label,
            "class_number": fine_label + 1,
            "class_name": class_name,
            "coarse_label": coarse_label,
            "coarse_name": coarse_name,
            "is_merged": is_merged,
            "cluster_name": cluster_name,
        })

    mapping = pd.DataFrame(rows)
    expected_coarse = (
        NUM_CLASSES
        - sum(len(members) for members in clusters.values())
        + len(clusters)
    )
    number_coarse = mapping["coarse_label"].nunique()
    if number_coarse != expected_coarse:
        raise RuntimeError(
            f"Expected {expected_coarse} coarse labels, "
            f"constructed {number_coarse}."
        )
    return mapping


def read_split_annotations(path: Path) -> list[tuple[str, int]]:
    """Read one IP102 image-name and fine-label annotation file."""
    samples: list[tuple[str, int]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"Invalid row in {path} line {line_number}: "
                    f"{line!r}"
                )

            image_name = fields[0]
            label = int(fields[1])
            if label < 0 or label >= NUM_CLASSES:
                raise ValueError(
                    f"Invalid label {label} in {path} "
                    f"line {line_number}."
                )
            samples.append((image_name, label))
    return samples


def write_cluster_subsets(
    clusters: dict[str, list[int]],
    class_names: list[str],
    split_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    """Create local-label annotations for every frozen cluster and split."""
    split_samples = {
        split: read_split_annotations(
            split_root / f"{split}.txt"
        )
        for split in SPLITS
    }
    summary_rows: list[dict[str, int | str]] = []

    for cluster_name, members in clusters.items():
        cluster_dir = output_root / cluster_name
        cluster_dir.mkdir(parents=True, exist_ok=True)
        local_mapping = {
            fine_label: local_label
            for local_label, fine_label in enumerate(members)
        }
        mapping_rows = [
            {
                "local_label": local_label,
                "fine_label": fine_label,
                "class_number": fine_label + 1,
                "class_name": class_names[fine_label],
            }
            for fine_label, local_label in local_mapping.items()
        ]
        (
            pd.DataFrame(mapping_rows)
            .sort_values("local_label")
            .to_csv(
                cluster_dir / "local_label_mapping.csv",
                index=False,
            )
        )

        summary_row: dict[str, int | str] = {
            "cluster_name": cluster_name,
            "number_classes": len(members),
            "class_members": " | ".join(
                class_names[label]
                for label in members
            ),
        }
        for split, samples in split_samples.items():
            selected = [
                (image_name, local_mapping[label])
                for image_name, label in samples
                if label in local_mapping
            ]
            annotation_path = cluster_dir / f"{split}.txt"
            annotation_path.write_text(
                "".join(
                    f"{image_name} {local_label}\n"
                    for image_name, local_label in selected
                ),
                encoding="utf-8",
            )
            summary_row[f"{split}_samples"] = len(selected)

        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows)


def remove_stale_cluster_subsets(
    output_root: Path,
    active_cluster_names: set[str],
) -> list[str]:
    """Remove obsolete directories previously generated by this script."""
    summary_path = output_root / "cluster_subset_summary.csv"
    if not summary_path.is_file():
        return []

    previous = pd.read_csv(summary_path)
    if "cluster_name" not in previous.columns:
        raise ValueError(
            "Existing cluster_subset_summary.csv has no cluster_name column."
        )

    removed: list[str] = []
    for value in previous["cluster_name"].dropna():
        cluster_name = str(value).strip()
        if not cluster_name or cluster_name in active_cluster_names:
            continue
        if Path(cluster_name).name != cluster_name:
            raise ValueError(
                f"Unsafe cluster directory name in old summary: {cluster_name!r}"
            )

        cluster_dir = output_root / cluster_name
        if not cluster_dir.exists():
            continue
        if not cluster_dir.is_dir() or cluster_dir.is_symlink():
            raise RuntimeError(
                f"Refusing to remove unexpected cluster path: {cluster_dir}"
            )

        children = list(cluster_dir.iterdir())
        unexpected = [
            child.name
            for child in children
            if child.name not in GENERATED_CLUSTER_FILES
            or not child.is_file()
        ]
        if unexpected:
            raise RuntimeError(
                f"Refusing to remove {cluster_dir}; unexpected contents: "
                f"{sorted(unexpected)}"
            )

        for child in children:
            child.unlink()
        cluster_dir.rmdir()
        removed.append(cluster_name)

    return removed


def build_preparation_summary(
    clusters: dict[str, list[int]],
    coarse_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise the frozen fine-to-coarse preparation outputs."""
    return pd.DataFrame([
        {
            "summary_order": 1,
            "item": "original_classes",
            "value": NUM_CLASSES,
        },
        {
            "summary_order": 2,
            "item": "selected_clusters",
            "value": len(clusters),
        },
        {
            "summary_order": 3,
            "item": "clustered_classes",
            "value": sum(
                len(members)
                for members in clusters.values()
            ),
        },
        {
            "summary_order": 4,
            "item": "coarse_classes",
            "value": coarse_mapping["coarse_label"].nunique(),
        },
    ])


def main() -> None:
    """Create the coarse mapping and annotation subsets from frozen input."""
    args = parse_arguments()
    project_root = resolve_project_root(args.project_root)
    paths = resolve_paths(args, project_root)

    class_names = load_class_names(paths["classes"])
    _, clusters = load_frozen_clusters(
        paths["selected_clusters"],
        class_names,
    )
    removed_clusters = remove_stale_cluster_subsets(
        paths["cluster_subsets_dir"],
        set(clusters),
    )
    coarse_mapping = build_coarse_mapping(
        class_names,
        clusters,
    )
    subset_summary = write_cluster_subsets(
        clusters,
        class_names,
        paths["split_root"],
        paths["cluster_subsets_dir"],
    )
    preparation_summary = build_preparation_summary(
        clusters,
        coarse_mapping,
    )

    coarse_mapping.to_csv(
        paths["coarse_mapping"],
        index=False,
    )
    preparation_summary.to_csv(
        paths["preparation_summary"],
        index=False,
    )
    subset_summary.to_csv(
        paths["subset_summary"],
        index=False,
    )

    print("Separability data prepared successfully.")
    print("Selected clusters:", len(clusters))
    if removed_clusters:
        print("Removed stale generated cluster subsets:", removed_clusters)
    print(
        "Clustered classes:",
        sum(len(members) for members in clusters.values()),
    )
    print(
        "Coarse classes:",
        coarse_mapping["coarse_label"].nunique(),
    )
    print("Selected clusters:", paths["selected_clusters"])
    print("Coarse mapping:", paths["coarse_mapping"])
    print("Cluster subsets:", paths["cluster_subsets_dir"])


if __name__ == "__main__":
    main()
