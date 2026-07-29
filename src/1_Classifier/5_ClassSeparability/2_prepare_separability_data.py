"""
Prepare fixed label mappings and annotation subsets for IP102 separability tests.

This script uses validation results only to:

1. Convert the 102 original labels into 75 coarse labels by merging selected
   high-confidence confusion clusters.
2. Divide original classes into cluster_hard, easy, diffuse_hard, and
   uncertain groups.
3. Create train/val/test annotation subsets for every selected cluster without
   copying any image files.

With the current validation-cluster CSV and the default minimum visual score
of 50.0, the following 35 original classes are merged into 8 coarse classes.
All label IDs below are the zero-based labels used by PyTorch:

1. cutworms (New coarse class 0)
   - Label 18: black cutworm
   - Label 19: large cutworm
   - Label 20: yellow cutworm

2. plant_hoppers (New coarse class 1)
   - Label 7: brown plant hopper
   - Label 8: white backed plant hopper
   - Label 9: small brown plant hopper

3. fruit_flies (New coarse class 2)
   - Label 83: Tetradacus c Bactrocera minax
   - Label 84: Dacus dorsalis(Hendel)
   - Label 85: Bactrocera tsuneonis

4. army_worms (New coarse class 3)
   - Label 23: army worm
   - Label 38: cabbage army worm
   - Label 39: beet army worm
   - Label 86: Prodenia litura
   - Label 45: flax budworm

5. blister_beetles (New coarse class 4)
   - Label 49: lytta polita
   - Label 50: legume blister beetle
   - Label 51: blister beetle

6. aphids (New coarse class 5)
   - Label 24: aphids
   - Label 27: english grain aphid
   - Label 28: green bug
   - Label 29: bird cherry-oataphid
   - Label 52: therioaphis maculata Buckton
   - Label 89: Toxoptera citricidus
   - Label 90: Toxoptera aurantii
   - Label 91: Aphis citricola Vander Goot

7. mites (New coarse class 6)
   - Label 21: red spider
   - Label 32: longlegged spider mite
   - Label 31: penthaleus major
   - Label 74: Panonchus citri McGregor
   - Label 60: Colomerus vitis
   - Label 63: Polyphagotars onemus latus

8. plant_bugs_and_miridae (New coarse class 7)
   - Label 57: Apolygus lucorum
   - Label 46: alfalfa plant bug
   - Label 47: tarnished plant bug
   - Label 70: Miridae

The other 67 original classes remain separate singleton coarse classes:
67 singleton classes + 8 merged classes = 75 coarse classes.

You can check these new 75 coarse classes number in :
COMP9444_Group/outputs/classifier/class_separability/coarse_label_mapping.csv
ex:
    18,19,black cutworm,0,cutworms,True,cutworms
    19,20,large cutworm,0,cutworms,True,cutworms
    20,21,yellow cutworm,0,cutworms,True,cutworms
shows that the original classes 18,19,20 are merged into a new coarse class 0 named cutworms.

These lists document the current default configuration; they are not
hard-coded into the mapping logic. The validation cluster CSV and
--minimum-visual-score remain the source of truth when the script runs.

The produced definitions must be frozen before any test-set evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NUM_CLASSES = 102
DEFAULT_MINIMUM_VISUAL_SCORE = 50.0
DEFAULT_EASY_MIN_SUPPORT = 20
DEFAULT_EASY_MIN_F1 = 0.70
DEFAULT_EASY_MIN_RECALL = 0.70
DEFAULT_EASY_MAX_DOMINANT_CONFUSION = 0.15
DEFAULT_HARD_MAX_F1 = 0.60
DEFAULT_HARD_MAX_RECALL = 0.60


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fixed 102-to-coarse mappings, class-difficulty "
            "groups, and cluster annotation subsets."
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
        "--cluster-csv",
        type=Path,
        default=None,
        help=(
            "Validation high-confidence visual-cluster CSV. Default: "
            "outputs/classifier/resnet50/analyze_confusion_matrix/"
            "val_high_confidence_visual_clusters.csv."
        ),
    )
    parser.add_argument(
        "--val-per-class",
        type=Path,
        default=None,
        help=(
            "Validation per-class metrics CSV. Both "
            "val_confusion_per_class.csv and "
            "val_classification_report.csv formats are accepted."
        ),
    )
    parser.add_argument(
        "--val-confusion-matrix",
        type=Path,
        default=None,
        help="Validation 102x102 confusion-matrix CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Analysis output directory. Default: "
            "outputs/classifier/class_separability."
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
    parser.add_argument(
        "--minimum-visual-score",
        type=float,
        default=DEFAULT_MINIMUM_VISUAL_SCORE,
    )
    parser.add_argument(
        "--easy-min-support",
        type=int,
        default=DEFAULT_EASY_MIN_SUPPORT,
    )
    parser.add_argument(
        "--easy-min-f1",
        type=float,
        default=DEFAULT_EASY_MIN_F1,
    )
    parser.add_argument(
        "--easy-min-recall",
        type=float,
        default=DEFAULT_EASY_MIN_RECALL,
    )
    parser.add_argument(
        "--easy-max-dominant-confusion",
        type=float,
        default=DEFAULT_EASY_MAX_DOMINANT_CONFUSION,
    )
    parser.add_argument(
        "--hard-max-f1",
        type=float,
        default=DEFAULT_HARD_MAX_F1,
    )
    parser.add_argument(
        "--hard-max-recall",
        type=float,
        default=DEFAULT_HARD_MAX_RECALL,
    )

    args = parser.parse_args()

    probability_arguments = [
        args.easy_min_f1,
        args.easy_min_recall,
        args.easy_max_dominant_confusion,
        args.hard_max_f1,
        args.hard_max_recall,
    ]
    if args.minimum_visual_score < 0:
        parser.error("--minimum-visual-score cannot be negative.")
    if args.easy_min_support < 0:
        parser.error("--easy-min-support cannot be negative.")
    if any(value < 0 or value > 1 for value in probability_arguments):
        parser.error(
            "F1, recall, and confusion thresholds must be in [0, 1]."
        )

    return args


def resolve_path(path: Path, project_root: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def resolve_project_root(requested: Path | None) -> Path:
    if requested is None:
        project_root = Path(__file__).resolve().parents[3]
    else:
        project_root = requested.expanduser().resolve()

    required = [
        project_root
        / "datasets"
        / "raw"
        / "Classification"
        / "classes.txt",
        project_root
        / "datasets"
        / "raw"
        / "Classification"
        / "ip102_v1.1"
        / "train.txt",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Invalid project root or missing IP102 files:\n"
            f"{text}"
        )

    return project_root


def resolve_paths(
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Path]:
    baseline_analysis = (
        project_root
        / "outputs"
        / "classifier"
        / "resnet50"
        / "analyze_confusion_matrix"
    )
    baseline_output = (
        project_root
        / "outputs"
        / "classifier"
        / "resnet50"
    )

    cluster_csv = (
        baseline_analysis
        / "val_high_confidence_visual_clusters.csv"
        if args.cluster_csv is None
        else resolve_path(args.cluster_csv, project_root)
    )

    if args.val_per_class is None:
        per_class_candidates = [
            baseline_analysis / "val_confusion_per_class.csv",
            baseline_output / "val_classification_report.csv",
        ]
        val_per_class = next(
            (
                path
                for path in per_class_candidates
                if path.is_file()
            ),
            per_class_candidates[0],
        )
    else:
        val_per_class = resolve_path(
            args.val_per_class,
            project_root,
        )

    if args.val_confusion_matrix is None:
        confusion_candidates = [
            baseline_output / "val_confusion_matrix.csv",
            baseline_analysis / "val_confusion_matrix.csv",
        ]
        val_confusion = next(
            (
                path
                for path in confusion_candidates
                if path.is_file()
            ),
            confusion_candidates[0],
        )
    else:
        val_confusion = resolve_path(
            args.val_confusion_matrix,
            project_root,
        )

    output_dir = (
        project_root
        / "outputs"
        / "classifier"
        / "class_separability"
        if args.output_dir is None
        else resolve_path(args.output_dir, project_root)
    )
    cluster_subsets_dir = (
        project_root
        / "datasets"
        / "processed"
        / "ip102_cluster_subsets"
        if args.cluster_subsets_dir is None
        else resolve_path(args.cluster_subsets_dir, project_root)
    )

    required_inputs = [
        cluster_csv,
        val_per_class,
        val_confusion,
    ]
    missing = [path for path in required_inputs if not path.is_file()]
    if missing:
        text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required validation analysis files are missing:\n"
            f"{text}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_subsets_dir.mkdir(parents=True, exist_ok=True)

    classification_root = (
        project_root
        / "datasets"
        / "raw"
        / "Classification"
    )

    return {
        "classes": classification_root / "classes.txt",
        "split_root": classification_root / "ip102_v1.1",
        "cluster_csv": cluster_csv,
        "val_per_class": val_per_class,
        "val_confusion": val_confusion,
        "output_dir": output_dir,
        "cluster_subsets_dir": cluster_subsets_dir,
        "coarse_mapping": output_dir / "coarse_label_mapping.csv",
        "difficulty_groups": (
            output_dir / "class_difficulty_groups.csv"
        ),
        "selected_clusters": output_dir / "selected_clusters.csv",
        "preparation_summary": (
            output_dir / "separability_preparation_summary.csv"
        ),
    }


def load_class_names(classes_path: Path) -> list[str]:
    label_to_name: dict[int, str] = {}

    with classes_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                class_number, class_name = stripped.split(
                    maxsplit=1
                )
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
    if pd.isna(value):
        return []
    return [
        member.strip()
        for member in str(value).split("|")
        if member.strip()
    ]


def load_selected_clusters(
    csv_path: Path,
    class_names: list[str],
    minimum_visual_score: float,
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    dataframe = pd.read_csv(csv_path)
    required = {
        "cluster_name",
        "class_members",
        "visual_confusion_score",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Cluster CSV is missing columns: {sorted(missing)}"
        )

    dataframe["visual_confusion_score"] = pd.to_numeric(
        dataframe["visual_confusion_score"],
        errors="raise",
    )
    if "split" in dataframe.columns:
        dataframe = dataframe[
            dataframe["split"].astype(str).str.lower() == "val"
        ].copy()
        if dataframe.empty:
            raise ValueError(
                "Cluster CSV contains no validation rows."
            )

    selected = dataframe[
        dataframe["visual_confusion_score"]
        >= minimum_visual_score
    ].copy()
    if selected.empty:
        raise ValueError(
            "No clusters meet the visual-score threshold."
        )

    if "visual_cluster_rank" in selected.columns:
        selected = selected.sort_values(
            ["visual_cluster_rank", "visual_confusion_score"],
            ascending=[True, False],
        )
    else:
        selected = selected.sort_values(
            "visual_confusion_score",
            ascending=False,
        )
    selected = selected.reset_index(drop=True)

    name_to_label = {
        name: label
        for label, name in enumerate(class_names)
    }
    clusters: dict[str, list[int]] = {}
    used_labels: dict[int, str] = {}

    for row_number, row in selected.iterrows():
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
                f"Cluster {cluster_name!r} needs at least 2 classes."
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

    return selected, clusters


def build_coarse_mapping(
    class_names: list[str],
    clusters: dict[str, list[int]],
) -> pd.DataFrame:
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
    number_coarse = mapping["coarse_label"].nunique()
    expected_coarse = (
        NUM_CLASSES
        - sum(len(members) for members in clusters.values())
        + len(clusters)
    )
    if number_coarse != expected_coarse:
        raise RuntimeError(
            f"Expected {expected_coarse} coarse labels, "
            f"constructed {number_coarse}."
        )

    return mapping


def load_per_class_metrics(
    csv_path: Path,
    class_names: list[str],
) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)

    aliases = {
        "label": ["label", "fine_label"],
        "class_name": ["insect_name", "class_name"],
        "support": ["support"],
        "precision": ["precision"],
        "recall": ["recall"],
        "f1": ["f1_score", "f1-score", "f1"],
    }
    resolved: dict[str, str] = {}

    for canonical, candidates in aliases.items():
        match = next(
            (
                column
                for column in candidates
                if column in dataframe.columns
            ),
            None,
        )
        if match is None and canonical == "class_name":
            continue
        if match is None:
            raise ValueError(
                f"Per-class CSV lacks a {canonical!r} column. "
                f"Accepted names: {candidates}"
            )
        resolved[canonical] = match

    normalized = pd.DataFrame({
        "label": pd.to_numeric(
            dataframe[resolved["label"]],
            errors="raise",
        ).astype(int),
        "support": pd.to_numeric(
            dataframe[resolved["support"]],
            errors="raise",
        ).astype(int),
        "precision": pd.to_numeric(
            dataframe[resolved["precision"]],
            errors="raise",
        ),
        "recall": pd.to_numeric(
            dataframe[resolved["recall"]],
            errors="raise",
        ),
        "f1": pd.to_numeric(
            dataframe[resolved["f1"]],
            errors="raise",
        ),
    })

    if "class_name" in resolved:
        normalized["class_name"] = (
            dataframe[resolved["class_name"]]
            .astype(str)
            .str.strip()
        )
    else:
        normalized["class_name"] = normalized["label"].map(
            dict(enumerate(class_names))
        )

    expected = set(range(NUM_CLASSES))
    observed = set(normalized["label"])
    if observed != expected:
        raise ValueError(
            "Per-class CSV must contain all 102 labels. "
            f"Missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    normalized = normalized.sort_values("label").reset_index(
        drop=True
    )
    expected_names = pd.Series(class_names, name="class_name")
    if not normalized["class_name"].reset_index(
        drop=True
    ).equals(expected_names):
        mismatches = normalized[
            normalized["class_name"]
            != expected_names
        ][["label", "class_name"]]
        raise ValueError(
            "Per-class class names do not match classes.txt:\n"
            f"{mismatches.head(10).to_string(index=False)}"
        )

    return normalized


def load_confusion_matrix(
    csv_path: Path,
    class_names: list[str],
) -> np.ndarray:
    dataframe = pd.read_csv(csv_path, index_col=0)
    dataframe.index = dataframe.index.astype(str).str.strip()
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    if dataframe.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(
            "Validation confusion matrix must be 102x102, "
            f"not {dataframe.shape}."
        )
    if dataframe.index.tolist() != class_names:
        raise ValueError(
            "Confusion-matrix rows do not match classes.txt order."
        )
    if dataframe.columns.tolist() != class_names:
        raise ValueError(
            "Confusion-matrix columns do not match classes.txt order."
        )

    try:
        matrix = dataframe.to_numpy(dtype=np.int64)
    except ValueError as error:
        raise ValueError(
            "Confusion matrix contains non-numeric values."
        ) from error
    if np.any(matrix < 0):
        raise ValueError("Confusion matrix contains negative counts.")

    return matrix


def build_difficulty_groups(
    per_class: pd.DataFrame,
    confusion: np.ndarray,
    coarse_mapping: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    mapping = coarse_mapping.set_index("fine_label")
    rows: list[dict[str, int | float | str | bool]] = []

    for row in per_class.itertuples(index=False):
        label = int(row.label)
        support = int(confusion[label].sum())
        if support != int(row.support):
            raise ValueError(
                f"Support mismatch for label {label}: "
                f"report={row.support}, confusion={support}."
            )

        off_diagonal = confusion[label].copy()
        off_diagonal[label] = 0
        dominant_count = int(off_diagonal.max())
        dominant_label = int(off_diagonal.argmax())
        dominant_rate = (
            dominant_count / support
            if support > 0
            else 0.0
        )

        mapping_row = mapping.loc[label]
        is_cluster_hard = bool(mapping_row["is_merged"])

        if is_cluster_hard:
            difficulty_group = "cluster_hard"
            reason = "member_of_selected_visual_cluster"
        elif (
            support >= args.easy_min_support
            and float(row.f1) >= args.easy_min_f1
            and float(row.recall) >= args.easy_min_recall
            and dominant_rate
            < args.easy_max_dominant_confusion
        ):
            difficulty_group = "easy"
            reason = "stable_high_validation_performance"
        elif (
            float(row.f1) < args.hard_max_f1
            or float(row.recall) < args.hard_max_recall
        ):
            difficulty_group = "diffuse_hard"
            reason = "low_validation_metric_without_selected_cluster"
        else:
            difficulty_group = "uncertain"
            reason = "intermediate_or_low_support"

        rows.append({
            "label": label,
            "class_number": label + 1,
            "class_name": row.class_name,
            "cluster_name": mapping_row["cluster_name"],
            "coarse_label": int(mapping_row["coarse_label"]),
            "coarse_name": mapping_row["coarse_name"],
            "is_merged": is_cluster_hard,
            "val_support": support,
            "val_precision": float(row.precision),
            "val_recall": float(row.recall),
            "val_f1": float(row.f1),
            "dominant_wrong_label": (
                dominant_label
                if dominant_count > 0
                else -1
            ),
            "dominant_wrong_class": (
                str(mapping.loc[dominant_label, "class_name"])
                if dominant_count > 0
                else ""
            ),
            "dominant_wrong_count": dominant_count,
            "dominant_confusion_rate": dominant_rate,
            "difficulty_group": difficulty_group,
            "group_reason": reason,
        })

    result = pd.DataFrame(rows)
    order = {
        "easy": 1,
        "cluster_hard": 2,
        "diffuse_hard": 3,
        "uncertain": 4,
    }
    result.insert(
        0,
        "difficulty_group_order",
        result["difficulty_group"].map(order),
    )
    return result.sort_values(
        ["difficulty_group_order", "val_f1", "label"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def read_split_annotations(path: Path) -> list[tuple[str, int]]:
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
    split_samples = {
        split: read_split_annotations(
            split_root / f"{split}.txt"
        )
        for split in ("train", "val", "test")
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
        pd.DataFrame(mapping_rows).sort_values(
            "local_label"
        ).to_csv(
            cluster_dir / "local_label_mapping.csv",
            index=False,
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


def main() -> None:
    args = parse_arguments()
    project_root = resolve_project_root(args.project_root)
    paths = resolve_paths(args, project_root)

    class_names = load_class_names(paths["classes"])
    selected_clusters, clusters = load_selected_clusters(
        paths["cluster_csv"],
        class_names,
        args.minimum_visual_score,
    )
    coarse_mapping = build_coarse_mapping(
        class_names,
        clusters,
    )
    per_class = load_per_class_metrics(
        paths["val_per_class"],
        class_names,
    )
    confusion = load_confusion_matrix(
        paths["val_confusion"],
        class_names,
    )
    difficulty_groups = build_difficulty_groups(
        per_class,
        confusion,
        coarse_mapping,
        args,
    )
    subset_summary = write_cluster_subsets(
        clusters,
        class_names,
        paths["split_root"],
        paths["cluster_subsets_dir"],
    )

    coarse_mapping.to_csv(
        paths["coarse_mapping"],
        index=False,
    )
    difficulty_groups.to_csv(
        paths["difficulty_groups"],
        index=False,
        float_format="%.6f",
    )
    selected_clusters.to_csv(
        paths["selected_clusters"],
        index=False,
        float_format="%.6f",
    )

    group_counts = (
        difficulty_groups["difficulty_group"]
        .value_counts()
        .to_dict()
    )
    summary_rows = [
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
    ]
    for index, group_name in enumerate(
        ("easy", "cluster_hard", "diffuse_hard", "uncertain"),
        start=5,
    ):
        summary_rows.append({
            "summary_order": index,
            "item": f"{group_name}_classes",
            "value": int(group_counts.get(group_name, 0)),
        })

    pd.DataFrame(summary_rows).to_csv(
        paths["preparation_summary"],
        index=False,
    )
    subset_summary.to_csv(
        paths["cluster_subsets_dir"]
        / "cluster_subset_summary.csv",
        index=False,
    )

    print("Separability definitions prepared successfully.")
    print("Project root:", project_root)
    print("Selected clusters:", len(clusters))
    print(
        "Clustered classes:",
        sum(len(members) for members in clusters.values()),
    )
    print(
        "Coarse classes:",
        coarse_mapping["coarse_label"].nunique(),
    )
    print("Difficulty groups:", group_counts)
    print("Coarse mapping:", paths["coarse_mapping"])
    print("Difficulty groups:", paths["difficulty_groups"])
    print("Selected clusters:", paths["selected_clusters"])
    print("Cluster subsets:", paths["cluster_subsets_dir"])


if __name__ == "__main__":
    main()