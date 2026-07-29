"""
Evaluate whether selected visual clusters remain difficult under oracle routing.

For each sample whose true label belongs to a selected cluster, oracle routing
reveals the correct cluster and restricts prediction to that cluster's member
classes. This isolates within-cluster separability from coarse-routing errors.

Each model input must be an NPZ file containing:

    image_names:  one-dimensional string array
    true_labels:  one-dimensional integer array
    probabilities: N x 102 probability array

An N x 102 ``logits`` array may be supplied instead of ``probabilities``.
Multiple --model NAME=PATH arguments enable equal-weight probability voting
and consensus-error analysis. Model weights are never tuned on the test set.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


NUM_FINE_CLASSES = 102
SPLITS = ("train", "val", "test")
ENSEMBLE_NAME = "equal_soft_voting"


@dataclass
class ModelOutput:
    name: str
    path: Path
    image_names: np.ndarray
    true_labels: np.ndarray
    probabilities: np.ndarray


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure within-cluster performance under perfect oracle "
            "cluster routing."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=NPZ_PATH",
        help=(
            "Model name and probability NPZ. Repeat this option for "
            "multiple models."
        ),
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="test",
        help="Dataset split represented by the NPZ files.",
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
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Directory produced by 1_prepare_separability_data.py. "
            "Default: outputs/classifier/class_separability."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Result directory. Default: "
            "<analysis-dir>/oracle_<split>."
        ),
    )
    return parser.parse_args()


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
    if not (project_root / "outputs" / "classifier").is_dir():
        raise NotADirectoryError(
            f"Invalid COMP9444_Group project root: {project_root}"
        )
    return project_root


def parse_model_specifications(
    specifications: list[str],
    project_root: Path,
) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    used_names: set[str] = set()

    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                "--model must use NAME=NPZ_PATH format, received: "
                f"{specification!r}"
            )
        name, raw_path = specification.split("=", maxsplit=1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(
                "--model requires a non-empty name and path."
            )
        if name in used_names:
            raise ValueError(f"Duplicate model name: {name!r}")

        path = resolve_path(Path(raw_path), project_root)
        if not path.is_file():
            raise FileNotFoundError(
                f"Probability NPZ not found for {name!r}: {path}"
            )
        parsed.append((name, path))
        used_names.add(name)

    return parsed


def load_mapping(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            "Run 1_prepare_separability_data.py first. "
            f"Mapping not found: {path}"
        )

    dataframe = pd.read_csv(path)
    required = {
        "fine_label",
        "class_name",
        "coarse_label",
        "coarse_name",
        "is_merged",
        "cluster_name",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Coarse mapping is missing columns: {sorted(missing)}"
        )
    dataframe["fine_label"] = pd.to_numeric(
        dataframe["fine_label"],
        errors="raise",
    ).astype(int)
    if set(dataframe["fine_label"]) != set(
        range(NUM_FINE_CLASSES)
    ):
        raise ValueError(
            "Coarse mapping must contain labels 0-101 exactly."
        )
    return dataframe.sort_values("fine_label").reset_index(drop=True)


def selected_clusters(
    mapping: pd.DataFrame,
) -> dict[str, list[int]]:
    values = mapping["is_merged"]
    if values.dtype == bool:
        is_merged = values
    else:
        normalized = values.astype(str).str.strip().str.lower()
        unknown = set(normalized) - {"true", "false", "1", "0"}
        if unknown:
            raise ValueError(
                "is_merged contains invalid Boolean values: "
                f"{sorted(unknown)}"
            )
        is_merged = normalized.isin({"true", "1"})

    merged = mapping[is_merged].copy()
    if merged.empty:
        raise ValueError("Coarse mapping contains no merged clusters.")

    clusters: dict[str, list[int]] = {}
    for cluster_name, rows in merged.groupby(
        "cluster_name",
        sort=False,
    ):
        name = str(cluster_name).strip()
        members = rows.sort_values("fine_label")[
            "fine_label"
        ].astype(int).tolist()
        if not name or len(members) < 2:
            raise ValueError(
                f"Invalid merged cluster {cluster_name!r}."
            )
        clusters[name] = members
    return clusters


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != NUM_FINE_CLASSES:
        raise ValueError(
            "Model probabilities/logits must have shape N x 102, "
            f"not {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Model probabilities contain NaN or infinite values."
        )
    if np.any(values < 0):
        raise ValueError("Probabilities contain negative values.")

    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(
            "Every probability row must have a positive sum."
        )
    return values / row_sums


def load_model_output(name: str, path: Path) -> ModelOutput:
    try:
        archive = np.load(path, allow_pickle=False)
    except ValueError as error:
        raise ValueError(
            f"Could not safely load {path}. Save image_names with "
            "dtype=str rather than dtype=object."
        ) from error

    available = set(archive.files)
    if "image_names" not in available:
        raise ValueError(f"{path} lacks the image_names array.")

    label_key = next(
        (
            key
            for key in ("true_labels", "labels")
            if key in available
        ),
        None,
    )
    if label_key is None:
        raise ValueError(
            f"{path} lacks true_labels (or labels)."
        )

    if "probabilities" in available:
        probabilities = normalize_probabilities(
            archive["probabilities"]
        )
    elif "probs" in available:
        probabilities = normalize_probabilities(archive["probs"])
    elif "logits" in available:
        logits = np.asarray(archive["logits"], dtype=np.float64)
        if (
            logits.ndim != 2
            or logits.shape[1] != NUM_FINE_CLASSES
            or not np.all(np.isfinite(logits))
        ):
            raise ValueError(
                f"Invalid logits array in {path}: {logits.shape}"
            )
        probabilities = stable_softmax(logits)
    else:
        raise ValueError(
            f"{path} needs probabilities, probs, or logits."
        )

    image_names = np.asarray(archive["image_names"]).astype(str)
    true_labels = np.asarray(archive[label_key])
    if image_names.ndim != 1 or true_labels.ndim != 1:
        raise ValueError(
            "image_names and true_labels must be one-dimensional."
        )
    if len(image_names) != len(true_labels):
        raise ValueError(
            f"Length mismatch in {path}: image_names="
            f"{len(image_names)}, labels={len(true_labels)}."
        )
    if len(image_names) != len(probabilities):
        raise ValueError(
            f"Length mismatch in {path}: image_names="
            f"{len(image_names)}, probability rows="
            f"{len(probabilities)}."
        )
    if len(set(image_names.tolist())) != len(image_names):
        raise ValueError(f"{path} contains duplicate image names.")

    if not np.issubdtype(true_labels.dtype, np.integer):
        if not np.all(
            np.equal(true_labels, np.rint(true_labels))
        ):
            raise ValueError(
                f"{path} contains non-integer true labels."
            )
    true_labels = true_labels.astype(np.int64)
    if np.any(
        (true_labels < 0)
        | (true_labels >= NUM_FINE_CLASSES)
    ):
        raise ValueError(
            f"{path} contains true labels outside 0-101."
        )

    return ModelOutput(
        name=name,
        path=path,
        image_names=image_names,
        true_labels=true_labels,
        probabilities=probabilities,
    )


def align_model_outputs(
    models: list[ModelOutput],
) -> list[ModelOutput]:
    reference = models[0]
    reference_names = reference.image_names.tolist()
    reference_set = set(reference_names)
    aligned = [reference]

    for model in models[1:]:
        model_set = set(model.image_names.tolist())
        if model_set != reference_set:
            missing = list(reference_set - model_set)[:5]
            extra = list(model_set - reference_set)[:5]
            raise ValueError(
                f"Image sets differ for {model.name!r}. "
                f"Missing examples={missing}, extra examples={extra}."
            )

        positions = {
            image_name: index
            for index, image_name in enumerate(model.image_names)
        }
        order = np.asarray(
            [positions[name] for name in reference_names],
            dtype=np.int64,
        )
        labels = model.true_labels[order]
        if not np.array_equal(labels, reference.true_labels):
            mismatch = int(
                np.flatnonzero(labels != reference.true_labels)[0]
            )
            raise ValueError(
                f"True-label mismatch for {model.name!r} at "
                f"{reference_names[mismatch]!r}."
            )

        aligned.append(ModelOutput(
            name=model.name,
            path=model.path,
            image_names=model.image_names[order],
            true_labels=labels,
            probabilities=model.probabilities[order],
        ))

    return aligned


def geometric_mean_from_recalls(recalls: np.ndarray) -> float:
    recalls = np.asarray(recalls, dtype=np.float64)
    if len(recalls) == 0 or np.any(recalls <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def calculate_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    labels: list[int],
) -> dict[str, float | int]:
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=labels,
            zero_division=0,
        )
    )
    return {
        "number_samples": int(len(true_labels)),
        "number_correct": int(
            np.sum(true_labels == predicted_labels)
        ),
        "number_errors": int(
            np.sum(true_labels != predicted_labels)
        ),
        "accuracy": float(
            accuracy_score(true_labels, predicted_labels)
        ),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "g_mean": geometric_mean_from_recalls(recall),
    }


def restrict_probabilities(
    probabilities: np.ndarray,
    members: list[int],
) -> np.ndarray:
    local = probabilities[:, members]
    sums = local.sum(axis=1, keepdims=True)
    return np.divide(
        local,
        sums,
        out=np.full_like(
            local,
            1.0 / len(members),
            dtype=np.float64,
        ),
        where=sums > 0,
    )


def local_to_global_predictions(
    local_probabilities: np.ndarray,
    members: list[int],
) -> np.ndarray:
    member_array = np.asarray(members, dtype=np.int64)
    return member_array[np.argmax(local_probabilities, axis=1)]


def sanitize_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("._") or "unnamed"


def save_cluster_confusion(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    members: list[int],
    label_to_name: dict[int, str],
    path: Path,
) -> None:
    names = [label_to_name[label] for label in members]
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=members,
    )
    pd.DataFrame(
        matrix,
        index=names,
        columns=names,
    ).to_csv(path, index=True)


def main() -> None:
    args = parse_arguments()
    project_root = resolve_project_root(args.project_root)
    analysis_dir = (
        project_root
        / "outputs"
        / "classifier"
        / "class_separability"
        if args.analysis_dir is None
        else resolve_path(args.analysis_dir, project_root)
    )
    output_dir = (
        analysis_dir / f"oracle_{args.split}"
        if args.output_dir is None
        else resolve_path(args.output_dir, project_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir = output_dir / "cluster_confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(
        analysis_dir / "coarse_label_mapping.csv"
    )
    clusters = selected_clusters(mapping)
    label_to_name = mapping.set_index("fine_label")[
        "class_name"
    ].to_dict()
    model_specs = parse_model_specifications(
        args.model,
        project_root,
    )
    models = align_model_outputs([
        load_model_output(name, path)
        for name, path in model_specs
    ])

    image_names = models[0].image_names
    true_labels = models[0].true_labels
    all_cluster_labels = sorted({
        label
        for members in clusters.values()
        for label in members
    })
    cluster_for_label = {
        label: cluster_name
        for cluster_name, members in clusters.items()
        for label in members
    }
    hard_mask = np.isin(true_labels, all_cluster_labels)
    if not np.any(hard_mask):
        raise ValueError(
            f"No selected-cluster samples exist in the {args.split} "
            "probability files."
        )

    full_probabilities = {
        model.name: model.probabilities
        for model in models
    }
    full_probabilities[ENSEMBLE_NAME] = np.mean(
        np.stack(
            [model.probabilities for model in models],
            axis=0,
        ),
        axis=0,
    )
    method_names = [
        model.name
        for model in models
    ] + [ENSEMBLE_NAME]

    oracle_predictions = {
        method_name: np.full(
            len(true_labels),
            -1,
            dtype=np.int64,
        )
        for method_name in method_names
    }
    cluster_rows: list[dict[str, int | float | str]] = []
    per_class_rows: list[dict[str, int | float | str]] = []

    for cluster_order, (cluster_name, members) in enumerate(
        clusters.items(),
        start=1,
    ):
        mask = np.isin(true_labels, members)
        if not np.any(mask):
            raise ValueError(
                f"Cluster {cluster_name!r} has no {args.split} samples."
            )
        cluster_true = true_labels[mask]

        local_probabilities: dict[str, np.ndarray] = {
            model.name: restrict_probabilities(
                model.probabilities[mask],
                members,
            )
            for model in models
        }
        local_probabilities[ENSEMBLE_NAME] = np.mean(
            np.stack(
                [
                    local_probabilities[model.name]
                    for model in models
                ],
                axis=0,
            ),
            axis=0,
        )

        for method_order, method_name in enumerate(
            method_names,
            start=1,
        ):
            full_predictions = np.argmax(
                full_probabilities[method_name][mask],
                axis=1,
            )
            oracle_pred = local_to_global_predictions(
                local_probabilities[method_name],
                members,
            )
            oracle_predictions[method_name][mask] = oracle_pred

            full_metrics = calculate_metrics(
                cluster_true,
                full_predictions,
                members,
            )
            oracle_metrics = calculate_metrics(
                cluster_true,
                oracle_pred,
                members,
            )
            cluster_rows.append({
                "cluster_order": cluster_order,
                "method_order": method_order,
                "split": args.split,
                "cluster_name": cluster_name,
                "number_classes": len(members),
                "class_members": " | ".join(
                    label_to_name[label]
                    for label in members
                ),
                "method": method_name,
                **{
                    f"full_{key}": value
                    for key, value in full_metrics.items()
                },
                **{
                    f"oracle_{key}": value
                    for key, value in oracle_metrics.items()
                },
                "oracle_accuracy_gain": (
                    float(oracle_metrics["accuracy"])
                    - float(full_metrics["accuracy"])
                ),
                "oracle_macro_f1_gain": (
                    float(oracle_metrics["macro_f1"])
                    - float(full_metrics["macro_f1"])
                ),
            })

            precision, recall, f1, support = (
                precision_recall_fscore_support(
                    cluster_true,
                    oracle_pred,
                    labels=members,
                    zero_division=0,
                )
            )
            for index, label in enumerate(members):
                per_class_rows.append({
                    "cluster_order": cluster_order,
                    "method_order": method_order,
                    "split": args.split,
                    "cluster_name": cluster_name,
                    "method": method_name,
                    "label": label,
                    "class_name": label_to_name[label],
                    "support": int(support[index]),
                    "oracle_precision": float(precision[index]),
                    "oracle_recall": float(recall[index]),
                    "oracle_f1": float(f1[index]),
                    "oracle_error_rate": (
                        1.0 - float(recall[index])
                    ),
                })

            save_cluster_confusion(
                cluster_true,
                oracle_pred,
                members,
                label_to_name,
                confusion_dir
                / (
                    f"{cluster_order:02d}_"
                    f"{sanitize_name(cluster_name)}__"
                    f"{sanitize_name(method_name)}.csv"
                ),
            )

    hard_true = true_labels[hard_mask]
    overall_rows: list[dict[str, int | float | str]] = []
    cluster_results = pd.DataFrame(cluster_rows)

    for method_order, method_name in enumerate(
        method_names,
        start=1,
    ):
        full_pred = np.argmax(
            full_probabilities[method_name][hard_mask],
            axis=1,
        )
        oracle_pred = oracle_predictions[method_name][hard_mask]
        full_metrics = calculate_metrics(
            hard_true,
            full_pred,
            all_cluster_labels,
        )
        oracle_metrics = calculate_metrics(
            hard_true,
            oracle_pred,
            all_cluster_labels,
        )
        method_cluster_rows = cluster_results[
            cluster_results["method"] == method_name
        ]
        overall_rows.append({
            "method_order": method_order,
            "split": args.split,
            "method": method_name,
            "number_clusters": len(clusters),
            "number_classes": len(all_cluster_labels),
            **{
                f"full_{key}": value
                for key, value in full_metrics.items()
            },
            **{
                f"oracle_{key}": value
                for key, value in oracle_metrics.items()
            },
            "oracle_accuracy_gain": (
                float(oracle_metrics["accuracy"])
                - float(full_metrics["accuracy"])
            ),
            "oracle_macro_f1_gain": (
                float(oracle_metrics["macro_f1"])
                - float(full_metrics["macro_f1"])
            ),
            "mean_cluster_oracle_macro_f1": float(
                method_cluster_rows[
                    "oracle_macro_f1"
                ].mean()
            ),
        })

    prediction_rows = pd.DataFrame({
        "split": args.split,
        "image_name": image_names[hard_mask],
        "true_label": hard_true,
        "true_class": [
            label_to_name[label]
            for label in hard_true
        ],
        "oracle_cluster": [
            cluster_for_label[label]
            for label in hard_true
        ],
    })
    individual_correct_columns: list[str] = []
    individual_prediction_columns: list[str] = []

    for model in models:
        safe_name = sanitize_name(model.name)
        prediction_column = f"{safe_name}_oracle_label"
        class_column = f"{safe_name}_oracle_class"
        correct_column = f"{safe_name}_oracle_correct"
        predictions = oracle_predictions[model.name][hard_mask]
        prediction_rows[prediction_column] = predictions
        prediction_rows[class_column] = [
            label_to_name[label]
            for label in predictions
        ]
        prediction_rows[correct_column] = (
            predictions == hard_true
        )
        individual_prediction_columns.append(prediction_column)
        individual_correct_columns.append(correct_column)

    ensemble_predictions = oracle_predictions[
        ENSEMBLE_NAME
    ][hard_mask]
    prediction_rows["ensemble_oracle_label"] = (
        ensemble_predictions
    )
    prediction_rows["ensemble_oracle_class"] = [
        label_to_name[label]
        for label in ensemble_predictions
    ]
    prediction_rows["ensemble_oracle_correct"] = (
        ensemble_predictions == hard_true
    )

    all_individual_wrong = ~prediction_rows[
        individual_correct_columns
    ].any(axis=1)
    consensus_errors = prediction_rows[
        all_individual_wrong
    ].copy()
    predicted_values = consensus_errors[
        individual_prediction_columns
    ]
    consensus_errors["all_models_same_wrong_label"] = (
        predicted_values.nunique(axis=1) == 1
    )
    consensus_errors["shared_wrong_label"] = np.where(
        consensus_errors["all_models_same_wrong_label"],
        predicted_values.iloc[:, 0],
        -1,
    )
    consensus_errors["shared_wrong_class"] = [
        (
            label_to_name[int(label)]
            if int(label) >= 0
            else ""
        )
        for label in consensus_errors["shared_wrong_label"]
    ]
    consensus_errors.insert(
        0,
        "consensus_error_rank",
        range(1, len(consensus_errors) + 1),
    )

    per_class_results = pd.DataFrame(per_class_rows).sort_values(
        [
            "method_order",
            "oracle_f1",
            "oracle_recall",
            "support",
        ],
        ascending=[True, True, True, False],
    )
    cluster_results = cluster_results.sort_values(
        ["method_order", "oracle_macro_f1", "cluster_order"],
        ascending=[True, True, True],
    )
    overall_results = pd.DataFrame(overall_rows)

    output_paths = {
        "overall": (
            output_dir
            / f"{args.split}_oracle_overall_results.csv"
        ),
        "clusters": (
            output_dir
            / f"{args.split}_oracle_cluster_results.csv"
        ),
        "per_class": (
            output_dir
            / f"{args.split}_oracle_per_class_results.csv"
        ),
        "predictions": (
            output_dir
            / f"{args.split}_oracle_all_predictions.csv"
        ),
        "consensus_errors": (
            output_dir
            / f"{args.split}_oracle_consensus_errors.csv"
        ),
    }
    overall_results.to_csv(
        output_paths["overall"],
        index=False,
        float_format="%.6f",
    )
    cluster_results.to_csv(
        output_paths["clusters"],
        index=False,
        float_format="%.6f",
    )
    per_class_results.to_csv(
        output_paths["per_class"],
        index=False,
        float_format="%.6f",
    )
    prediction_rows.to_csv(
        output_paths["predictions"],
        index=False,
    )
    consensus_errors.to_csv(
        output_paths["consensus_errors"],
        index=False,
    )

    print("Oracle within-cluster evaluation complete.")
    print("Split:", args.split)
    print("Models:", ", ".join(model.name for model in models))
    print("Selected clusters:", len(clusters))
    print("Clustered classes:", len(all_cluster_labels))
    print("Cluster samples:", int(hard_mask.sum()))
    print("Consensus errors:", len(consensus_errors))
    print("Output directory:", output_dir)
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    print("Confusion matrices:", confusion_dir)


if __name__ == "__main__":
    main()
