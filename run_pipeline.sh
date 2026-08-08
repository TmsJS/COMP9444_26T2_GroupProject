#!/usr/bin/env bash

set -Eeuo pipefail


usage() {
    cat <<'EOF'
Usage: ./run_pipeline.sh [options]

Run the complete IP102 classification workflow from the project root.

Options:
  --skip-training      Use existing checkpoints instead of retraining models.
  --skip-download      Do not run the dataset download/extraction script.
  --skip-separability  Skip confusion analysis, coarse evaluation, and oracle comparison.
  --dry-run            Print commands in execution order without running them.
  -h, --help           Show this help message.

Examples:
  ./run_pipeline.sh
  ./run_pipeline.sh --skip-training --skip-download
  ./run_pipeline.sh --dry-run
EOF
}


SKIP_TRAINING=0
SKIP_DOWNLOAD=0
SKIP_SEPARABILITY=0
DRY_RUN=0

while (($# > 0)); do
    case "$1" in
        --skip-training)
            SKIP_TRAINING=1
            ;;
        --skip-download)
            SKIP_DOWNLOAD=1
            ;;
        --skip-separability)
            SKIP_SEPARABILITY=1
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done


SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"
PROJECT_ROOT="$SCRIPT_DIR"

if [[ ! -d "$PROJECT_ROOT/src/0_Data_Prep" ]] \
    || [[ ! -d "$PROJECT_ROOT/src/1_Classifier" ]]; then
    echo "Project src directory was not found beside this script:" >&2
    echo "$PROJECT_ROOT" >&2
    exit 1
fi

cd "$PROJECT_ROOT"


select_python() {
    if [[ -n "${VIRTUAL_ENV:-}" ]] \
        && [[ -x "$VIRTUAL_ENV/bin/python" ]]; then
        printf '%s\n' "$VIRTUAL_ENV/bin/python"
    elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
        printf '%s\n' "$PROJECT_ROOT/.venv/bin/python"
    elif [[ -x "$PROJECT_ROOT/env/bin/python" ]]; then
        printf '%s\n' "$PROJECT_ROOT/env/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        echo "No usable Python interpreter was found." >&2
        return 1
    fi
}


PYTHON_BIN="$(select_python)"
CURRENT_STEP="pipeline initialisation"

if ((DRY_RUN == 0)); then
    LOG_DIR="$PROJECT_ROOT/outputs/pipeline_logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/pipeline_$(date '+%Y%m%d_%H%M%S').log"
    exec > >(tee -a "$LOG_FILE") 2>&1
else
    LOG_FILE="(disabled for dry-run)"
fi


on_error() {
    local exit_code=$?
    echo
    echo "Pipeline failed during: $CURRENT_STEP" >&2
    echo "Exit code: $exit_code" >&2
    exit "$exit_code"
}

trap on_error ERR


print_command() {
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
}


run_command() {
    if ((DRY_RUN == 1)); then
        print_command "$@"
    else
        "$@"
    fi
}


run_step() {
    CURRENT_STEP="$1"
    shift

    echo
    echo "[$(date '+%H:%M:%S')] $CURRENT_STEP"
    run_command "$@"
}


require_file() {
    local path=$1
    local description=$2

    if ((DRY_RUN == 1)); then
        return
    fi

    if [[ ! -f "$path" ]]; then
        echo "$description was not found:" >&2
        echo "$path" >&2
        return 1
    fi
}


require_checkpoint() {
    local checkpoint=$1

    require_file "$checkpoint" "Model checkpoint"

    if ((DRY_RUN == 1)); then
        return
    fi

    local first_line=""
    IFS= read -r first_line < "$checkpoint" || true

    if [[ "$first_line" == "version https://git-lfs.github.com/spec/v1" ]]; then
        echo "Checkpoint is a Git LFS pointer, not a PyTorch model:" >&2
        echo "$checkpoint" >&2
        echo "Run 'git lfs pull' or train the model before evaluation." >&2
        return 1
    fi
}


validate_dataset() {
    if ((DRY_RUN == 1)); then
        return
    fi

    local data_root="$PROJECT_ROOT/datasets/raw/Classification"
    local required_paths=(
        "$data_root/classes.txt"
        "$data_root/ip102_v1.1/images"
        "$data_root/ip102_v1.1/train.txt"
        "$data_root/ip102_v1.1/val.txt"
        "$data_root/ip102_v1.1/test.txt"
    )
    local path

    for path in "${required_paths[@]}"; do
        if [[ ! -e "$path" ]]; then
            echo "Required dataset path is missing:" >&2
            echo "$path" >&2
            return 1
        fi
    done
}


validate_evaluation_outputs() {
    local output_dir=$1
    local split=$2

    require_file \
        "$output_dir/${split}_summary.txt" \
        "$split evaluation summary"
    require_file \
        "$output_dir/${split}_classification_report.csv" \
        "$split classification report"
    require_file \
        "$output_dir/${split}_predictions.csv" \
        "$split predictions"
    require_file \
        "$output_dir/${split}_probabilities.npz" \
        "$split probability archive"
    require_file \
        "$output_dir/${split}_confusion_matrix.csv" \
        "$split confusion matrix"
}


train_model() {
    local model_name=$1
    local training_script=$2

    if ((SKIP_TRAINING == 1)); then
        echo
        echo "[skip] $model_name training"
        return
    fi

    run_step \
        "Train $model_name" \
        "$PYTHON_BIN" "$training_script"
}


evaluate_model() {
    local model_name=$1
    local evaluator=$2
    local checkpoint=$3
    local output_dir=$4
    local split

    require_checkpoint "$checkpoint"

    for split in train val test; do
        run_step \
            "Evaluate $model_name on $split" \
            "$PYTHON_BIN" "$evaluator" "$checkpoint" \
            --split "$split"

        validate_evaluation_outputs "$output_dir" "$split"
    done
}


plot_model() {
    local model_name=$1
    local output_dir=$2
    local split

    require_file \
        "$output_dir/training_history.csv" \
        "$model_name training history"

    run_step \
        "Plot $model_name training curves" \
        "$PYTHON_BIN" \
        src/1_Classifier/4_PlotEvaluation/plot_training_curves.py \
        "$output_dir" \
        --model-name "$model_name"

    for split in train val test; do
        run_step \
            "Plot $model_name $split per-class metrics" \
            "$PYTHON_BIN" \
            src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
            "$output_dir" \
            --model-name "$model_name" \
            --split "$split"

        run_step \
            "Plot $model_name $split confusion matrices" \
            "$PYTHON_BIN" \
            src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
            "$output_dir" \
            --model-name "$model_name" \
            --split "$split"
    done
}


analyse_confusion() {
    local model_name=$1
    local output_dir=$2

    run_step \
        "Analyse $model_name confusion structure" \
        "$PYTHON_BIN" \
        src/1_Classifier/5_ClassSeparability/1_analyze_confusion_matrix.py \
        "$output_dir" \
        --split all \
        --min-pair-count 2
}


evaluate_coarse_labels() {
    local model_name=$1
    local output_dir=$2

    run_step \
        "Evaluate $model_name with the frozen coarse mapping" \
        "$PYTHON_BIN" \
        src/1_Classifier/5_ClassSeparability/4_evaluate_coarse.py \
        "$output_dir" \
        --split test
}


echo "COMP9444 IP102 complete experiment pipeline"
echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Log: $LOG_FILE"
echo "Skip training: $SKIP_TRAINING"
echo "Skip download: $SKIP_DOWNLOAD"
echo "Skip separability: $SKIP_SEPARABILITY"
echo "Dry run: $DRY_RUN"

if ((SKIP_TRAINING == 0)); then
    echo
    echo "Warning: training is enabled and may overwrite existing checkpoints"
    echo "and training histories. The complete run can take many hours."
fi


run_step \
    "Check required Python packages" \
    "$PYTHON_BIN" -c \
    "import cv2, gdown, imblearn, matplotlib, networkx, numpy, pandas, sklearn, timm, torch, torchvision"

if ((SKIP_DOWNLOAD == 0)); then
    run_step \
        "Download and extract IP102" \
        "$PYTHON_BIN" src/0_Data_Prep/1_download_data.py
else
    echo
    echo "[skip] dataset download"
fi

CURRENT_STEP="Validate IP102 dataset"
validate_dataset

run_step \
    "Generate IP102 class-distribution analysis" \
    "$PYTHON_BIN" src/0_Data_Prep/2_class_distribution.py


BASELINE_OUTPUT="$PROJECT_ROOT/outputs/classifier/resnet50"
BASELINE_CHECKPOINT="$BASELINE_OUTPUT/base_resnet50_best_model.pth"

train_model \
    "Baseline ResNet50" \
    src/1_Classifier/1_ResNet50/base_train_resnet.py
evaluate_model \
    "Baseline ResNet50" \
    src/1_Classifier/1_ResNet50/evaluate_resnet.py \
    "$BASELINE_CHECKPOINT" \
    "$BASELINE_OUTPUT"
plot_model "Baseline ResNet50" "$BASELINE_OUTPUT"

if ((SKIP_SEPARABILITY == 0)); then
    analyse_confusion "Baseline ResNet50" "$BASELINE_OUTPUT"

    run_step \
        "Freeze validation-derived difficulty groups" \
        "$PYTHON_BIN" \
        src/1_Classifier/5_ClassSeparability/2_evaluate_difficulty_groups.py \
        "$BASELINE_OUTPUT"

    run_step \
        "Prepare the frozen class-separability mapping" \
        "$PYTHON_BIN" \
        src/1_Classifier/5_ClassSeparability/3_prepare_separability_data.py

    evaluate_coarse_labels "Baseline ResNet50" "$BASELINE_OUTPUT"
fi


IMBALANCE_OUTPUT="$PROJECT_ROOT/outputs/classifier/resnet50_imbalance"
IMBALANCE_CHECKPOINT="$IMBALANCE_OUTPUT/resnet50_imbalance_best_model.pth"

train_model \
    "Imbalance-Aware ResNet50" \
    src/1_Classifier/1_ResNet50/imbalance_train_resnet.py
evaluate_model \
    "Imbalance-Aware ResNet50" \
    src/1_Classifier/1_ResNet50/evaluate_resnet.py \
    "$IMBALANCE_CHECKPOINT" \
    "$IMBALANCE_OUTPUT"
plot_model "Imbalance-Aware ResNet50" "$IMBALANCE_OUTPUT"

if ((SKIP_SEPARABILITY == 0)); then
    analyse_confusion "Imbalance-Aware ResNet50" "$IMBALANCE_OUTPUT"
    evaluate_coarse_labels \
        "Imbalance-Aware ResNet50" \
        "$IMBALANCE_OUTPUT"
fi


EFFICIENTNET_OUTPUT="$PROJECT_ROOT/outputs/classifier/efficientnet_b3"
EFFICIENTNET_CHECKPOINT="$EFFICIENTNET_OUTPUT/efficientnet_b3_best_model.pth"

train_model \
    "EfficientNet-B3" \
    src/1_Classifier/3_EfficientNet/base_train_effnet.py
evaluate_model \
    "EfficientNet-B3" \
    src/1_Classifier/3_EfficientNet/evaluate_efficientnet.py \
    "$EFFICIENTNET_CHECKPOINT" \
    "$EFFICIENTNET_OUTPUT"
plot_model "EfficientNet-B3" "$EFFICIENTNET_OUTPUT"

if ((SKIP_SEPARABILITY == 0)); then
    analyse_confusion "EfficientNet-B3" "$EFFICIENTNET_OUTPUT"
    evaluate_coarse_labels "EfficientNet-B3" "$EFFICIENTNET_OUTPUT"
fi


DEIT_OUTPUT="$PROJECT_ROOT/outputs/classifier/deit3_base_patch16_224"
DEIT_CHECKPOINT="$DEIT_OUTPUT/deit3_base_patch16_224_best_model.pth"

train_model \
    "DeiT-III Base/16" \
    src/1_Classifier/2_ViT/base_train_vit.py
evaluate_model \
    "DeiT-III Base/16" \
    src/1_Classifier/2_ViT/evaluate_vit.py \
    "$DEIT_CHECKPOINT" \
    "$DEIT_OUTPUT"
plot_model "DeiT-III Base/16" "$DEIT_OUTPUT"

if ((SKIP_SEPARABILITY == 0)); then
    analyse_confusion "DeiT-III Base/16" "$DEIT_OUTPUT"
    evaluate_coarse_labels "DeiT-III Base/16" "$DEIT_OUTPUT"

    run_step \
        "Compare cross-model within-cluster separability" \
        "$PYTHON_BIN" \
        src/1_Classifier/5_ClassSeparability/5_evaluate_oracle_clusters.py \
        --model \
        "baseline_resnet50=$BASELINE_OUTPUT/test_probabilities.npz" \
        --model \
        "imbalance_resnet50=$IMBALANCE_OUTPUT/test_probabilities.npz" \
        --model \
        "efficientnet_b3=$EFFICIENTNET_OUTPUT/test_probabilities.npz" \
        --model \
        "deit3_base16=$DEIT_OUTPUT/test_probabilities.npz" \
        --split test
fi


CURRENT_STEP="complete"
echo
echo "Pipeline completed successfully."
echo "Log: $LOG_FILE"
