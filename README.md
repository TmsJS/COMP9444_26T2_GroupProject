# COMP9444 Bugcatchers — IP102 Classification

This repository contains a reproducible workflow for fine-grained insect pest
classification on the IP102 dataset. The main experiments compare:

1. Baseline ResNet50;
2. Imbalance-Aware ResNet50;
3. EfficientNet-B3;
4. DeiT-III Base/16.

All commands below must be executed from the project root.

## 1. Environment Setup

The installer automatically detects Ubuntu/WSL or macOS and creates the
appropriate virtual environment and PyTorch installation.

### Ubuntu/WSL with NVIDIA CUDA

On Ubuntu/WSL, the installer creates or reuses `env` and installs the CUDA
11.8 build of PyTorch 2.5.1:

```bash
./src/0_Data_Prep/0_install_libraries.sh
source env/bin/activate
```

### macOS with Apple Silicon

On macOS, the same installer creates or reuses `.venv` and installs the native
PyTorch 2.5.1 build. PyTorch will use the Apple Silicon GPU through MPS when it
is available. Xcode Command Line Tools and Python 3.10-3.12 must already be
installed.

```bash
./src/0_Data_Prep/0_install_libraries.sh
source .venv/bin/activate
```

If Xcode Command Line Tools are missing, install them first with:

```bash
xcode-select --install
```

To reuse a virtual environment at a different location, set `VENV_DIR` before
running the installer. For example:

```bash
VENV_DIR="$PWD/my_env" ./src/0_Data_Prep/0_install_libraries.sh
source my_env/bin/activate
```

Confirm that the intended environment is active before continuing:

```bash
python --version
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

## 2. Saved Checkpoints and Git LFS

Model checkpoints (`*.pth`) are managed by Git LFS. If you want to evaluate
the saved models without retraining them, download the actual checkpoint files
first:

```bash
git lfs install
git lfs pull
```

A checkpoint that is only about 130 bytes is a Git LFS pointer, not a loadable
PyTorch model. A real checkpoint should be tens or hundreds of megabytes.

This step is not required if you train each model from scratch, because the
training scripts generate new checkpoints locally.

## 3. Dataset Preparation

Download and extract IP102 under `datasets/raw`:

```bash
python src/0_Data_Prep/1_download_data.py
```

Generate the class-distribution analysis:

```bash
python src/0_Data_Prep/2_class_distribution.py
```

The expected classification dataset is:

```text
datasets/raw/Classification/
├── classes.txt
└── ip102_v1.1/
    ├── images/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## 4. Baseline ResNet50

### 4.1 Train

> Training creates or overwrites the best checkpoint and training history in
> `outputs/classifier/resnet50`.

```bash
python src/1_Classifier/1_ResNet50/base_train_resnet.py
```

### 4.2 Evaluate train, validation, and test splits

```bash
python src/1_Classifier/1_ResNet50/evaluate_resnet.py \
  outputs/classifier/resnet50/base_resnet50_best_model.pth \
  --split train

python src/1_Classifier/1_ResNet50/evaluate_resnet.py \
  outputs/classifier/resnet50/base_resnet50_best_model.pth \
  --split val

python src/1_Classifier/1_ResNet50/evaluate_resnet.py \
  outputs/classifier/resnet50/base_resnet50_best_model.pth \
  --split test
```

Each evaluation generates split-specific summaries, per-class reports,
predictions, probability arrays, and confusion matrices.

### 4.3 Plot results

```bash
python src/1_Classifier/4_PlotEvaluation/plot_training_curves.py \
  outputs/classifier/resnet50 \
  --model-name "ResNet50"

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/resnet50 \
  --model-name "ResNet50" \
  --split train

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/resnet50 \
  --model-name "ResNet50" \
  --split val

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/resnet50 \
  --model-name "ResNet50" \
  --split test

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/resnet50 \
  --model-name "ResNet50" \
  --split train

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/resnet50 \
  --model-name "ResNet50" \
  --split val

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/resnet50 \
  --model-name "ResNet50" \
  --split test
```

### 4.4 Analyse confusion and class separability

Run this section only after all three ResNet50 splits have been evaluated.

```bash
python src/1_Classifier/5_ClassSeparability/1_analyze_confusion_matrix.py \
  outputs/classifier/resnet50 \
  --split all \
  --min-pair-count 2

python src/1_Classifier/5_ClassSeparability/2_evaluate_difficulty_groups.py \
  outputs/classifier/resnet50

python src/1_Classifier/5_ClassSeparability/3_prepare_separability_data.py

python src/1_Classifier/5_ClassSeparability/4_evaluate_coarse.py \
  outputs/classifier/resnet50 \
  --split test
```

The confusion clusters and difficulty groups are discovered from validation
results before their final test-set evaluation.

## 5. Imbalance-Aware ResNet50

### 5.1 Train

The imbalance-aware model uses two-sided resampling, class-weighted loss, and
stronger online augmentation for minority classes.

> Training creates or overwrites files in
> `outputs/classifier/resnet50_imbalance`.

```bash
python src/1_Classifier/1_ResNet50/imbalance_train_resnet.py
```

### 5.2 Evaluate train, validation, and test splits

```bash
python src/1_Classifier/1_ResNet50/evaluate_resnet.py \
  outputs/classifier/resnet50_imbalance/resnet50_imbalance_best_model.pth \
  --split train

python src/1_Classifier/1_ResNet50/evaluate_resnet.py \
  outputs/classifier/resnet50_imbalance/resnet50_imbalance_best_model.pth \
  --split val

python src/1_Classifier/1_ResNet50/evaluate_resnet.py \
  outputs/classifier/resnet50_imbalance/resnet50_imbalance_best_model.pth \
  --split test
```

### 5.3 Plot results

```bash
python src/1_Classifier/4_PlotEvaluation/plot_training_curves.py \
  outputs/classifier/resnet50_imbalance \
  --model-name "Imbalance-Aware ResNet50"

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/resnet50_imbalance \
  --model-name "Imbalance-Aware ResNet50" \
  --split train

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/resnet50_imbalance \
  --model-name "Imbalance-Aware ResNet50" \
  --split val

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/resnet50_imbalance \
  --model-name "Imbalance-Aware ResNet50" \
  --split test

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/resnet50_imbalance \
  --model-name "Imbalance-Aware ResNet50" \
  --split train

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/resnet50_imbalance \
  --model-name "Imbalance-Aware ResNet50" \
  --split val

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/resnet50_imbalance \
  --model-name "Imbalance-Aware ResNet50" \
  --split test
```

### 5.4 Analyse confusion and frozen class separability

Run this command only after the train, validation, and test confusion matrices
have been generated:

```bash
python src/1_Classifier/5_ClassSeparability/1_analyze_confusion_matrix.py \
  outputs/classifier/resnet50_imbalance \
  --split all \
  --min-pair-count 2

python src/1_Classifier/5_ClassSeparability/4_evaluate_coarse.py \
  outputs/classifier/resnet50_imbalance \
  --split test
```

The coarse evaluation reuses the class mapping frozen from the Baseline
ResNet50 validation analysis in Section 4.4.

## 6. EfficientNet-B3

### 6.1 Train

> Training creates or overwrites files in
> `outputs/classifier/efficientnet_b3`.

```bash
python src/1_Classifier/3_EfficientNet/base_train_effnet.py
```

### 6.2 Evaluate train, validation, and test splits

```bash
python src/1_Classifier/3_EfficientNet/evaluate_efficientnet.py \
  outputs/classifier/efficientnet_b3/efficientnet_b3_best_model.pth \
  --split train

python src/1_Classifier/3_EfficientNet/evaluate_efficientnet.py \
  outputs/classifier/efficientnet_b3/efficientnet_b3_best_model.pth \
  --split val

python src/1_Classifier/3_EfficientNet/evaluate_efficientnet.py \
  outputs/classifier/efficientnet_b3/efficientnet_b3_best_model.pth \
  --split test
```

### 6.3 Plot results

```bash
python src/1_Classifier/4_PlotEvaluation/plot_training_curves.py \
  outputs/classifier/efficientnet_b3 \
  --model-name "EfficientNet-B3"

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/efficientnet_b3 \
  --model-name "EfficientNet-B3" \
  --split train

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/efficientnet_b3 \
  --model-name "EfficientNet-B3" \
  --split val

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/efficientnet_b3 \
  --model-name "EfficientNet-B3" \
  --split test

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/efficientnet_b3 \
  --model-name "EfficientNet-B3" \
  --split train

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/efficientnet_b3 \
  --model-name "EfficientNet-B3" \
  --split val

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/efficientnet_b3 \
  --model-name "EfficientNet-B3" \
  --split test
```

### 6.4 Analyse confusion and frozen class separability

```bash
python src/1_Classifier/5_ClassSeparability/1_analyze_confusion_matrix.py \
  outputs/classifier/efficientnet_b3 \
  --split all \
  --min-pair-count 2

python src/1_Classifier/5_ClassSeparability/4_evaluate_coarse.py \
  outputs/classifier/efficientnet_b3 \
  --split test
```

The coarse evaluation reuses the class mapping frozen from the Baseline
ResNet50 validation analysis in Section 4.4.

## 7. DeiT-III Base/16

### 7.1 Train

DeiT-III Base/16 is an ImageNet-1K-pretrained vision transformer implemented
through TIMM.

> Training creates or overwrites files in
> `outputs/classifier/deit3_base_patch16_224`.

```bash
python src/1_Classifier/2_ViT/base_train_vit.py
```

### 7.2 Evaluate train, validation, and test splits

```bash
python src/1_Classifier/2_ViT/evaluate_vit.py \
  outputs/classifier/deit3_base_patch16_224/deit3_base_patch16_224_best_model.pth \
  --split train

python src/1_Classifier/2_ViT/evaluate_vit.py \
  outputs/classifier/deit3_base_patch16_224/deit3_base_patch16_224_best_model.pth \
  --split val

python src/1_Classifier/2_ViT/evaluate_vit.py \
  outputs/classifier/deit3_base_patch16_224/deit3_base_patch16_224_best_model.pth \
  --split test
```

### 7.3 Plot results

```bash
python src/1_Classifier/4_PlotEvaluation/plot_training_curves.py \
  outputs/classifier/deit3_base_patch16_224 \
  --model-name "DeiT-III Base/16"

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/deit3_base_patch16_224 \
  --model-name "DeiT-III Base/16" \
  --split train

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/deit3_base_patch16_224 \
  --model-name "DeiT-III Base/16" \
  --split val

python src/1_Classifier/4_PlotEvaluation/plot_class_metrics.py \
  outputs/classifier/deit3_base_patch16_224 \
  --model-name "DeiT-III Base/16" \
  --split test

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/deit3_base_patch16_224 \
  --model-name "DeiT-III Base/16" \
  --split train

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/deit3_base_patch16_224 \
  --model-name "DeiT-III Base/16" \
  --split val

python src/1_Classifier/4_PlotEvaluation/plot_confusion_matrix.py \
  outputs/classifier/deit3_base_patch16_224 \
  --model-name "DeiT-III Base/16" \
  --split test
```

### 7.4 Analyse confusion and frozen class separability

```bash
python src/1_Classifier/5_ClassSeparability/1_analyze_confusion_matrix.py \
  outputs/classifier/deit3_base_patch16_224 \
  --split all \
  --min-pair-count 2

python src/1_Classifier/5_ClassSeparability/4_evaluate_coarse.py \
  outputs/classifier/deit3_base_patch16_224 \
  --split test
```

The coarse evaluation reuses the class mapping frozen from the Baseline
ResNet50 validation analysis in Section 4.4.

## 8. Cross-Model Class-Separability Comparison

After all four models have produced `test_probabilities.npz`, compare their
within-cluster separability and an equal-weight probability ensemble using the
same validation-derived class mapping:

```bash
python src/1_Classifier/5_ClassSeparability/5_evaluate_oracle_clusters.py \
  --model baseline_resnet50=outputs/classifier/resnet50/test_probabilities.npz \
  --model imbalance_resnet50=outputs/classifier/resnet50_imbalance/test_probabilities.npz \
  --model efficientnet_b3=outputs/classifier/efficientnet_b3/test_probabilities.npz \
  --model deit3_base16=outputs/classifier/deit3_base_patch16_224/test_probabilities.npz \
  --split test
```

## 9. Execution Notes

- Run training only when a new checkpoint is required. Training may overwrite
  the existing best-model checkpoint and `training_history.csv`.
- The first training run may download ImageNet-pretrained weights if they are
  not already cached.
- Evaluation and plotting scripts save their outputs inside the selected model
  output directory.
- For a fair final comparison, use validation results for model selection and
  reserve the test set for final reporting.

## 10. One-Command Runner

The executable `run_pipeline.sh` reproduces the complete workflow in the
dependency-safe order documented above. By default, it downloads the dataset,
trains all four models, evaluates every split, generates all plots, freezes the
validation-derived class mapping, evaluates coarse labels, and runs the final
cross-model oracle comparison.

Review the complete command sequence without running it:

```bash
./run_pipeline.sh --dry-run
```

Run the complete workflow, including training:

```bash
./run_pipeline.sh
```

Reuse an existing dataset and existing checkpoints:

```bash
./run_pipeline.sh --skip-download --skip-training
```

Run model training, evaluation, and plotting without the class-separability
stages:

```bash
./run_pipeline.sh --skip-separability
```

The runner automatically selects the active virtual environment,
`.venv/bin/python`, or `env/bin/python`. It stops on the first failed stage,
rejects Git LFS pointer files when existing checkpoints are requested, and
writes timestamped logs to `outputs/pipeline_logs`.
