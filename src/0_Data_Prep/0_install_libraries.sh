#!/usr/bin/env bash

set -Eeuo pipefail


SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"


case "$OS_NAME" in
    Linux)
        PLATFORM_NAME="Ubuntu/WSL"
        DEFAULT_VENV_NAME="env"
        ;;
    Darwin)
        PLATFORM_NAME="macOS"
        DEFAULT_VENV_NAME=".venv"
        ;;
    *)
        echo "Unsupported operating system: $OS_NAME" >&2
        echo "This installer supports Ubuntu/WSL and macOS." >&2
        exit 1
        ;;
esac


# Override with VENV_DIR=/path/to/venv when a different location is required.
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/$DEFAULT_VENV_NAME}"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"

echo "Setting up the COMP9444 project environment"
echo "Platform: $PLATFORM_NAME ($ARCH_NAME)"
echo "Project root: $PROJECT_ROOT"
echo "Virtual environment: $VENV_DIR"


if [[ "$OS_NAME" == "Linux" ]]; then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "apt-get was not found. The Linux branch supports Ubuntu/WSL only." >&2
        exit 1
    fi

    APT_PREFIX=()
    if ((EUID != 0)); then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "sudo is required to install Ubuntu system packages." >&2
            exit 1
        fi
        APT_PREFIX=(sudo)
    fi

    "${APT_PREFIX[@]}" apt-get update
    "${APT_PREFIX[@]}" apt-get install -y \
        python3 python3-pip python3-venv unzip
else
    if ! command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
        echo "$PYTHON_COMMAND was not found." >&2
        echo "Install Python 3.10-3.12, then run this installer again." >&2
        exit 1
    fi

    if ! xcode-select -p >/dev/null 2>&1; then
        echo "Xcode Command Line Tools are required by some Python packages." >&2
        echo "Install them with 'xcode-select --install', then run this script again." >&2
        exit 1
    fi
fi


if [[ -e "$VENV_DIR" && ! -x "$VENV_DIR/bin/python" ]]; then
    echo "The environment path exists but does not contain a usable Python:" >&2
    echo "$VENV_DIR" >&2
    echo "Move it aside or set VENV_DIR to a different path." >&2
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtual environment..."
    "$PYTHON_COMMAND" -m venv "$VENV_DIR"
else
    echo "Reusing existing virtual environment."
fi

VENV_PYTHON="$VENV_DIR/bin/python"

echo "Python version:"
"$VENV_PYTHON" --version

echo "Updating pip build tools..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo "Installing numerical and plotting libraries..."
"$VENV_PYTHON" -m pip install \
    numpy==2.2.6 \
    opencv-python==5.0.0.93 \
    scikit-learn \
    imbalanced-learn \
    networkx \
    pandas \
    matplotlib \
    seaborn \
    pillow \
    tqdm

echo "Installing PyTorch..."
if [[ "$OS_NAME" == "Linux" ]]; then
    # CUDA 11.8 wheels for Ubuntu/WSL with a compatible NVIDIA driver.
    "$VENV_PYTHON" -m pip install \
        torch==2.5.1+cu118 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu118
else
    # PyPI supplies the native macOS wheel. PyTorch uses MPS when available.
    "$VENV_PYTHON" -m pip install \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1
fi

echo "Installing project libraries..."
"$VENV_PYTHON" -m pip install \
    ultralytics \
    pycocotools \
    faster-coco-eval \
    timm==0.9.16 \
    effdet==0.4.1 \
    'torchmetrics[detection]' \
    kagglehub \
    gdown \
    albumentations==1.4.16


echo "Verifying the installation..."
"$VENV_PYTHON" - <<'PY'
import platform

import cv2
import numpy
import pandas
import sklearn
import timm
import torch
import torchvision

cuda_available = torch.cuda.is_available()
mps_available = (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
)

if cuda_available:
    device = torch.cuda.get_device_name(0)
elif mps_available:
    device = "Apple Silicon GPU (MPS)"
else:
    device = "CPU"

print(f"Python: {platform.python_version()}")
print(f"NumPy: {numpy.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"TIMM: {timm.__version__}")
print(f"CUDA available: {cuda_available}")
print(f"MPS available: {mps_available}")
print(f"Device: {device}")
PY

echo
echo "All libraries installed successfully."
echo "Activate this environment from the project root with:"
printf 'source %q\n' "$VENV_DIR/bin/activate"
