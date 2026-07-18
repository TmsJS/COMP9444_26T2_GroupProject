#!/bin/bash
# Stop immediately if any command fails.
set -e
echo "Setting up python environment for this project:"

# 0. Basic system update
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv unzip

# Optional: Create a venv so everything stays clean
python3 -m venv env
source env/bin/activate

echo "Python version:"
python3 --version

echo "Installing core DeepLearning libraries:"


# 0. Stable versions of numpy & opencv
pip install numpy==2.2.6
pip install opencv-python==5.0.0.93
pip install scikit-learn
pip install pandas
pip install seaborn

# 1. Install PyTorch + CUDA
#    Torch 2.5.1 + CUDA 11.8
# pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch==2.5.1+cu118 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

echo "Torch installed:"
python3 - <<EOF
import torch
print('Torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
EOF

# 2. YOLO (Official ultralytics)
pip install ultralytics

# 3. Coco evaluation tools
pip install pycocotools
pip install faster-coco-eval

# 4. EfficientDet + TIMM (EfficientNet backbone dependency)
pip install timm==0.9.16
pip install effdet==0.4.1

# 5. Metrics tools
pip install torchmetrics
pip install torchmetrics[detection]
pip install imbalanced-learn

# 6. Kaggle or googledrive dataset download 
pip install kagglehub
pip install gdown

# 7. Data Augmentation
pip install albumentations==1.4.16


echo "All libraries installed successfully!"
echo "To activate venv(virtual env) later: source env/bin/activate"
