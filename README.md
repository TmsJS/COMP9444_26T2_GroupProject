### Since sourcefile folder is ready
### PLEASE GO TO YOUR PROJECT ROOT FILEPATH FIRST
#### ex: COMP9444_Group(root)/source
#### Each command below is executed under COMP9444_Group(root)

### Data preparation

#### 1: Install python libraries + Download Dataset
##### (1) Execute 0_install_libraries.sh to install libraries
src/0_Data_Prep/0_install_libraries.sh

##### (2) Create & LOAD a virtual enviroment for this project
source env/bin/activate
pip list

##### (3) Download & Unzip dateset from google drive
python3 src/0_Data_Prep/1_download_data.py


### Classifier 
#### 1: Resnet50
##### (1) Train Resnet50 & Imbalance Resnet50
python3 src/1_Classifier/1_ResNet50/train_resnet.py
python3 src/1_Classifier/1_ResNet50/imbalance_train_resnet.py

##### (2) Evaluate Baseline Resnet50 & Imbalance Resnet50
python3 src/1_Classifier/1_ResNet50/evaluate_resnet.py \
  outputs/classifier/resnet50/resnet50_best_model.pth

python3 src/1_Classifier/1_ResNet50/evaluate_resnet.py \
  outputs/classifier/resnet50_imbalance/resnet50_imbalance_best_model.pth


##### (3) Plot Resnet50 & Imbalance Resnet50 evaluation result
python3 src/1_Classifier/1_ResNet50/plot_class_metrics.py \
  outputs/classifier/resnet50
python3 src/1_Classifier/1_ResNet50/plot_confusion_matrix.py \
  outputs/classifier/resnet50
python3 src/1_Classifier/1_ResNet50/plot_training_curve.py \
  outputs/classifier/resnet50

python3 src/1_Classifier/1_ResNet50/plot_class_metrics.py \
  outputs/classifier/resnet50_imbalance
python3 src/1_Classifier/1_ResNet50/plot_confusion_matrix.py \
  outputs/classifier/resnet50_imbalance
python3 src/1_Classifier/1_ResNet50/plot_training_curve.py \
  outputs/classifier/resnet50_imbalance