# Since sourcefile folder is ready
# PLEASE GO TO YOUR PROJECT ROOT FILEPATH FIRST
## ex: COMP9444_Group(root)/source
## Every command is executed under COMP9444_Group(root)

# Data preparation

## 1: Install python libraries + Download Dataset
# (1) Execute 0_install_libraries.sh to install libraries
source/0_Data_Prep/0_install_libraries.sh

# (2) Create & LOAD a virtual enviroment for this project
source env/bin/activate
pip list

# (3) Download & Unzip dateset from google drive
python3 source/0_Data_Prep/1_download_data.py
