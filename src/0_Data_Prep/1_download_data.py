from pathlib import Path
import gdown
import tarfile

# Find project_rootpath--->COMP9444_Group
# COMP9444_Group<---source<----0_Data_Prep
project_root = Path(__file__).resolve().parents[2]

# Dataset URL
url = "https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo"

# download to COMP9444_Group/datasets/raw
output_dir = project_root / "datasets" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)


classification_tar = output_dir / "Classification" / "ip102_v1.1.tar"
classification_dir = output_dir / "Classification" / "ip102_v1.1"

if not classification_tar.exists() and not classification_dir.exists():
    # Download Dataset by gdown
    gdown.download_folder(
        url=url,
        output=str(output_dir),
        quiet=False,
    )
else:
    print("Dataset already downloaded, skipping download")

# Unzip all tar files
# Find all tar_files first
tar_files = list(output_dir.rglob("*.tar"))
print("Found", len(tar_files), "tar files")

for tar_path in tar_files:
    print("Extracting:", tar_path)
    # unzipping 
    with tarfile.open(tar_path, "r") as tar_file:
        tar_file.extractall(path=tar_path.parent)

    # If successfully unzip, delete original tar files
    tar_path.unlink()
    print("Deleted:", tar_path)

print("All tar files have been extracted and deleted")