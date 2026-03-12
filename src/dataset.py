# Downloads FGNET and CACD datasets from Kaggle and places them under data/raw/

import kagglehub
import shutil
import os

# Run from project root
fgnet_path = kagglehub.dataset_download("aiolapo/fgnet-dataset")
print("Path to dataset files:", fgnet_path)

cacd_path = kagglehub.dataset_download("pdombrza/cacd-filtered-dataset")
print("Path to dataset files:", cacd_path)

# Creating necessary directories
os.makedirs("data/raw/fgnet", exist_ok=True)
os.makedirs("data/raw/cacd", exist_ok=True)

# Copying to project data directories
shutil.copytree(fgnet_path, "data/raw/fgnet", dirs_exist_ok=True)
shutil.copytree(cacd_path, "data/raw/cacd", dirs_exist_ok=True)

# Flatten FGNET folder structure: data/raw/fgnet/FGNET/images -> data/raw/fgnet/
fgnet_images_src = os.path.join("data", "raw", "fgnet", "FGNET", "images")
fgnet_images_dst = os.path.join("data", "raw", "fgnet", "images")

shutil.move(fgnet_images_src, fgnet_images_dst)
shutil.rmtree(os.path.join("data", "raw", "fgnet", "FGNET"))
shutil.move(fgnet_images_dst, os.path.join("data", "raw"))
shutil.rmtree(os.path.join("data", "raw", "fgnet"))
os.rename(os.path.join("data", "raw", "images"), os.path.join("data", "raw", "fgnet"))

print("Datasets downloaded and organized under data/raw/")
