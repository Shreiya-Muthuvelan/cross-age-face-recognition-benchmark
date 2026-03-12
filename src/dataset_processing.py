
# Organizes raw FGNET and CACD images into per-subject folders and writes metadata.json per subject.
# Run from project root after src/dataset.py has been executed.

import os
import shutil
import json
from tqdm import tqdm

fgnet_raw  = os.path.join("data", "raw", "fgnet")
cacd_raw   = os.path.join("data", "raw", "cacd")
output_dir = os.path.join("data", "processed")

os.makedirs(output_dir, exist_ok=True)


def process_fgnet():
    """
    Groups FGNET images by subject (first 3 chars of filename) into
    data/processed/FGNET/<subject_id>/ and writes a metadata.json per subject.

    FGNET filename convention: <subject_id><gender><age_with_leading_zero>.jpg
    e.g. 001A05.jpg -> subject=001, age=5
    """
    fgnet_out = os.path.join(output_dir, "FGNET")
    os.makedirs(fgnet_out, exist_ok=True)

    subjects = {}

    for file in os.listdir(fgnet_raw):
        if file.lower().endswith(".jpg"):
            subject_id = file[:3]
            age = int(file[4:6])
            subject_folder = os.path.join(fgnet_out, subject_id)
            os.makedirs(subject_folder, exist_ok=True)

            src = os.path.join(fgnet_raw, file)
            dst = os.path.join(subject_folder, file)
            shutil.copy(src, dst)

            if subject_id not in subjects:
                subjects[subject_id] = {}
            subjects[subject_id][file] = {"age": age}

    # Write metadata.json once per subject
    for subj_id, meta in subjects.items():
        meta_path = os.path.join(fgnet_out, subj_id, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    print("FGNET dataset processed and organized under data/processed/FGNET/")


def process_cacd():
    """
    Copies CACD images (already organised by subject folder) into
    data/processed/CACD/<subject>/ and writes a metadata.json per subject.

    CACD filename convention: <age>_<FirstName>_<LastName>_<index>.jpg
    e.g. 32_Aaron_Eckhart_0001.jpg -> age=32
    """
    cacd_out = os.path.join(output_dir, "CACD")
    os.makedirs(cacd_out, exist_ok=True)

    for subj in tqdm(os.listdir(cacd_raw), desc="Processing CACD subjects"):
        subj_path = os.path.join(cacd_raw, subj)
        if not os.path.isdir(subj_path):
            continue

        subj_folder = os.path.join(cacd_out, subj)
        os.makedirs(subj_folder, exist_ok=True)

        meta = {}
        for file in os.listdir(subj_path):
            if file.endswith(".jpg"):
                age = int(file.split("_")[0])
                shutil.copy(os.path.join(subj_path, file), os.path.join(subj_folder, file))
                meta[file] = {"age": age}

        with open(os.path.join(subj_folder, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)

    print("CACD dataset processed and organized under data/processed/CACD/")


if __name__ == "__main__":
    process_fgnet()
    process_cacd()
