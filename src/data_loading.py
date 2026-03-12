# Utility module for loading processed dataset metadata into a DataFrame.
# Run from project root: python src/data_loading.py
# Or import load_metadata() / assign_groups() from other scripts.

import json
import os
import pandas as pd


def load_metadata(dataset_root):
    """
    Walks data/processed/CACD and data/processed/FGNET, reads each subject's
    metadata.json, and returns a flat DataFrame with one row per image.
    """
    data = []
    for ds in ['CACD', 'FGNET']:
        base = os.path.join(dataset_root, ds)
        if not os.path.isdir(base):
            print(f"Warning: {base} does not exist, skipping.")
            continue
        for subj in os.listdir(base):
            subj_dir = os.path.join(base, subj)
            meta_path = os.path.join(subj_dir, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                for fname, info in meta.items():
                    full_path = os.path.join(subj_dir, fname)
                    data.append({
                        'dataset':   ds.lower(),
                        'subject':   subj,
                        'file_path': full_path,
                        'age':       info['age'],
                        'filepath':  full_path,    # kept for backward compatibility
                    })
    return pd.DataFrame(data)


def assign_groups(age):
    """Maps a numeric age to a broad age-group label."""
    if age <= 12:
        return 'child'
    elif age <= 19:
        return 'teen'
    elif age <= 35:
        return 'young_adult'
    elif age <= 55:
        return 'adult'
    else:
        return 'senior'


if __name__ == "__main__":
    processed_root = os.path.join("data", "processed")
    df = load_metadata(processed_root)
    df['age_group'] = df['age'].apply(assign_groups)
    print(df.head())
    print(f"\nTotal images loaded: {len(df)}")
    print(df['dataset'].value_counts())
