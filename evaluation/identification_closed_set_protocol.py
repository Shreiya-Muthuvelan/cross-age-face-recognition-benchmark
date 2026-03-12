
# Closed-set identification evaluation for FGNET and CACD datasets.

# Usage (run from project root):
#   python evaluation/identification_closed_set_protocol.py --dataset fgnet --model vggface
#   python evaluation/identification_closed_set_protocol.py --dataset cacd  --model facenet

# Supported models: facenet, arcface, vggface, openface
# Results are saved to results/results_identification_<dataset>_<model>.csv

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import re
import numpy as np
import pandas as pd

from src.identification import identification_eval


def load_all_embeddings(folder, dataset):
    """
    Loads per-image .npy embedding files from data/embeddings/<model>/<dataset>/
    and returns stacked embeddings with their subject IDs.

    """
    embeddings = []
    subjects   = []

    for file in os.listdir(folder):
        if not file.endswith(".npy"):
            continue
        # Skip the combined batch file produced by extract_embeddings_and_save
        if file in (f"{dataset}_embeddings.npy",):
            continue

        file_path = os.path.join(folder, file)
        emb = np.load(file_path)

        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        elif emb.ndim != 2:
            print(f"Skipping {file}, unexpected shape: {emb.shape}")
            continue

        if dataset.lower() == "cacd":
            # e.g. "14_Aaron_Johnson_0001.npy" -> "Aaron_Johnson"
            parts = file.replace(".npy", "").split("_")
            if len(parts) >= 3:
                subj_id = "_".join(parts[1:-1])
            else:
                print(f"Skipping file {file}, unexpected name pattern")
                continue
            subj_ids = [subj_id] * emb.shape[0]

        elif dataset.lower() == "fgnet":
            # e.g. "001A05.npy" -> "001"
            match = re.match(r"(\d{3})", file)
            if match:
                subj_id  = match.group(1)
                subj_ids = [subj_id] * emb.shape[0]
            else:
                subj_ids = [f"subj_{i+1}" for i in range(emb.shape[0])]
                print(f"Warning: {file} does not match pattern, using dummy subjects")

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        embeddings.append(emb)
        subjects.extend(subj_ids)

    if not embeddings:
        raise RuntimeError(
            f"No embeddings found in {folder}. "
            "Run scripts/run_embeddings.py first."
        )

    embeddings = np.vstack(embeddings)
    subjects   = np.array(subjects)

    assert embeddings.shape[0] == len(subjects), \
        "Mismatch between number of embeddings and subject labels."

    print(f"Embeddings shape : {embeddings.shape}")
    print(f"Unique subjects  : {len(np.unique(subjects))}")
    return embeddings, subjects


def main(dataset, model):
    print(f"\n[Closed-Set Identification] dataset={dataset}  model={model}")

    folder = os.path.join("data", "embeddings", model, dataset)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Embeddings folder not found: {folder}\n"
            "Run scripts/run_embeddings.py first."
        )

    embeddings, subjects = load_all_embeddings(folder, dataset)
    print(f"Loaded {embeddings.shape[0]} embeddings  dim={embeddings.shape[1]}")

    # Use the full set as both gallery and probe (closed-set protocol)
    results, ranks = identification_eval(
        gallery_embeddings=embeddings,
        gallery_subjects=subjects,
        probe_embeddings=embeddings,
        probe_subjects=subjects,
        K_list=[1, 5, 10],
    )

    print("\nIdentification Results:")
    for k, v in results.items():
        print(f"  {k} Accuracy: {v:.3f}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"results_identification_{dataset}_{model}.csv")
    rows = [{"metric": k, "accuracy": v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Closed-set identification evaluation"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["fgnet", "cacd"],
                        help="Dataset to evaluate on")
    parser.add_argument("--model", type=str, required=True,
                        choices=["facenet", "arcface", "vggface", "openface"],
                        help="Face recognition model")
    args = parser.parse_args()
    main(args.dataset, args.model)
