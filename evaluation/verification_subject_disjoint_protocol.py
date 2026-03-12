
# Subject-disjoint verification evaluation across age bins.

# Usage (run from project root):
#   python evaluation/verification_subject_disjoint_protocol.py --dataset fgnet --model vggface
#   python evaluation/verification_subject_disjoint_protocol.py --dataset cacd  --model facenet
#
# Supported models: facenet, arcface, vggface, openface
# Results are saved to results/results_agebins_<dataset>_<model>.csv

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd

from src.utils import load_metadata_csv, load_embeddings
from src.age_group_pairing_subject_disjoint import (
    create_age_bins,
    age_gap_pairing,
    evaluate_by_age_bins,
)


def main(dataset, model):
    print(f"\n[Subject-Disjoint Verification] dataset={dataset}  model={model}")

    meta_path = "metadata.csv"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"metadata.csv not found at project root.\n"
            "Generate it by running src/data_loading.py first."
        )

    df = load_metadata_csv(meta_path)
    print(f"Metadata loaded: {df.shape[0]} rows")

    df_model = df[(df['dataset'] == dataset) & (df['model'] == model)].reset_index(drop=True)
    print(f"Filtered rows for dataset={dataset}, model={model}: {len(df_model)}")

    embeddings = load_embeddings(df_model, 'embedding_path')
    print(f"Embeddings loaded: {embeddings.shape}")

    df_model = create_age_bins(df_model, age_col='age')
    print("Age bin distribution:\n", df_model['age_bin'].value_counts().to_string())

    by_bin = age_gap_pairing(df_model, embeddings)

    results = evaluate_by_age_bins(by_bin)

    print(f"\nResults for dataset={dataset}, model={model}:")
    for k, v in results.items():
        if v is None:
            print(f"  {k}: no pairs available")
        else:
            print(f"  {k}: AUC={v['auc']:.3f}  EER={v['eer']:.3f}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"results_agebins_{dataset}_{model}.csv")
    rows = []
    for k, v in results.items():
        if v is not None:
            rows.append({'bin_pair': str(k), 'auc': v['auc'], 'eer': v['eer']})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subject-disjoint age-bin verification evaluation"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["fgnet", "cacd"],
                        help="Dataset to evaluate on")
    parser.add_argument("--model", type=str, required=True,
                        choices=["facenet", "arcface", "vggface", "openface"],
                        help="Face recognition model")
    args = parser.parse_args()
    main(args.dataset, args.model)
