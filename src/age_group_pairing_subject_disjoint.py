
import numpy as np
import pandas as pd
import random
from src.utils import cosine_sim, compute_roc_aur_eer


def create_age_bins(df, age_col='age'):
    df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
    bins   = [0,  12,  18,  30,  45,  60, 200]
    labels = ["0-11", "12-17", "18-29", "30-44", "45-59", "60+"]
    df['age_bin'] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
    return df


def age_gap_pairing(df, embeddings, subject_col='subject_id', age_col='age', bin_col='age_bin',
                    max_pairs_per_bin=20000, mode='closed', test_frac=0.2, seed=42):
    """
    Creates positive/negative pairs for age-bin verification.
    """
    np.random.seed(seed)
    random.seed(seed)

    all_subjects = df[subject_col].unique()

    if mode == 'subject_disjoint':
        np.random.shuffle(all_subjects)
        split_idx = int(len(all_subjects) * (1 - test_frac))
        eval_subjects = all_subjects[split_idx:]
        df = df[df[subject_col].isin(eval_subjects)].reset_index(drop=True)

    by_bin = {}
    bins = df[bin_col].dropna().unique().tolist()
    indices_by_bin = {b: df[df[bin_col] == b].index.values for b in bins}

    for bx in bins:
        for by in bins:
            ix = indices_by_bin[bx]
            iy = indices_by_bin[by]

            scores = []
            labels = []

            # ---- Positive pairs ----
            if bx == by:
                # Same age-bin: pair images of the same subject within the bin
                subs = df.loc[ix, subject_col].unique()
                for s in subs:
                    idxs = df[(df[bin_col] == bx) & (df[subject_col] == s)].index.values
                    if len(idxs) < 2:
                        continue
                    comb = [(idxs[i], idxs[j])
                            for i in range(len(idxs))
                            for j in range(i + 1, len(idxs))]
                    random.shuffle(comb)
                    comb = comb[:1000]   # cap per subject
                    for a, b in comb:
                        scores.append(cosine_sim(embeddings[a], embeddings[b]))
                        labels.append(1)
            else:
                # Cross age-bin: pair images of the same subject across the two bins
                subs = set(df.loc[ix, subject_col].unique()).intersection(
                    set(df.loc[iy, subject_col].unique()))
                for s in subs:
                    a_indices = df[(df[subject_col] == s) & (df[bin_col] == bx)].index.values
                    b_indices = df[(df[subject_col] == s) & (df[bin_col] == by)].index.values
                    for a in a_indices:
                        for b in b_indices:
                            scores.append(cosine_sim(embeddings[a], embeddings[b]))
                            labels.append(1)

            # ---- Negative pairs ----
            num_neg = min(len(ix) * 2, max_pairs_per_bin)
            for _ in range(num_neg):
                a = np.random.choice(ix)
                b = np.random.choice(iy)
                if df.loc[a, subject_col] == df.loc[b, subject_col]:
                    continue
                scores.append(cosine_sim(embeddings[a], embeddings[b]))
                labels.append(0)

            by_bin[(bx, by)] = {
                'scores': np.array(scores),
                'labels': np.array(labels),
            }

    return by_bin


def evaluate_by_age_bins(by_bin_dict):
    """
    Computes AUC and EER for each age-bin pair.
    """
    results = {}
    for k, v in by_bin_dict.items():
        labels = v['labels']
        scores = v['scores']
        if len(labels) == 0:
            results[k] = None
            continue
        m = compute_roc_aur_eer(labels, scores)
        results[k] = {'auc': m['auc'], 'eer': m['eer']}
    return results
