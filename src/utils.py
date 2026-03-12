
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from collections import defaultdict
import math
import random
import os


def load_metadata_csv(meta_csv):
    df = pd.read_csv(meta_csv)
    return df


def load_embeddings(df, path_column):
    embeddings = []
    for rel_path in df[path_column]:
        abs_path = os.path.abspath(rel_path)
        emb = np.load(abs_path)
        embeddings.append(emb)
    return np.array(embeddings)


def cosine_sim(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.dot(b_norm, a_norm) if b_norm.ndim == 2 else np.dot(a_norm, b_norm)


def pairwise_cosine_matrix(X, Y):
    Xn = X / np.linalg.norm(X, axis=-1, keepdims=True)
    Yn = Y / np.linalg.norm(Y, axis=-1, keepdims=True)
    return Xn.dot(Yn.T)


def compute_roc_aur_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_threshold = interp1d(fpr - fnr, thr)(0.0)
    eer_fpr = interp1d(thr, fpr)(eer_threshold)
    eer = eer_fpr
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thr': thr,
        'auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
    }


def tar_at_far(labels, scores, target_far=1e-3):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    idx = np.where(fpr <= target_far)[0]
    if len(idx) == 0:
        return 0.0
    return tpr[idx[-1]]
