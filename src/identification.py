import numpy as np

def batch_pairwise_cosine(probe_embeddings, gallery_embeddings, batch_size=1000):
    """
    Compute pairwise cosine similarity in batches to save memory.
    """
    sims_list = []
    for i in range(0, len(probe_embeddings), batch_size):
        batch = probe_embeddings[i:i + batch_size]
        sims_batch = batch.dot(gallery_embeddings.T)
        sims_list.append(sims_batch)
    return np.vstack(sims_list)


def identification_eval(gallery_embeddings, gallery_subjects,
                        probe_embeddings, probe_subjects,
                        K_list=[1, 5, 10], batch_size=1000):
    """
    Closed-set identification evaluation.
    """
    sims = batch_pairwise_cosine(probe_embeddings, gallery_embeddings, batch_size=batch_size)

    ranks = []
    for i in range(sims.shape[0]):
        sim_row = sims[i].copy()
        # Mask self-match when gallery == probe set
        if len(probe_embeddings) == len(gallery_embeddings):
            sim_row[i] = -np.inf

        sorted_idx = np.argsort(-sim_row)
        sorted_subjects = gallery_subjects[sorted_idx]

        rank_pos = np.where(sorted_subjects == probe_subjects[i])[0]
        rank = rank_pos[0] + 1 if len(rank_pos) > 0 else np.inf
        ranks.append(rank)

    ranks = np.array(ranks)
    results = {}
    for k in K_list:
        results[f'Rank-{k}'] = np.mean(ranks <= k)

    return results, ranks
