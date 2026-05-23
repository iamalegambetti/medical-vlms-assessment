"""
retrieval.py — Image-text retrieval evaluation.
"""

import torch
import torch.nn.functional as F


def evaluate_retrieval(data, ks=(1, 5, 10)):
    """
    Compute I→T and T→I Recall@k for a list of feature dicts.

    Each element of data must have:
        "image" — (1, D) tensor
        "text"  — (C, D) tensor  (C captions per image)

    Returns a dict with keys "I→T R@k", "T→I R@k", and "rSum".
    """
    imgs  = torch.cat([d["image"] for d in data], dim=0)   # (N, D)
    texts = torch.cat([d["text"]  for d in data], dim=0)   # (N*C, D)

    imgs  = F.normalize(imgs,  dim=-1)
    texts = F.normalize(texts, dim=-1)

    N = len(data)
    C = data[0]["text"].shape[0]   # captions per image

    sim = imgs @ texts.T            # (N, N*C)
    results = {}

    # I→T: for image i, is any of its C captions among the top-k retrieved texts?
    gt_caps = [list(range(i * C, i * C + C)) for i in range(N)]
    for k in ks:
        hits = sum(
            any(c in sim[i].topk(k).indices.tolist() for c in gt_caps[i])
            for i in range(N)
        )
        results[f"I→T R@{k}"] = hits / N

    # T→I: for caption j, is its ground-truth image (j // C) among the top-k?
    n_caps = N * C
    for k in ks:
        hits = sum(
            (j // C) in sim[:, j].topk(k).indices.tolist()
            for j in range(n_caps)
        )
        results[f"T→I R@{k}"] = hits / n_caps

    results["rSum"] = sum(
        results[f"I→T R@{k}"] + results[f"T→I R@{k}"] for k in ks
    ) * 100

    return results
