"""Per-class anchored cordon (always keep audited samples)."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _same_class_distances(features: np.ndarray, labels: np.ndarray):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(features)
    kn = max(1, int(np.sqrt(n / 2.0)))
    neigh_idx = np.full((n, kn), -1, dtype=np.int64)
    neigh_sq = np.zeros((n, kn), dtype=np.float64)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) == 1:
            continue
        k = min(kn, len(idx) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(features[idx])
        dist, ind = nn.kneighbors(features[idx], return_distance=True)
        dist, ind = dist[:, 1:], ind[:, 1:]
        global_ind = idx[ind]
        sq = dist ** 2
        if k < kn:
            pad_i = np.repeat(global_ind[:, :1], kn - k, axis=1)
            pad_d = np.repeat(sq[:, :1], kn - k, axis=1)
            global_ind = np.concatenate([global_ind, pad_i], axis=1)
            sq = np.concatenate([sq, pad_d], axis=1)
        neigh_idx[idx] = global_ind
        neigh_sq[idx] = sq
    s_dist = neigh_sq.mean(axis=1)
    means = []
    for i in range(n):
        valid = neigh_idx[i][neigh_idx[i] >= 0]
        if len(valid) == 0:
            means.append(features[i])
        else:
            means.append(features[valid].mean(axis=0))
    means = np.stack(means, axis=0)
    m_dist = ((features - means) ** 2).sum(axis=1)
    return s_dist, m_dist


def anchored_cordon(features, labels, audited, remember_rate: float = 0.6) -> np.ndarray:
    """Keep every audited sample; keep unaudited samples inside their class cordon."""
    labels = np.asarray(labels)
    audited = np.asarray(audited, dtype=bool)
    if len(audited) != len(labels):
        raise ValueError("audited mask and labels must have the same length")
    s_dist, m_dist = _same_class_distances(features, labels)
    mask = audited.copy()
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        num_keep = max(1, int(remember_rate * len(idx)))
        delta_s = np.sort(s_dist[idx])[min(num_keep, len(idx) - 1)]
        delta_m = np.sort(m_dist[idx])[min(num_keep, len(idx) - 1)]
        inside = (s_dist[idx] <= delta_s) & (m_dist[idx] <= delta_m)
        mask[idx] = mask[idx] | inside
    if mask.sum() == 0:
        mask[int(np.argmin(s_dist + m_dist))] = True
    return mask
