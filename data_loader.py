"""
ModelNet40 dataset loader with on-the-fly augmentation.

Reads pre-processed data.npy of shape (N, 10000, 6) fp16,
returns (B, num_points, C) with augmentations applied.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


# Augmentation utils ----------------------------------------------------------
def pc_normalize(points):
    """Normalize to unit ball, center at origin. points: (N, 3) np.float32"""
    centroid = points.mean(axis=0)
    points = points - centroid
    m = np.max(np.linalg.norm(points, axis=1))
    points = points / (m + 1e-9)
    return points


def random_rotate_z(points, normals=None):
    """Random rotation around Z axis (up axis)."""
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    points = points @ R.T
    if normals is not None:
        normals = normals @ R.T
    return points, normals


def random_anisotropic_scale(points, lo=0.8, hi=1.25):
    s = np.random.uniform(lo, hi, size=3).astype(np.float32)
    return points * s


def random_translate(points, shift=0.1):
    t = np.random.uniform(-shift, shift, size=3).astype(np.float32)
    return points + t


def random_jitter(points, sigma=0.01, clip=0.05):
    noise = np.clip(sigma * np.random.randn(*points.shape).astype(np.float32), -clip, clip)
    return points + noise


def random_point_dropout(points, normals=None, max_dropout=0.875):
    """Randomly replace some points with the first point (input dropout)."""
    p = np.random.uniform(0, max_dropout)
    mask = np.random.rand(points.shape[0]) < p
    if mask.any():
        points[mask] = points[0]
        if normals is not None:
            normals[mask] = normals[0]
    return points, normals


# Dataset ---------------------------------------------------------------------
class ModelNet40Dataset(Dataset):
    def __init__(self, data, labels, indices, num_points=1024, use_normals=True,
                 augment=True, fps=False):
        """
        data:   (N, 10000, 6) array (may be fp16) — full dataset, all on RAM
        labels: (N,)
        indices: which rows of `data` belong to this split
        num_points: how many points to use per sample
        """
        self.data = data
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.num_points = num_points
        self.use_normals = use_normals
        self.augment = augment
        self.fps = fps

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        sample = np.asarray(self.data[real_idx], dtype=np.float32)  # (10000, 6)
        N = sample.shape[0]

        # subsample to num_points
        if self.num_points < N:
            if self.augment:
                choice = np.random.choice(N, self.num_points, replace=False)
            else:
                choice = np.arange(self.num_points)  # deterministic prefix
            sample = sample[choice]

        points = sample[:, :3].astype(np.float32)
        normals = sample[:, 3:6].astype(np.float32) if sample.shape[1] >= 6 else None

        # normalize
        points = pc_normalize(points)

        if self.augment:
            points, normals = random_rotate_z(points, normals)
            points = random_anisotropic_scale(points)
            points = random_translate(points, shift=0.1)
            points = random_jitter(points)
            points, normals = random_point_dropout(points, normals)

        if self.use_normals and normals is not None:
            feat = np.concatenate([points, normals], axis=1)  # (P, 6)
        else:
            feat = points

        label = int(self.labels[real_idx])
        return torch.from_numpy(feat.astype(np.float32)), label


def stratified_split(labels, val_ratio=0.1, seed=2026):
    """Per-class stratified split."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


if __name__ == '__main__':
    data = np.load('preprocessed/data.npy', mmap_mode='r')
    labels = np.load('preprocessed/labels.npy')
    print(f"data {data.shape} {data.dtype}, labels {labels.shape}")
    train_idx, val_idx = stratified_split(labels, val_ratio=0.1)
    print(f"split: train={len(train_idx)} val={len(val_idx)}")
    ds = ModelNet40Dataset(data, labels, train_idx, num_points=1024)
    x, y = ds[0]
    print(f"sample: {x.shape} {y}")
