"""Evaluate PointMLP + TTA on the held-out validation split.

Usage:
    python eval_val.py --ckpt checkpoints/pointmlp.pt --tta 10
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import stratified_split
from models import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Path to PointMLP checkpoint')
    p.add_argument('--data_dir', default='preprocessed')
    p.add_argument('--num_points', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--tta', type=int, default=10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    data = np.load(os.path.join(args.data_dir, 'data.npy'), mmap_mode='r')
    labels = np.load(os.path.join(args.data_dir, 'labels.npy'))
    _, val_idx = stratified_split(labels, val_ratio=0.1, seed=args.seed)
    val_data = np.array(data[val_idx]).astype(np.float32)
    val_labels = labels[val_idx]
    M = len(val_idx)
    print(f"[val] {M} samples (seed {args.seed})")

    # Pre-normalize: center + unit-ball over all 10000 points
    pts = val_data[:, :, :3].copy()
    centroid = pts.mean(axis=1, keepdims=True)
    pts = pts - centroid
    rad = np.linalg.norm(pts, axis=2).max(axis=1, keepdims=True)[:, :, None]
    val_data[:, :, :3] = pts / (rad + 1e-9)

    # Load model
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    npts = ck.get('args', {}).get('num_points', args.num_points)
    model = build_model('pointmlp', num_classes=40, points=npts).to(device).eval()
    model.load_state_dict(ck['state_dict'])
    n_params = sum(p.numel() for p in model.parameters())
    val_iacc_ckpt = ck.get('val_iacc', float('nan'))
    val_cacc_ckpt = ck.get('val_cacc', float('nan'))
    print(f"  PointMLP {n_params / 1e6:.2f}M params")
    print(f"  ckpt epoch {ck.get('epoch', '?')} val_iAcc={val_iacc_ckpt:.4f} val_cAcc={val_cacc_ckpt:.4f}")

    # TTA evaluation
    accum = np.zeros((M, 40), dtype=np.float64)
    rng = np.random.default_rng(2026)

    for t in range(args.tta):
        seed = int(rng.integers(0, 10 ** 9))
        sampled = np.zeros((M, args.num_points, 3), dtype=np.float32)
        for i in range(M):
            ind = np.random.default_rng(seed + i).choice(val_data.shape[1], args.num_points, replace=False)
            sampled[i] = val_data[i, ind, :3]  # xyz only
        for i in range(0, M, args.batch_size):
            x = torch.from_numpy(sampled[i:i + args.batch_size]).to(device)
            with torch.no_grad():
                p = F.softmax(model(x), dim=1)
            accum[i:i + args.batch_size] += p.cpu().numpy()
        print(f"  tta {t + 1}/{args.tta}")

    accum /= args.tta
    pred = accum.argmax(axis=1)

    correct = (pred == val_labels).sum()
    iAcc = correct / M
    cls_c = np.zeros(40)
    cls_t = np.zeros(40)
    for c in range(40):
        m = val_labels == c
        if m.sum() > 0:
            cls_c[c] = (pred[m] == c).sum()
            cls_t[c] = m.sum()
    cAcc = (cls_c / np.maximum(cls_t, 1)).mean()

    print(f"\n[PointMLP + TTA{args.tta}] iAcc={iAcc:.4f}  cAcc={cAcc:.4f}  ({correct}/{M})")

    # Per-class breakdown (most useful for debugging)
    shape_names = np.load(os.path.join(args.data_dir, 'shape_names.npy'), allow_pickle=True)
    for c in range(40):
        if cls_t[c] > 0:
            print(f"  {c:2d} {str(shape_names[c]):20s} R={cls_c[c] / cls_t[c]:.3f}  ({int(cls_c[c])}/{int(cls_t[c])})")


if __name__ == '__main__':
    main()
