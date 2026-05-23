"""Evaluate ensemble + TTA on the held-out validation split."""
import argparse, os, numpy as np, torch
import torch.nn.functional as F
from data_loader import stratified_split
from models import build_model


def parse_ckpt_name(path):
    name = os.path.basename(path).replace('.pt', '')
    parts = name.split('_')
    if parts[-1].startswith('s') and parts[-1][1:].isdigit():
        parts = parts[:-1]
    return '_'.join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', required=True)
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
    print(f"[val] {M} samples")

    pts = val_data[:, :, :3]
    centroid = pts.mean(axis=1, keepdims=True)
    pts = pts - centroid
    m = np.linalg.norm(pts, axis=2).max(axis=1, keepdims=True)[:, :, None]
    val_data[:, :, :3] = pts / (m + 1e-9)

    models, uns = [], []
    for cp in args.ckpts:
        ck = torch.load(cp, map_location=device, weights_only=False)
        un = bool(ck.get('args', {}).get('use_normals', True))
        mn = parse_ckpt_name(cp)
        mdl = build_model(mn, num_classes=40, use_normals=un).to(device).eval()
        mdl.load_state_dict(ck['state_dict'])
        models.append(mdl); uns.append(un)
        print(f"  {cp} -> {mn} use_normals={un}")

    accum = np.zeros((M, 40), dtype=np.float64)
    rng = np.random.default_rng(2026)
    for t in range(args.tta):
        seed = int(rng.integers(0, 10**9))
        sampled = np.zeros((M, args.num_points, 6), dtype=np.float32)
        for i in range(M):
            ind = np.random.default_rng(seed + i).choice(10000, args.num_points, replace=False)
            sampled[i] = val_data[i, ind]
        for i in range(0, M, args.batch_size):
            x = torch.from_numpy(sampled[i:i + args.batch_size]).to(device)
            probs = None
            with torch.no_grad():
                for mdl, un in zip(models, uns):
                    feat = x if un else x[:, :, :3]
                    p = F.softmax(mdl(feat), dim=1)
                    probs = p if probs is None else probs + p
            accum[i:i + args.batch_size] += (probs / len(models)).cpu().numpy()
        print(f"  tta {t+1}/{args.tta}")
    accum /= args.tta
    pred = accum.argmax(axis=1)

    correct = (pred == val_labels).sum()
    iAcc = correct / M
    cls_c = np.zeros(40); cls_t = np.zeros(40)
    for c in range(40):
        m = val_labels == c
        if m.sum() > 0:
            cls_c[c] = (pred[m] == c).sum()
            cls_t[c] = m.sum()
    cAcc = (cls_c / np.maximum(cls_t, 1)).mean()
    print(f"\n[ensemble] iAcc={iAcc:.4f}  cAcc={cAcc:.4f}  ({correct}/{M})")


if __name__ == '__main__':
    main()
