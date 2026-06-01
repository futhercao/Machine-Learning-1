"""Predict on a test set with PointMLP + TTA and produce submission CSV.

Supports three input modes:
  --test_npy  /path/to/test_data.npy   shape (M, P>=1024, 6)
  --test_dir  /path/to/raw_dir/        .txt files (recursive or flat)
  --test_list /path/to/test.txt + --test_root

Output: <out_csv> with columns "id,class_name" (no header).

Usage:
    python predict.py --ckpt checkpoints/pointmlp.pt \
        --test_npy test.npy --ids_npy test_ids.npy --out submit.csv --tta 10
"""
import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from models import build_model

warnings.filterwarnings('ignore')


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    npts = ck.get('args', {}).get('num_points', 1024)
    m = build_model('pointmlp', num_classes=40, points=npts).to(device)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m


def read_txt_pointcloud(path):
    arr = np.loadtxt(path, delimiter=',', dtype=np.float32)
    if arr.shape[0] < 10000:
        pad = np.tile(arr[-1:], (10000 - arr.shape[0], 1))
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > 10000:
        arr = arr[:10000]
    return arr


def load_test_data(args):
    if args.test_npy:
        data = np.load(args.test_npy).astype(np.float32)
        if args.ids_npy:
            ids = list(np.load(args.ids_npy, allow_pickle=True))
        elif args.ids_txt:
            with open(args.ids_txt) as f:
                ids = [s.strip() for s in f if s.strip()]
        else:
            ids = [f"sample_{i:05d}" for i in range(len(data))]
    elif args.test_list and args.test_root:
        with open(args.test_list) as f:
            sids = [s.strip() for s in f if s.strip()]
        data = np.zeros((len(sids), 10000, 6), dtype=np.float32)
        ids = sids
        for i, sid in enumerate(sids):
            cls = '_'.join(sid.split('_')[:-1])
            path = os.path.join(args.test_root, cls, sid + '.txt')
            if not os.path.exists(path):
                path = os.path.join(args.test_root, sid + '.txt')
            data[i] = read_txt_pointcloud(path)
            if (i + 1) % 200 == 0:
                print(f"  loaded {i+1}/{len(sids)}")
    elif args.test_dir:
        paths = []
        ids = []
        for d, _, files in os.walk(args.test_dir):
            for f in files:
                if f.endswith('.txt') and not f.startswith('modelnet40_'):
                    paths.append(os.path.join(d, f))
                    ids.append(f.replace('.txt', ''))
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        ids = [ids[i] for i in order]
        paths = [paths[i] for i in order]
        data = np.zeros((len(ids), 10000, 6), dtype=np.float32)
        for i, p in enumerate(paths):
            data[i] = read_txt_pointcloud(p)
            if (i + 1) % 200 == 0:
                print(f"  loaded {i+1}/{len(ids)}")
    else:
        raise ValueError("provide --test_npy, --test_list+--test_root, or --test_dir")
    return data, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Path to PointMLP checkpoint')
    p.add_argument('--test_npy', default=None)
    p.add_argument('--ids_npy', default=None)
    p.add_argument('--ids_txt', default=None)
    p.add_argument('--test_list', default=None)
    p.add_argument('--test_root', default=None)
    p.add_argument('--test_dir', default=None)
    p.add_argument('--shape_names', default='preprocessed/shape_names.npy')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_points', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--tta', type=int, default=10,
                   help='Number of random subsample rounds for TTA')
    p.add_argument('--fast', action='store_true',
                   help='Fast mode: TTA=1, no GPU needed')
    args = p.parse_args()

    if args.fast:
        print("[fast mode] tta=1")
        args.tta = 1

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("[load test]")
    data, ids = load_test_data(args)
    print(f"  data {data.shape}, ids n={len(ids)}")

    # Pre-normalize: center + unit-ball over all 10000 points
    pts = data[:, :, :3].copy()
    centroid = pts.mean(axis=1, keepdims=True)
    pts = pts - centroid
    rad = np.linalg.norm(pts, axis=2).max(axis=1, keepdims=True)[:, :, None]
    data[:, :, :3] = pts / (rad + 1e-9)

    shape_names = np.load(args.shape_names)
    print(f"  classes: {len(shape_names)}")

    print(f"[load model] {args.ckpt}")
    model = load_model(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  PointMLP {n_params / 1e6:.2f}M params")

    M = len(data)
    P = data.shape[1]
    accum_probs = np.zeros((M, 40), dtype=np.float64)
    rng = np.random.default_rng(2026)

    for t in range(args.tta):
        print(f"[tta {t + 1}/{args.tta}]")
        seed = int(rng.integers(0, 10 ** 9))
        sampled = np.zeros((M, args.num_points, 3), dtype=np.float32)
        for i in range(M):
            if P > args.num_points:
                inds = np.random.default_rng(seed + i).choice(P, args.num_points, replace=False)
            else:
                inds = np.arange(args.num_points) % P
            sampled[i] = data[i, inds, :3]  # xyz only

        for i in range(0, M, args.batch_size):
            x = torch.from_numpy(sampled[i:i + args.batch_size]).to(device)
            with torch.no_grad():
                p = F.softmax(model(x), dim=1)
            accum_probs[i:i + args.batch_size] += p.cpu().numpy()

    accum_probs /= args.tta
    pred_idx = accum_probs.argmax(axis=1)
    pred_names = [str(shape_names[i]) for i in pred_idx]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for sid, name in zip(ids, pred_names):
            f.write(f"{sid},{name}\n")
    print(f"[done] wrote {args.out}  ({M} rows)")


if __name__ == '__main__':
    main()
