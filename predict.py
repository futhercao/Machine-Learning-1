"""
Predict on a test set and produce submission CSV.

Two input modes:
  --test_npy  /path/to/test_data.npy   shape (M, P>=1024, 6)  fp16/fp32
  --test_dir  /path/to/raw_dir/        same .txt format as training (recurses categories
                                       OR a flat dir of {sample_id}.txt files)
  --test_list /path/to/test.txt        list of sample ids (xxx_0001) with --test_root
                                       pointing to the raw_resampled root

Output: <out_csv> with columns "id,class_name" (no header).

Usage examples:
    # Single model
    python predict.py --ckpts checkpoints/pointnet2_ssg_s2026.pt \
        --test_npy test.npy --ids_npy test_ids.npy --out submit.csv

    # Ensemble + TTA (10 random samplings averaged)
    python predict.py --ckpts checkpoints/pointnet2_ssg_s2026.pt checkpoints/dgcnn_s2026.pt \
        --test_npy test.npy --ids_npy test_ids.npy --out submit.csv --tta 10
"""
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F

from models import build_model
from data_loader import pc_normalize


def parse_ckpt_name(path):
    """Infer model name from filename like pointnet2_ssg_s2026.pt or dgcnn_s42.pt"""
    name = os.path.basename(path).replace('.pt', '')
    # strip trailing _sXXXX
    parts = name.split('_')
    if parts[-1].startswith('s') and parts[-1][1:].isdigit():
        parts = parts[:-1]
    return '_'.join(parts)


def load_model(ckpt_path, device, use_normals=True):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = parse_ckpt_name(ckpt_path)
    args = ck.get('args', {})
    use_normals = bool(args.get('use_normals', use_normals))
    m = build_model(model_name, num_classes=40, use_normals=use_normals).to(device)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m, model_name, use_normals


def read_txt_pointcloud(path):
    """Read a .txt file (x,y,z,nx,ny,nz per line)."""
    arr = np.loadtxt(path, delimiter=',', dtype=np.float32)
    if arr.shape[0] < 10000:
        pad = np.tile(arr[-1:], (10000 - arr.shape[0], 1))
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > 10000:
        arr = arr[:10000]
    return arr


def load_test_data(args):
    """Return (data, ids). data: (M, P, 6) fp32, ids: list of sample id strings."""
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
        # walk dir for .txt files
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


def sample_points(arr, n=1024, seed=None):
    """arr (10000, 6) -> (n, 6) random sample. Returns view if seed=None else deterministic."""
    rng = np.random.default_rng(seed)
    if seed is None:
        idx = rng.choice(arr.shape[0], n, replace=False)
    else:
        idx = np.arange(n) if n <= arr.shape[0] else rng.choice(arr.shape[0], n, replace=True)
    return arr[idx]


def predict_batch(models, x, device, use_normals_list):
    """Average softmax probabilities across an ensemble of models on a batch."""
    probs = None
    n = 0
    for m, un in zip(models, use_normals_list):
        feat = x if un else x[:, :, :3]
        logits = m(feat)
        p = F.softmax(logits, dim=1)
        probs = p if probs is None else probs + p
        n += 1
    return probs / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', required=True, help='one or more ckpt paths')
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
    p.add_argument('--tta', type=int, default=10, help='# random samplings to average')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"[load test]")
    data, ids = load_test_data(args)
    print(f"  data {data.shape}, ids n={len(ids)}")

    # normalize whole sample first (in xyz)
    pts = data[:, :, :3]
    centroid = pts.mean(axis=1, keepdims=True)
    pts = pts - centroid
    m = np.linalg.norm(pts, axis=2).max(axis=1, keepdims=True)[:, :, None]
    pts = pts / (m + 1e-9)
    data = data.copy()
    data[:, :, :3] = pts

    shape_names = np.load(args.shape_names)
    print(f"  classes: {len(shape_names)}")

    print(f"[load models]")
    models, use_normals_list = [], []
    for cp in args.ckpts:
        m, name, un = load_model(cp, device)
        print(f"  {cp} -> model={name} use_normals={un}")
        models.append(m); use_normals_list.append(un)

    M = len(data)
    accum_probs = np.zeros((M, 40), dtype=np.float64)
    rng = np.random.default_rng(2026)

    for t in range(args.tta):
        print(f"[tta {t+1}/{args.tta}]")
        # sample once across all samples (same seed for all batches in this TTA round)
        seed = int(rng.integers(0, 10**9))
        sampled = np.zeros((M, args.num_points, 6), dtype=np.float32)
        for i in range(M):
            inds = np.random.default_rng(seed + i).choice(10000, args.num_points, replace=False)
            sampled[i] = data[i, inds]
        # batched inference
        for i in range(0, M, args.batch_size):
            x = torch.from_numpy(sampled[i:i + args.batch_size]).to(device)
            with torch.no_grad():
                prob = predict_batch(models, x, device, use_normals_list)
            accum_probs[i:i + args.batch_size] += prob.cpu().numpy()
    accum_probs /= args.tta
    pred_idx = accum_probs.argmax(axis=1)
    pred_names = [str(shape_names[i]) for i in pred_idx]

    # write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for sid, name in zip(ids, pred_names):
            f.write(f"{sid},{name}\n")
    print(f"[done] wrote {args.out}  ({M} rows)")


if __name__ == '__main__':
    main()
