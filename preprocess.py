"""
Pre-process ModelNet40 normal_resampled .txt files into a single numpy file.

Only processes samples actually present on disk (= 9843 train samples; the standard
ModelNet40 test set is NOT in this download because it's released live for evaluation).

Output:
  data.npy   (N, 10000, 6) fp16 — full xyz+normal
  labels.npy (N,) int64 — class index
  ids.npy    (N,) <U32 — e.g. 'airplane_0001'
  shape_names.npy (40,) <U32
"""
import argparse
import os
import time
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True,
                   help='Root directory containing class subdirs with .txt files')
    p.add_argument('--out_dir', required=True,
                   help='Output directory for preprocessed .npy files')
    p.add_argument('--shape_names_file', default='modelnet40_shape_names.txt',
                   help='File in --root listing 40 class names (default: modelnet40_shape_names.txt)')
    p.add_argument('--sample_list', default='modelnet40_train.txt',
                   help='File in --root listing sample IDs to process (default: modelnet40_train.txt)')
    p.add_argument('--num_points', type=int, default=10000,
                   help='Number of points per sample (default: 10000)')
    p.add_argument('--num_features', type=int, default=6,
                   help='Feature dimension xyz+normal=6 or xyz-only=3 (default: 6)')
    p.add_argument('--out_prefix', default='',
                   help='Optional prefix for output filenames (e.g. "test_" -> test_data.npy)')
    p.add_argument('--use_fp16', action='store_true', default=True,
                   help='Store as float16 to save disk space (default)')
    p.add_argument('--no_fp16', action='store_true',
                   help='Store as float32 instead of float16')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    use_fp16 = not args.no_fp16

    with open(os.path.join(args.root, args.shape_names_file), encoding='utf-8') as f:
        shape_names = [s.strip() for s in f if s.strip()]
    name2idx = {n: i for i, n in enumerate(shape_names)}
    print(f"[shapes] {len(shape_names)} classes (from {args.shape_names_file})")

    with open(os.path.join(args.root, args.sample_list), encoding='utf-8') as f:
        sample_ids = [s.strip() for s in f if s.strip()]
    print(f"[sample list] {len(sample_ids)} samples (from {args.sample_list})")

    # filter to only those existing on disk
    samples = []
    for sid in sample_ids:
        cls = '_'.join(sid.split('_')[:-1])
        path = os.path.join(args.root, cls, sid + '.txt')
        if os.path.exists(path):
            samples.append((sid, cls, path))
    print(f"[on disk] {len(samples)} files exist")

    N = len(samples)
    NP = args.num_points
    NF = args.num_features
    dtype = np.float16 if use_fp16 else np.float32
    data = np.zeros((N, NP, NF), dtype=dtype)
    labels = np.zeros((N,), dtype=np.int64)
    ids = np.empty((N,), dtype=object)

    t0 = time.time()
    for i, (sid, cls, path) in enumerate(samples):
        arr = np.loadtxt(path, delimiter=',', dtype=np.float32)
        if arr.shape[0] >= NP:
            arr = arr[:NP]
        else:
            pad = np.tile(arr[-1:], (NP - arr.shape[0], 1))
            arr = np.concatenate([arr, pad], axis=0)
        if arr.shape[1] > NF:
            arr = arr[:, :NF]
        data[i] = arr.astype(dtype)
        labels[i] = name2idx[cls]
        ids[i] = sid
        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (N - i - 1)
            print(f"  [{i+1}/{N}] {dt:.1f}s elapsed, ETA {eta:.1f}s")

    pf = args.out_prefix
    np.save(os.path.join(args.out_dir, f'{pf}data.npy'), data)
    np.save(os.path.join(args.out_dir, f'{pf}labels.npy'), labels)
    np.save(os.path.join(args.out_dir, f'{pf}ids.npy'), np.array(list(ids), dtype='U32'))
    np.save(os.path.join(args.out_dir, f'{pf}shape_names.npy'), np.array(shape_names, dtype='U32'))
    print(f"[done] saved to {args.out_dir}")
    print(f"  {pf}data.npy   shape={data.shape} dtype={data.dtype} size={data.nbytes/1e6:.0f} MB")


if __name__ == '__main__':
    main()
