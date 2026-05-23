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
    p.add_argument('--root', required=True)
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.root, 'modelnet40_shape_names.txt'), encoding='utf-8') as f:
        shape_names = [s.strip() for s in f if s.strip()]
    name2idx = {n: i for i, n in enumerate(shape_names)}
    print(f"[shapes] {len(shape_names)} classes")

    with open(os.path.join(args.root, 'modelnet40_train.txt'), encoding='utf-8') as f:
        train_ids = [s.strip() for s in f if s.strip()]  # 'airplane_0001'
    print(f"[train list] {len(train_ids)} samples")

    # filter to only those existing on disk
    samples = []
    for sid in train_ids:
        cls = '_'.join(sid.split('_')[:-1])  # e.g. 'night_stand_0001' -> 'night_stand'
        path = os.path.join(args.root, cls, sid + '.txt')
        if os.path.exists(path):
            samples.append((sid, cls, path))
    print(f"[on disk] {len(samples)} files exist")

    N = len(samples)
    data = np.zeros((N, 10000, 6), dtype=np.float16)
    labels = np.zeros((N,), dtype=np.int64)
    ids = np.empty((N,), dtype=object)

    t0 = time.time()
    for i, (sid, cls, path) in enumerate(samples):
        arr = np.loadtxt(path, delimiter=',', dtype=np.float32)
        if arr.shape != (10000, 6):
            # some files might have different point counts; pad/truncate
            if arr.shape[0] >= 10000:
                arr = arr[:10000]
            else:
                pad = np.tile(arr[-1:], (10000 - arr.shape[0], 1))
                arr = np.concatenate([arr, pad], axis=0)
        data[i] = arr.astype(np.float16)
        labels[i] = name2idx[cls]
        ids[i] = sid
        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (N - i - 1)
            print(f"  [{i+1}/{N}] {dt:.1f}s elapsed, ETA {eta:.1f}s")

    np.save(os.path.join(args.out_dir, 'data.npy'), data)
    np.save(os.path.join(args.out_dir, 'labels.npy'), labels)
    np.save(os.path.join(args.out_dir, 'ids.npy'), np.array(list(ids), dtype='U32'))
    np.save(os.path.join(args.out_dir, 'shape_names.npy'), np.array(shape_names, dtype='U32'))
    print(f"[done] saved to {args.out_dir}")
    print(f"  data.npy   shape={data.shape} dtype={data.dtype} size={data.nbytes/1e6:.0f} MB")


if __name__ == '__main__':
    main()
