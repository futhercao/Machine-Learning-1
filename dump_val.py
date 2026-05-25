"""
Materialize the validation split so you can audit it manually.

Re-runs the exact same deterministic stratified split used in train.py
(val_ratio=0.1, seed=2026 by default) and saves the IDs + labels to disk.

Usage:
    python dump_val.py
        -> writes preprocessed/val_ids.npy   (M strings, M ~= 966)
                  preprocessed/val_labels.npy (M ints, 0..39)
                  preprocessed/val_class_names.npy (M strings)
                  preprocessed/val_summary.csv (id,label,class_name per row)
"""
import argparse
import os
import numpy as np

from data_loader import stratified_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='preprocessed')
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()

    labels = np.load(os.path.join(args.data_dir, 'labels.npy'))
    ids = np.load(os.path.join(args.data_dir, 'ids.npy'))
    shape_names = np.load(os.path.join(args.data_dir, 'shape_names.npy'))

    train_idx, val_idx = stratified_split(labels, val_ratio=args.val_ratio,
                                          seed=args.seed)
    print(f"split: train={len(train_idx)}, val={len(val_idx)} (val_ratio={args.val_ratio}, seed={args.seed})")

    val_ids = ids[val_idx]
    val_labels = labels[val_idx]
    val_names = np.array([shape_names[c] for c in val_labels])

    np.save(os.path.join(args.data_dir, 'val_ids.npy'), val_ids)
    np.save(os.path.join(args.data_dir, 'val_labels.npy'), val_labels)
    np.save(os.path.join(args.data_dir, 'val_class_names.npy'), val_names)
    print(f"  -> val_ids.npy  ({len(val_ids)} entries)")
    print(f"  -> val_labels.npy")
    print(f"  -> val_class_names.npy")

    csv_path = os.path.join(args.data_dir, 'val_summary.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('id,label,class_name\n')
        for sid, lab, name in zip(val_ids, val_labels, val_names):
            f.write(f"{sid},{int(lab)},{name}\n")
    print(f"  -> val_summary.csv")

    print(f"\nPer-class val sample count:")
    for c in range(40):
        n = int((val_labels == c).sum())
        print(f"  {c:2d} {str(shape_names[c]):20s}  {n}")


if __name__ == '__main__':
    main()
