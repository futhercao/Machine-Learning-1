"""Run full test set evaluation with progress reporting.

Usage:
    python eval_test.py

Configurable via variables at the top of main() or edit the defaults below.
"""
import numpy as np
import torch
import torch.nn.functional as F
from models import build_model
import time


def main():
    TEST_NPY = r"C:\Users\Administrator\Desktop\test_data.npy"
    TEST_IDS = r"C:\Users\Administrator\Desktop\test_ids.npy"
    CKPT = "checkpoints/pointmlp.pt"
    SHAPE_NAMES = "preprocessed/shape_names.npy"
    OUT = r"C:\Users\Administrator\Desktop\test_full_tta10.csv"
    TTA = 10
    BATCH_SIZE = 16
    NUM_POINTS = 1024

    device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    # Load data
    t0 = time.time()
    data = np.load(TEST_NPY).astype(np.float32)
    ids = np.load(TEST_IDS, allow_pickle=True)
    M = len(data)
    print(f"[load] {M} samples, shape={data.shape}, {time.time()-t0:.1f}s", flush=True)

    # Normalize
    t0 = time.time()
    pts = data[:, :, :3].copy()
    centroid = pts.mean(axis=1, keepdims=True)
    pts = pts - centroid
    rad = np.linalg.norm(pts, axis=2).max(axis=1, keepdims=True)[:, :, None]
    data[:, :, :3] = pts / (rad + 1e-9)
    print(f"[norm] {time.time()-t0:.1f}s", flush=True)

    # Load model
    t0 = time.time()
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = build_model('pointmlp', num_classes=40, points=NUM_POINTS).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.2f}M params, epoch={ck.get('epoch')}, {time.time()-t0:.1f}s", flush=True)

    # Load shape names for ground truth comparison
    shape_names = np.load(SHAPE_NAMES, allow_pickle=True)
    name2idx = {str(n): i for i, n in enumerate(shape_names)}

    # Build ground truth labels from ids
    true_labels = np.array([name2idx['_'.join(str(sid).split('_')[:-1])] for sid in ids])
    print(f"[labels] {len(true_labels)} ground truth labels", flush=True)

    # TTA
    accum_probs = np.zeros((M, 40), dtype=np.float64)
    rng = np.random.default_rng(2026)
    tta_times = []

    for t in range(TTA):
        t_start = time.time()
        seed = int(rng.integers(0, 10**9))
        sampled = np.zeros((M, NUM_POINTS, 3), dtype=np.float32)
        for i in range(M):
            ind = np.random.default_rng(seed + i).choice(data.shape[1], NUM_POINTS, replace=False)
            sampled[i] = data[i, ind, :3]

        for i in range(0, M, BATCH_SIZE):
            x = torch.from_numpy(sampled[i:i + BATCH_SIZE]).to(device)
            with torch.no_grad():
                p = F.softmax(model(x), dim=1)
            accum_probs[i:i + BATCH_SIZE] += p.cpu().numpy()

        dt = time.time() - t_start
        tta_times.append(dt)

        # Show current accuracy (greedy, for progress)
        cur_probs = accum_probs / (t + 1)
        cur_pred = cur_probs.argmax(axis=1)
        cur_iacc = (cur_pred == true_labels).mean()
        print(f"  [tta {t+1}/{TTA}] {dt:.0f}s  cum_iAcc={cur_iacc:.4f}", flush=True)

    accum_probs /= TTA
    pred_idx = accum_probs.argmax(axis=1)

    # Compute metrics
    correct = (pred_idx == true_labels).sum()
    iAcc = correct / M
    cls_c = np.zeros(40)
    cls_t = np.zeros(40)
    for c in range(40):
        m = true_labels == c
        if m.sum() > 0:
            cls_c[c] = (pred_idx[m] == c).sum()
            cls_t[c] = m.sum()
    cAcc = (cls_c / np.maximum(cls_t, 1)).mean()

    print(f"\n{'='*60}")
    print(f"Test Results (TTA={TTA}, {M} samples)")
    print(f"  Instance Accuracy: {iAcc:.4f} ({correct}/{M})")
    print(f"  Class-mean Accuracy: {cAcc:.4f}")
    print(f"  Total time: {sum(tta_times):.0f}s (TTA rounds: {[f'{x:.0f}s' for x in tta_times]})")
    print(f"{'='*60}")

    # Per-class breakdown
    print(f"\nPer-class recall (worst 10):")
    per_class = [(c, cls_c[c]/cls_t[c], int(cls_c[c]), int(cls_t[c]), str(shape_names[c]))
                 for c in range(40) if cls_t[c] > 0]
    per_class.sort(key=lambda x: x[1])
    for c, r, cc, ct, name in per_class[:10]:
        print(f"  {c:2d} {name:20s} R={r:.3f}  ({cc}/{ct})")

    # Save
    with open(OUT, 'w', encoding='utf-8') as f:
        for sid, idx in zip(ids, pred_idx):
            f.write(f"{sid},{shape_names[idx]}\n")
    print(f"\nSaved: {OUT}")

    return iAcc, cAcc


if __name__ == '__main__':
    main()
