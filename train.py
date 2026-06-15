"""Train PointMLP on ModelNet40 (xyz only, AMP mixed precision).

PointMLP recipe (Ma et al., ICLR 2022):
  - SGD momentum=0.9, weight_decay=2e-4
  - Cosine LR: 0.1 -> 0.005 over full epochs
  - Label smoothing = 0.2
  - Augmentation: random scale (0.66–1.5) + translate (±0.2), no rotation

Saves three checkpoints:
  - <ckpt>.pt         best val iAcc
  - <ckpt>_bestcacc.pt  best val cAcc
  - <ckpt>.last         final epoch

Usage:
    python train.py --epochs 200 --batch_size 32 --seed 2026
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import ModelNet40Dataset, pc_normalize, stratified_split
from models import build_model


class ModelNetADataset(torch.utils.data.Dataset):
    """xyz-only dataset over the 10000-pt preprocessed data for PointMLP."""

    def __init__(self, data, labels, indices, num_points=1024, augment=True):
        self.data = data
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ri = int(self.indices[i])
        sample = np.asarray(self.data[ri], dtype=np.float32)
        N = sample.shape[0]
        if self.augment:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            choice = np.arange(self.num_points)
        pts = sample[choice, :3]  # xyz only
        pts = pc_normalize(pts)
        if self.augment:
            # PointMLP-style augmentation: anisotropic scale + translation
            pts = pts * np.random.uniform(0.66, 1.5, 3).astype(np.float32)
            pts = pts + np.random.uniform(-0.2, 0.2, 3).astype(np.float32)
        return torch.from_numpy(pts.astype(np.float32)), int(self.labels[ri])


def cosine_lr(epoch, total, lr0, lr_min):
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * epoch / total))


def evaluate(model, loader, device, num_classes=40):
    """Evaluate in fp32 for consistent, accurate checkpoint selection."""
    model.eval()
    correct = 0
    total = 0
    cls_c = np.zeros(num_classes)
    cls_t = np.zeros(num_classes)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for c in range(num_classes):
                m = (y == c)
                if m.any():
                    cls_t[c] += m.sum().item()
                    cls_c[c] += (pred[m] == c).sum().item()
    iAcc = correct / max(total, 1)
    cAcc = (cls_c / np.maximum(cls_t, 1)).mean()
    return iAcc, cAcc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='preprocessed')
    p.add_argument('--ckpt', default='checkpoints/pointmlp.pt')
    p.add_argument('--log', default='logs/train_pointmlp.log')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_points', type=int, default=1024)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--lr_min', type=float, default=0.005)
    p.add_argument('--weight_decay', type=float, default=2e-4)
    p.add_argument('--label_smoothing', type=float, default=0.2)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--eval_every', type=int, default=2)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    data = np.load(os.path.join(args.data_dir, 'data.npy'), mmap_mode='r')
    labels = np.load(os.path.join(args.data_dir, 'labels.npy'))
    train_idx, val_idx = stratified_split(labels, val_ratio=0.1, seed=args.seed)
    print(f"[split] train={len(train_idx)} val={len(val_idx)} (seed {args.seed})")

    train_ds = ModelNetADataset(data, labels, train_idx, args.num_points, augment=True)
    val_ds = ModelNetADataset(data, labels, val_idx, args.num_points, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    model = build_model('pointmlp', num_classes=40, points=args.num_points).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    amp_device = 'cuda' if device.type == 'cuda' else 'cpu'
    print(f"[model] PointMLP {n_params / 1e6:.2f}M params, AMP ({amp_device})")

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(amp_device)

    log_f = open(args.log, 'a')
    best_iacc = 0.0
    best_cacc = 0.0

    for ep in range(args.epochs):
        lr = cosine_lr(ep, args.epochs, args.lr, args.lr_min)
        for g in opt.param_groups:
            g['lr'] = lr

        model.train()
        t0 = time.time()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(amp_device):
                logits = model(x)
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * y.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.size(0)

        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        do_eval = ((ep + 1) % args.eval_every == 0) or (ep + 1 == args.epochs)
        if do_eval:
            iAcc, cAcc = evaluate(model, val_loader, device)
        else:
            iAcc, cAcc = float('nan'), float('nan')

        dt = time.time() - t0
        mk = ''
        save = {
            'state_dict': model.state_dict(),
            'epoch': ep + 1,
            'args': vars(args),
            'val_iacc': iAcc,
            'val_cacc': cAcc,
            'model': 'pointmlp',
        }
        torch.save(save, args.ckpt + '.last')
        if do_eval and iAcc > best_iacc:
            best_iacc = iAcc
            torch.save(save, args.ckpt)
            mk += ' *iAcc'
        if do_eval and cAcc > best_cacc:
            best_cacc = cAcc
            torch.save(save, args.ckpt.replace('.pt', '_bestcacc.pt'))
            mk += ' *cAcc'

        line = (f"ep {ep + 1:3d}/{args.epochs} lr={lr:.4f} loss={tr_loss:.4f} "
                f"tr_acc={tr_acc:.4f} val_iAcc={iAcc:.4f} val_cAcc={cAcc:.4f} "
                f"[{dt:.0f}s]{mk}")
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    print(f"[done] best val iAcc={best_iacc:.4f} cAcc={best_cacc:.4f}")
    log_f.write(f"[done] best val iAcc={best_iacc:.4f} cAcc={best_cacc:.4f}\n")
    log_f.close()


if __name__ == '__main__':
    main()
