"""
Train a point cloud classifier.

Usage:
    python train.py --model pointnet2_ssg --epochs 200 --device cuda
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import ModelNet40Dataset, stratified_split
from models import build_model


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def evaluate(model, loader, device, num_classes=40):
    model.eval()
    correct = 0
    total = 0
    cls_correct = torch.zeros(num_classes)
    cls_total = torch.zeros(num_classes)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for c in range(num_classes):
                m = (y == c)
                if m.sum() > 0:
                    cls_correct[c] += (pred[m] == c).sum().item()
                    cls_total[c] += m.sum().item()
    instance_acc = correct / max(total, 1)
    class_acc = (cls_correct / cls_total.clamp(min=1)).mean().item()
    return instance_acc, class_acc


def warmup_cosine_lr(step, total, warmup, base, min_ratio=0.001):
    import math
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='preprocessed')
    p.add_argument('--model', required=True)
    p.add_argument('--num_points', type=int, default=1024)
    p.add_argument('--use_normals', type=int, default=1)
    p.add_argument('--num_classes', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_pct', type=float, default=0.02)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--ckpt_dir', default='checkpoints')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=40)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"[load] {args.data_dir}")
    data = np.load(os.path.join(args.data_dir, 'data.npy'), mmap_mode='r')  # (N, 10000, 6) fp16
    labels = np.load(os.path.join(args.data_dir, 'labels.npy'))
    print(f"  data {data.shape} {data.dtype}, labels {labels.shape}")

    train_idx, val_idx = stratified_split(labels, val_ratio=args.val_ratio, seed=args.seed)
    print(f"  split: train={len(train_idx)} val={len(val_idx)}")

    use_normals = bool(args.use_normals)
    train_ds = ModelNet40Dataset(data, labels, train_idx, num_points=args.num_points,
                                  use_normals=use_normals, augment=True)
    val_ds = ModelNet40Dataset(data, labels, val_idx, num_points=args.num_points,
                                use_normals=use_normals, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    model = build_model(args.model, num_classes=args.num_classes, use_normals=use_normals).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model} use_normals={use_normals} params={n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.999))

    steps_per_ep = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_ep
    warmup_steps = int(args.warmup_pct * total_steps)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, f"{args.model}_s{args.seed}.pt")
    best_acc = 0.0; bad = 0
    step = 0
    for ep in range(args.epochs):
        t0 = time.time(); model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            lr = warmup_cosine_lr(step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups: g['lr'] = lr
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.size(0)
            step += 1
        tr_loss /= max(1, len(train_loader))
        tr_acc = tr_correct / max(tr_total, 1)

        val_iacc, val_cacc = evaluate(model, val_loader, device, args.num_classes)
        dt = time.time() - t0
        mk = ''
        if val_iacc > best_acc:
            best_acc = val_iacc; bad = 0
            torch.save({'state_dict': model.state_dict(), 'args': vars(args),
                        'val_iacc': val_iacc, 'val_cacc': val_cacc, 'epoch': ep + 1}, best_path)
            mk = '  *'
        else:
            bad += 1
        print(f"ep {ep+1:3d}/{args.epochs} | lr {lr:.5f} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} "
              f"| val iAcc {val_iacc:.4f} cAcc {val_cacc:.4f} | {dt:.1f}s{mk}")
        if bad >= args.patience:
            print(f"[stop] no improvement for {args.patience}"); break
    print(f"[done] best val iAcc = {best_acc:.4f}  -> {best_path}")


if __name__ == '__main__':
    main()
