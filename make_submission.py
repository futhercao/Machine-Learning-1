"""
Package Track-1 (ModelNet40) submission:
  <NAME>_<ID>/
    code/                   src files (train, predict, data_loader, models, preprocess)
    checkpoints/            trained .pt files used at inference
    preprocessed/shape_names.npy
    submit.csv              prediction file (if present)
    设计思路.md
    运行说明.md
"""
import argparse
import os
import shutil
import zipfile

SRC = ['train.py', 'predict.py', 'data_loader.py', 'models.py', 'preprocess.py',
       'train_gpu0.sh', 'train_gpu0b.sh', '设计思路.md', '运行说明.md']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='姓名')
    ap.add_argument('--sid', required=True, help='学号')
    ap.add_argument('--root', default='.', help='project root')
    ap.add_argument('--out', default='submit', help='dir to write the package into')
    ap.add_argument('--csv', default='submit.csv', help='path to submission CSV (if any)')
    ap.add_argument('--ckpts', nargs='+', default=None,
                    help='specific ckpts to ship; default = everything under checkpoints/')
    args = ap.parse_args()

    pkg = os.path.join(args.out, f"{args.name}_{args.sid}")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    os.makedirs(pkg, exist_ok=True)
    code_d = os.path.join(pkg, 'code'); os.makedirs(code_d)
    ck_d = os.path.join(pkg, 'checkpoints'); os.makedirs(ck_d)
    pp_d = os.path.join(pkg, 'preprocessed'); os.makedirs(pp_d)

    # code
    for f in SRC:
        p = os.path.join(args.root, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(code_d, f))
            print(f"  + code/{f}")

    # ckpts
    if args.ckpts is None:
        src_ck = os.path.join(args.root, 'checkpoints')
        ckpts = [os.path.join(src_ck, f) for f in os.listdir(src_ck) if f.endswith('.pt')]
    else:
        ckpts = args.ckpts
    for cp in ckpts:
        shutil.copy2(cp, os.path.join(ck_d, os.path.basename(cp)))
        print(f"  + checkpoints/{os.path.basename(cp)}  ({os.path.getsize(cp)/1e6:.1f}MB)")

    # shape names
    sn = os.path.join(args.root, 'preprocessed', 'shape_names.npy')
    if os.path.exists(sn):
        shutil.copy2(sn, os.path.join(pp_d, 'shape_names.npy'))
        print(f"  + preprocessed/shape_names.npy")

    # submission csv
    if os.path.exists(args.csv):
        shutil.copy2(args.csv, os.path.join(pkg, 'submit.csv'))
        with open(args.csv) as f:
            n = sum(1 for _ in f)
        print(f"  + submit.csv  ({n} rows)")
    else:
        print(f"  ! submit.csv not found at {args.csv}, skipping")

    # zip
    zip_path = pkg + '.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for d, _, files in os.walk(pkg):
            for f in files:
                p = os.path.join(d, f)
                zf.write(p, os.path.relpath(p, args.out))
    print(f"[done] {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f}MB)")


if __name__ == '__main__':
    main()
