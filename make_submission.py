"""
Package Track-1 (ModelNet40) submission.

Per assignment spec the prediction CSV must be named:
    赛道1-姓名1学号1-姓名2学号2-姓名3学号3.csv
The materials zip uses the same prefix so they pair up at submission.

Layout inside <赛道1-...>.zip:
    code/                源码 (train, predict, data_loader, models, preprocess, dump_val, eval_val, make_submission)
    checkpoints/         5 个训练好的 ckpt
    preprocessed/shape_names.npy
    赛道1-...csv         预测文件 (重复一份在 zip 里, 方便交叉验证)
    设计思路.pdf / .tex / .md
    运行说明.md
    README.md

Usage:
    python make_submission.py \\
        --names "张三-2021001,李四-2021002,王五-2021003" \\
        --csv submit.csv

Produces:
    赛道1-张三2021001-李四2021002-王五2021003.csv   (re-named copy of the CSV)
    赛道1-张三2021001-李四2021002-王五2021003.zip   (materials package)
"""
import argparse
import os
import shutil
import zipfile

SRC = ['train.py', 'predict.py', 'data_loader.py', 'models.py', 'preprocess.py',
       'dump_val.py', 'eval_val.py', 'make_submission.py',
       '设计思路.md', '设计思路.tex', '设计思路.pdf',
       '运行说明.md', 'README.md']


def normalize_member(m):
    """'张三-2021001' -> '张三2021001' (drop separator per PDF spec)."""
    return m.replace('-', '').replace(' ', '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', required=True,
                    help='逗号分隔的 "姓名-学号" 列表, 例如 "张三-2021001,李四-2021002,王五-2021003"')
    ap.add_argument('--root', default='.', help='project root')
    ap.add_argument('--out', default='submit', help='dir to write the package into')
    ap.add_argument('--csv', default='submit.csv', help='path to submission CSV (id,class_name 每行一条)')
    ap.add_argument('--ckpts', nargs='+', default=None,
                    help='specific ckpts to ship; default = everything under checkpoints/')
    args = ap.parse_args()

    members = [normalize_member(s) for s in args.names.split(',') if s.strip()]
    base = '赛道1-' + '-'.join(members)
    os.makedirs(args.out, exist_ok=True)

    # 1) renamed copy of the CSV (作业要求的核心交付物)
    named_csv = os.path.join(args.out, f"{base}.csv")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"submit CSV not found: {args.csv}")
    shutil.copy2(args.csv, named_csv)
    with open(args.csv, encoding='utf-8') as f:
        n_rows = sum(1 for _ in f)
    print(f"  + {named_csv}  ({n_rows} rows)")

    # 2) materials package
    pkg = os.path.join(args.out, base)
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    os.makedirs(pkg)
    code_d = os.path.join(pkg, 'code'); os.makedirs(code_d)
    ck_d = os.path.join(pkg, 'checkpoints'); os.makedirs(ck_d)
    pp_d = os.path.join(pkg, 'preprocessed'); os.makedirs(pp_d)

    for f in SRC:
        p = os.path.join(args.root, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(code_d, f))

    if args.ckpts is None:
        src_ck = os.path.join(args.root, 'checkpoints')
        ckpts = [os.path.join(src_ck, f) for f in os.listdir(src_ck) if f.endswith('.pt')]
    else:
        ckpts = args.ckpts
    for cp in ckpts:
        shutil.copy2(cp, os.path.join(ck_d, os.path.basename(cp)))
        print(f"  + checkpoints/{os.path.basename(cp)}  ({os.path.getsize(cp)/1e6:.1f}MB)")

    sn = os.path.join(args.root, 'preprocessed', 'shape_names.npy')
    if os.path.exists(sn):
        shutil.copy2(sn, os.path.join(pp_d, 'shape_names.npy'))

    # 重复一份 CSV 进材料 zip, 防止 CSV 和材料解耦丢失
    shutil.copy2(args.csv, os.path.join(pkg, f"{base}.csv"))

    zip_path = pkg + '.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for d, _, files in os.walk(pkg):
            for f in files:
                p = os.path.join(d, f)
                zf.write(p, os.path.relpath(p, args.out))
    print(f"[done] materials {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f}MB)")
    print(f"       prediction {named_csv}")
    print(f"\n现场提交两个文件: {os.path.basename(named_csv)}  和  {os.path.basename(zip_path)}")


if __name__ == '__main__':
    main()
