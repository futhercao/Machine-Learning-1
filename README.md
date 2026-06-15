# 三维点云分类 — PointMLP

3D 点云分类：**PointMLP** (ICLR 2022) + 10× TTA 推理。

## 结果

| 指标 | 训练 val | 测试 test |
| --- | --- | --- |
| Instance Accuracy | **93.48%** | **93.19%** |
| Class-mean Accuracy | **90.61%** | **90.34%** |

> 目标：iAcc > 92%、cAcc > 90% ✅ 双双达标。

---

## 一、目录结构

```
ModelNetProject/
├─ preprocess.py        # 官方 raw .txt → preprocessed/*.npy
├─ data_loader.py       # 数据集 + 增强 + 分层验证集划分
├─ models.py            # PointMLP 模型
├─ train.py             # 单模型训练 (AMP + SGD cosine)
├─ predict.py           # PointMLP + TTA 推理 → submit.csv
├─ eval_val.py          # 在验证集上评测
├─ dump_val.py          # 导出 val 集 id / label 列表
├─ make_submission.py   # 打包提交 zip
├─ checkpoints/
│   └─ pointmlp.pt      # 训练好的 ckpt
├─ preprocessed/
│   └─ shape_names.npy  # 40 类名 (data.npy 由 preprocess.py 重建)
├─ 设计思路.md / .tex / .pdf
└─ 运行说明.md
```

---

## 二、整体流程

```
[官方 raw 数据集]            [测试集]
     ↓                         ↓
preprocess.py                  │
     ↓                         │
preprocessed/                  │
  ├─ data.npy (1.2GB)          │
  ├─ labels.npy                │
  ├─ ids.npy                   │
  └─ shape_names.npy           │
     ↓                         ↓
train.py                  predict.py
     ↓                         ↓
checkpoints/pointmlp.pt → submit.csv
     ↓                         ↓
eval_val.py              make_submission.py
                               ↓
                     <姓名>_<学号>.zip
```

---

## 三、方法概要

**PointMLP** (Ma et al., ICLR 2022) 是一个纯 MLP 架构：

- 输入：xyz 坐标（单点云 1024 点），无需法线
- 4 级 LocalGrouper（FPS 采样 + KNN 选邻域 + Geometric Affine 归一化）
- 前后残差 MLP block + 全局 max-pool
- 参数量 13.27M
- 官方 ModelNet40 结果：OA 94.1%、mAcc 91.5%

训练配方：
- **SGD** momentum=0.9, wd=2e-4
- Cosine LR: 0.1 → 0.005
- Label smoothing = 0.2
- AMP 混合精度
- 增强：各向异性缩放 (0.66–1.5) + 平移 (±0.2)
- TTA 推理：10 轮随机子采样 softmax 平均

---

## 四、训练

从零复现：

```bash
# 1) 预处理：raw .txt → npy
python preprocess.py --root <RAW_DIR> --out_dir preprocessed

# 2) 训练 PointMLP（单卡 RTX A6000，~5 小时 200 epoch）
python train.py --epochs 200 --batch_size 32 --seed 2026
```

训练超参（写在 `train.py` 默认值）：SGD lr=0.1 momentum=0.9 wd=2e-4 + cosine LR + label_smoothing 0.2 + AMP。每 2 个 epoch 在 val 上评测，保存 best iAcc / best cAcc / last 三个 ckpt。

---

## 五、测试

```bash
# npy 测试集
python predict.py --ckpt checkpoints/pointmlp.pt \
    --test_npy <test_data.npy> --ids_npy <test_ids.npy> \
    --shape_names preprocessed/shape_names.npy \
    --tta 10 --out submit.csv

# 应急快速模式（单模无 TTA，CPU 可跑）
python predict.py --fast --ckpt checkpoints/pointmlp.pt \
    --test_npy <test_data.npy> --ids_npy <test_ids.npy> \
    --shape_names preprocessed/shape_names.npy --out submit.csv

# txt 目录
python predict.py --ckpt checkpoints/pointmlp.pt \
    --test_dir <TEST_DIR> --shape_names preprocessed/shape_names.npy --out submit.csv
```

---

## 六、验证自检

```bash
python eval_val.py --ckpt checkpoints/pointmlp.pt --tta 10
```

---

## 七、打包提交

```bash
python make_submission.py --names "姓名-学号" --csv submit.csv
```
