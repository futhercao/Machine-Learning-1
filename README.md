# ModelNet40 Point Cloud Classification（赛道一）

3D 点云分类：5 个模型（PointNet++ SSG ×2 seed + PointNet++ MSG + DGCNN ×2 seed）做 softmax 平均 + 10× TTA 集成。

## 结果（验证集 966 个样本）
| 指标 | 取值 | 基础门槛 | 追加门槛 |
| --- | --- | --- | --- |
| Instance Accuracy | **96.27%** | 90% | 92% (+5) |
| Class-mean Accuracy | **93.46%** | 85% | 90% (+5) |

5 个模型 softmax 平均 + 10× TTA。

---

## 一、目录结构

```
ModelNetProject/
├─ preprocess.py        # 把官方 raw .txt → preprocessed/*.npy
├─ data_loader.py       # 数据集 + 增强 + 分层验证集划分
├─ models.py            # PointNet2SSG / PointNet2MSG / DGCNN
├─ train.py             # 单模型训练
├─ predict.py           # 多模型 + TTA 集成推理 → submit.csv
├─ eval_val.py          # 在内部验证集上算集成精度（自检）
├─ dump_val.py          # 导出 val 集 id / label 列表（自检）
├─ make_submission.py   # 打包提交 zip
├─ train_gpu0.sh / train_gpu0b.sh   # 双流并行训练
├─ checkpoints/         # 5 个 ckpt (已附, 33MB)
├─ preprocessed/
│   └─ shape_names.npy  # 40 类名 (data.npy 由 preprocess.py 重建, 1.2GB 故未纳入 git)
├─ 设计思路.pdf / .tex
└─ 运行说明.md
```

---

## 二、整体流程

```
[官方 raw 数据集]            [测试集 (评测时下发)]
     ↓                              ↓
preprocess.py                 
     ↓                              ↓
preprocessed/                   ┌──────────────────────────────┐
  ├─ data.npy (1.2GB)           │                              │
  ├─ labels.npy                 │ 已训练好的 5 个 ckpt          │
  ├─ ids.npy                    │ checkpoints/*.pt              │
  └─ shape_names.npy            │                              │
     ↓                          └──────────────┬───────────────┘
train.py × 5                                   ↓
     ↓                                    predict.py
checkpoints/*.pt ←──────────────────→  submit.csv (id,class_name)
     ↓                                         ↓
eval_val.py (集成自检)               make_submission.py
                                              ↓
                                  <姓名>_<学号>.zip (最终提交)
```

---

## 三、训练流程（已完成，附 ckpt，无需重跑）

如要从零复现，需要先有官方 `ModelNet40_Train`（含 `modelnet40_train.txt` 与各类别子目录），然后：

```bash
# 1) 预处理：raw .txt → npy（耗时 5-10 分钟）
python preprocess.py --raw <RAW_DIR> --out preprocessed

# 2) 训练 5 个 ckpt（单卡约 7-8 小时；并行可半之）
python train.py --model pointnet2_ssg --seed 2026 --epochs 200 --batch_size 32
python train.py --model pointnet2_ssg --seed 42   --epochs 200 --batch_size 32
python train.py --model pointnet2_msg --seed 2026 --epochs 200 --batch_size 24
python train.py --model dgcnn         --seed 2026 --epochs 200 --batch_size 24
python train.py --model dgcnn         --seed 42   --epochs 200 --batch_size 24
```

训练超参（写在 `train.py` 默认值）：AdamW + warmup-cosine（warmup 2%）+ label_smoothing 0.1，patience=40 早停。每个 ckpt 落盘的字段：`{state_dict, args, val_iacc, val_cacc, epoch}`，可以用 `torch.load(...)` 直接看。

---

## 四、验证集 & 如何自检

**val 集**
`data_loader.stratified_split(labels, val_ratio=0.1, seed=2026)`：对每个类别独立按 9:1 分层取，**完全由 seed 决定**，所有训练脚本和评估脚本用同一种子，因此 val 集是固定可复现的 966 个样本。

**两种自检方法：**

```bash
# A) 直接看集成精度（需要 preprocessed/data.npy）
python eval_val.py --ckpts checkpoints/*.pt --tta 10
# 期望输出: [ensemble] iAcc=0.9627  cAcc=0.9346  (930/966)

# B) 导出 val 集的具体 (id, label, class_name)（不需要 data.npy，只要 labels/ids/shape_names）
python dump_val.py
# 生成 preprocessed/val_ids.npy, val_labels.npy, val_summary.csv (人工可读)
```

> 注：`preprocessed/data.npy`（1.2GB）由于体积原因没放进 git，需要先跑 `preprocess.py` 生成。如果只想看精度数字，可参考 `设计思路.pdf` 里的表格，或运行 method A。

---

## 五、测试集到达后的步骤（最常用）

> 给一个 `test_data.npy`，形状 `(M, 10000, 6)` 或 `(M, P>=1024, 6)`，可能同时附 `test_ids.npy`（`(M,)` 字符串数组，每行是样本 id）。

```bash
# 1) 生成 submit.csv（id,class_name 每行一条，无表头）
python predict.py \
    --ckpts checkpoints/pointnet2_ssg_s2026.pt \
            checkpoints/pointnet2_ssg_s42.pt \
            checkpoints/pointnet2_msg_s2026.pt \
            checkpoints/dgcnn_s2026.pt \
            checkpoints/dgcnn_s42.pt \
    --test_npy <PATH_TO_test_data.npy> \
    --ids_npy <PATH_TO_test_ids.npy>  \
    --shape_names preprocessed/shape_names.npy \
    --tta 10 --out submit.csv

# 2) 打包成提交 zip
python make_submission.py --names "姓名1-学号1,姓名2-学号2,姓名3-学号3" --csv submit.csv
# 生成两个文件交付:
#   submit/赛道1-姓名1学号1-姓名2学号2-姓名3学号3.csv     ← 作业要求的预测 csv
#   submit/赛道1-姓名1学号1-姓名2学号2-姓名3学号3.zip     ← 材料(code+ckpt+docs+csv 副本)
```

### 现场时间紧的应急模式

集成 + TTA 在 CPU 上比较慢（约 10–20 分钟 / 250 样本）。若现场限时紧、只有 CPU，加 `--fast` 直接退化为单模无 TTA（~30 秒 / 250 样本），单模 iAcc 仍能稳过 90% 基础门槛：

```bash
python predict.py --fast --ckpts checkpoints/pointnet2_msg_s2026.pt \
    --test_npy <test_data.npy> --ids_npy <test_ids.npy> \
    --shape_names preprocessed/shape_names.npy --out submit.csv
# 注: 把单模性能最强的 pointnet2_msg_s2026.pt 放第一位
```

**其它可能的测试集形式**（`predict.py` 都支持）：

```bash
# 形式 B: 给一个目录, 里面散落 .txt 文件 (每个文件 = 一个样本)
python predict.py --ckpts checkpoints/*.pt --test_dir <TEST_DIR> \
    --shape_names preprocessed/shape_names.npy --out submit.csv

# 形式 C: 给一个 id 列表 + 资源根目录
python predict.py --ckpts checkpoints/*.pt --test_list test.txt --test_root <RAW> \
    --shape_names preprocessed/shape_names.npy --out submit.csv
```

`predict.py` 在 CPU/GPU 都可跑（默认 `--device cuda`，无 GPU 自动 fallback CPU）。GPU 上一次推理 ~30 秒，CPU 上一次约 30 分钟。

---

## 六、submit.csv 格式样例

```
sample_0001,airplane
sample_0002,chair
sample_0003,table
...
```
没有表头，每行 `id,class_name`，class_name 来自 `preprocessed/shape_names.npy`（40 个英文小写名）。

---

## 七、关键设计点
详见 `设计思路.pdf`。简要：
- 三类互补基线（SSG / MSG / DGCNN）保证集成多样性
- 训练增强：随机绕 Z 轴旋转 + 各向异性缩放 + 平移 + 抖动 + 点 dropout（法线随旋转矩阵同步变换）
- 推理：TTA 对每样本随机抽 10 次 1024 点，softmax 累加后 argmax
- 单模 cAcc 全部 < 90%，但 5 模集成 + TTA 把 cAcc 拉到 93.46%
