#!/bin/bash
# ============================================================
# ModelNet40 测试集完整评测脚本
# 用法: bash eval_test.sh
# ============================================================
set -e

# ======================== 配置区 ========================
# 测试集根目录（含 modelnet40_test.txt + 类子目录）
TEST_ROOT="/data4/ts_project/modelne40"

# 测试集样本列表（在 TEST_ROOT 下）
TEST_LIST="${TEST_ROOT}/modelnet40_test.txt"

# 项目根目录
PROJ="/data4/ts_project/ModelNetProject"

# Checkpoint（三选一，best cAcc 通常最优）
CKPT="${PROJ}/checkpoints/pointmlp_A_s2026_bestcacc.pt"

# 输出路径
OUT_CSV="${PROJ}/submit.csv"

# TTA 轮数（10 为标准，1 为快速验证）
TTA=10

# GPU 设备
GPU=0
# ============================================================

echo "============================================"
echo " ModelNet40 测试集评测"
echo "============================================"
echo " TEST_ROOT: ${TEST_ROOT}"
echo " TEST_LIST: ${TEST_LIST}"
echo " CKPT:      ${CKPT}"
echo " TTA:       ${TTA}"
echo " GPU:       ${GPU}"
echo "============================================"

# 检查必要文件
if [ ! -f "${TEST_LIST}" ]; then
    echo "[ERROR] 测试列表不存在: ${TEST_LIST}"
    exit 1
fi
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint 不存在: ${CKPT}"
    echo "  可用 ckpt:"
    ls -la ${PROJ}/checkpoints/*.pt
    exit 1
fi
if [ ! -f "${PROJ}/preprocessed/shape_names.npy" ]; then
    echo "[ERROR] shape_names.npy 不存在"
    exit 1
fi

echo ""
echo "[1/2] 测试集样本数:"
wc -l ${TEST_LIST}

echo ""
echo "[2/2] 开始推理..."
cd ${PROJ}

CUDA_VISIBLE_DEVICES=${GPU} /opt/conda/bin/python predict.py \
    --ckpt ${CKPT} \
    --test_list ${TEST_LIST} \
    --test_root ${TEST_ROOT} \
    --shape_names preprocessed/shape_names.npy \
    --tta ${TTA} \
    --batch_size 32 \
    --out ${OUT_CSV}

echo ""
echo "============================================"
echo " 评测完成"
echo " 输出: ${OUT_CSV}"
echo " 行数: $(wc -l < ${OUT_CSV})"
echo "============================================"

# 如果测试集有真实标签，自动计算准确率
LABELS="${TEST_ROOT}/modelnet40_test.txt"
if [ -f "${LABELS}" ]; then
    echo ""
    echo "[自动评估] 基于样本 ID 反推标签..."
    /opt/conda/bin/python -c "
import numpy as np

# 读取预测
with open('${OUT_CSV}') as f:
    lines = [l.strip().split(',') for l in f if l.strip()]
preds = {l[0]: l[1] for l in lines}

# 从样本 ID 反推真实标签
correct = 0
total = len(lines)
cls_c = {}
cls_t = {}
for sid, pred in preds.items():
    true_cls = '_'.join(sid.split('_')[:-1])
    cls_t[true_cls] = cls_t.get(true_cls, 0) + 1
    if true_cls not in cls_c:
        cls_c[true_cls] = 0
    if pred == true_cls:
        correct += 1
        cls_c[true_cls] += 1

iAcc = correct / total
cAcc = sum(cls_c[c] / cls_t[c] for c in cls_t) / len(cls_t)

print(f'  Instance Accuracy: {iAcc:.4f} ({correct}/{total})')
print(f'  Class-mean Accuracy: {cAcc:.4f}')

# 最差 10 类
per_class = [(c, cls_c.get(c,0)/cls_t[c], cls_c.get(c,0), cls_t[c]) for c in cls_t]
per_class.sort(key=lambda x: x[1])
print(f'  Worst 10 classes:')
for c, r, cc, ct in per_class[:10]:
    print(f'    {c:20s} R={r:.3f} ({cc}/{ct})')
"
fi
