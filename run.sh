#!/bin/bash
# ============================================================
# 验收脚本 — 一键运行：预处理测试集 + 推理 + 生成 submit.csv
#
# 用法:
#   bash run.sh                               # 完整运行
#   bash run.sh --fast                        # 快速模式 (TTA=1)
#   bash run.sh --ckpt checkpoints/xxx.pt     # 指定 ckpt
# ============================================================
set -e

# ======================== 配置区 ========================
# 测试集根目录（含 modelnet40_test.txt + modelnet40_shape_names.txt + 类子目录）
TEST_ROOT="/data4/ts_project/modelne40"

# 测试集样本列表（在 TEST_ROOT 下）
TEST_LIST="${TEST_ROOT}/modelnet40_test.txt"

# 项目根目录
PROJ="/data4/ts_project/ModelNetProject"

# Checkpoint（best cAcc 最优）
CKPT="${PROJ}/checkpoints/pointmlp_A_s2026_bestcacc.pt"

# 输出
OUT_CSV="${PROJ}/submit.csv"
TTA=10
GPU=0
FAST=false
# ============================================================

# 解析额外参数
for arg in "$@"; do
    case $arg in
        --fast) FAST=true; TTA=1 ;;
        --ckpt) CKPT="$2"; shift ;;
        --tta) TTA="$2"; shift ;;
        --gpu) GPU="$2"; shift ;;
    esac
    shift 2>/dev/null || true
done

echo "============================================"
echo " 赛道一: ModelNet40 验收脚本"
echo "============================================"
echo " TEST_ROOT: ${TEST_ROOT}"
echo " TEST_LIST: ${TEST_LIST}"
echo " CKPT:      ${CKPT}"
echo " TTA:       ${TTA}"
echo " GPU:       ${GPU}"
echo " OUT:       ${OUT_CSV}"
echo "============================================"

# ---------- 校验 ----------
if [ ! -f "${TEST_LIST}" ]; then
    echo "[ERROR] 测试列表不存在: ${TEST_LIST}"
    exit 1
fi
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint 不存在: ${CKPT}"
    echo "  可用 ckpt:"
    ls -la ${PROJ}/checkpoints/*.pt 2>/dev/null || echo "  (无)"
    exit 1
fi
if [ ! -f "${PROJ}/preprocessed/shape_names.npy" ]; then
    echo "[ERROR] shape_names.npy 不存在"
    exit 1
fi
N_SAMPLES=$(wc -l < ${TEST_LIST})
echo " 测试样本数: ${N_SAMPLES}"

# ---------- 推理 ----------
echo ""
echo "[推理] PointMLP + TTA=${TTA}..."

cd ${PROJ}
CUDA_VISIBLE_DEVICES=${GPU} /opt/conda/bin/python predict.py \
    --ckpt ${CKPT} \
    --test_list ${TEST_LIST} \
    --test_root ${TEST_ROOT} \
    --shape_names preprocessed/shape_names.npy \
    --tta ${TTA} \
    --batch_size 32 \
    --out ${OUT_CSV}

N_OUT=$(wc -l < ${OUT_CSV})
echo ""
echo "============================================"
echo " 完成!"
echo " 输出: ${OUT_CSV}  (${N_OUT} 行)"
echo "============================================"

# ---------- 头 5 行预览 ----------
echo ""
echo "前 5 行预览:"
head -5 ${OUT_CSV}
