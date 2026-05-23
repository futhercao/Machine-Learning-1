# ModelNet40 Point Cloud Classification

赛道一 提交代码。三种基线（PointNet++ SSG / MSG、DGCNN）多种子训练 + Test-Time Augmentation 集成。

## 结果
| 指标 | 验证集 | 阈值 | 加分 |
| --- | --- | --- | --- |
| Instance Accuracy | **96.27%** | 92% | +5 |
| Class-mean Accuracy | **93.46%** | 90% | +5 |

5 个模型（`pointnet2_ssg ×2 seed` + `pointnet2_msg` + `dgcnn ×2 seed`）softmax 平均 + 10× TTA。

## 仓库结构
```
ModelNetProject/
├─ preprocess.py        # 把官方 raw_resampled 处理成 npy
├─ data_loader.py       # ModelNet40Dataset + 增强 + stratified split
├─ models.py            # PointNet2SSG / PointNet2MSG / DGCNN
├─ train.py             # 单模训练
├─ predict.py           # 多模型 + TTA 集成推理 → CSV
├─ eval_val.py          # 在内部验证集上评估集成精度
├─ make_submission.py   # 打包 <姓名>_<学号>.zip
├─ train_gpu0.sh / train_gpu0b.sh   # 双流并行训练队列
├─ checkpoints/         # 5 个训练好的 ckpt (33MB)
├─ preprocessed/shape_names.npy     # 40 类名(其余 npy 见 .gitignore)
├─ 设计思路.md
└─ 运行说明.md
```

## 快速复现
完整流程参考 `运行说明.md`。最短：
```bash
python predict.py --ckpts checkpoints/*.pt --test_npy <test_data.npy> \
                  --ids_npy <test_ids.npy> --tta 10 --out submit.csv
```
