# 配置文件更新总结

## ✅ 已完成的配置更新

### 1. 预训练模型路径修正

**问题**: 配置文件使用了通用的预训练模型路径，但实际每个数据集都有专用的预训练模型。

**更新内容**:

#### NYUDepth v2
- **Baseline**: `local_configs/NYUDepthv2/DFormerv2_L.py`
  - ✅ 更新为: `checkpoints/pretrained/NYUDepthv2/NYUv2_DFormer_Large.pth`
  
- **vCLR**: `local_configs/NYUDepthv2/DFormerv2_L_vCLR.py`
  - ✅ 更新为: `checkpoints/pretrained/NYUDepthv2/NYUv2_DFormer_Large.pth`

#### SUN RGB-D
- **Baseline**: `local_configs/SUNRGBD/DFormerv2_L.py`
  - ✅ 更新为: `checkpoints/pretrained/SUNRGBD/SUNRGBD_DFormer_Large.pth`
  
- **vCLR**: `local_configs/SUNRGBD/DFormerv2_L_vCLR.py`
  - ✅ 更新为: `checkpoints/pretrained/SUNRGBD/SUNRGBD_DFormer_Large.pth`

---

### 2. GPU内存优化

**问题**: 原始配置的batch size太大，导致CUDA OOM错误。

**解决方案**:

#### NYUDepth v2
- **Baseline** (`DFormerv2_L.py`):
  - ✅ Batch size: `12` → `6`
  - ✅ 启用AMP混合精度训练 (在训练脚本中)
  
- **vCLR** (`DFormerv2_L_vCLR.py`):
  - ✅ Batch size: `12` → `6`
  - ✅ 启用AMP混合精度训练 (在训练脚本中)

#### SUN RGB-D
- **Baseline** (`DFormerv2_L.py`):
  - ⚠️  Batch size: `16` (保持不变，如遇OOM再调整)
  
- **vCLR** (`DFormerv2_L_vCLR.py`):
  - ✅ Batch size: `16` → `8`
  - ✅ 启用AMP混合精度训练 (在训练脚本中)

---

### 3. 训练脚本优化

**文件**: `train_nyu_baseline.sh`

**更新**:
- ✅ 从 `--no-amp` 改为 `--amp` (启用混合精度训练)
- ✅ 使用正确的预训练模型路径
- ✅ Batch size已通过配置文件调整

---

## 📋 配置文件清单

### NYUDepth v2
```
✅ local_configs/NYUDepthv2/DFormerv2_L.py        (Baseline)
✅ local_configs/NYUDepthv2/DFormerv2_L_vCLR.py   (vCLR)
```

### SUN RGB-D
```
✅ local_configs/SUNRGBD/DFormerv2_L.py        (Baseline)
✅ local_configs/SUNRGBD/DFormerv2_L_vCLR.py   (vCLR)
```

---

## 🚀 训练命令

### NYUDepth v2 Baseline
```bash
cd /root/DFormer
bash train_nyu_baseline.sh
```

### NYUDepth v2 with vCLR
```bash
cd /root/DFormer
bash train.sh --config local_configs.NYUDepthv2.DFormerv2_L_vCLR \
    --gpus=1 --syncbn --mst --amp --val_amp
```

### SUN RGB-D Baseline
```bash
cd /root/DFormer
bash train.sh --config local_configs.SUNRGBD.DFormerv2_L \
    --gpus=1 --syncbn --mst --amp --val_amp
```

### SUN RGB-D with vCLR
```bash
cd /root/DFormer
bash train.sh --config local_configs.SUNRGBD.DFormerv2_L_vCLR \
    --gpus=1 --syncbn --mst --amp --val_amp
```

---

## 📊 预训练模型位置

### NYUDepth v2 预训练模型
```
checkpoints/pretrained/NYUDepthv2/
├── NYUv2_DFormer_Large.pth  (448MB) ✅
├── NYUv2_DFormer_Base.pth   (339MB)
├── NYUv2_DFormer_Small.pth  (215MB)
└── NYUv2_DFormer_Tiny.pth   (70MB)
```

### SUN RGB-D 预训练模型
```
checkpoints/pretrained/SUNRGBD/
├── SUNRGBD_DFormer_Large.pth  (448MB) ✅
├── SUNRGBD_DFormer_Base.pth   (339MB)
├── SUNRGBD_DFormer_Small.pth  (215MB)
└── SUNRGBD_DFormer_Tiny.pth   (70MB)
```

---

## ✅ 验证清单

- [x] NYUDepth v2 baseline配置文件使用正确的预训练模型
- [x] NYUDepth v2 vCLR配置文件使用正确的预训练模型
- [x] SUN RGB-D baseline配置文件使用正确的预训练模型
- [x] SUN RGB-D vCLR配置文件使用正确的预训练模型
- [x] Batch size已优化以避免OOM
- [x] AMP混合精度训练已启用
- [x] 训练脚本已更新

---

## 📝 注意事项

1. **Batch Size**: 如果仍然遇到OOM错误，可以进一步减小batch size
2. **AMP**: 混合精度训练可以减少内存占用，但可能略微影响精度
3. **预训练模型**: 使用数据集专用的预训练模型可以获得更好的初始性能
4. **监控**: 使用 `monitor_training.sh` 脚本监控训练进度

---

## 🔄 当前训练状态

✅ **NYUDepth v2 Baseline训练已启动**
- 进程ID: 可运行 `ps aux | grep train.py` 查看
- 日志位置: `checkpoints/NYUDepthv2_DFormerv2_L_YYYYMMDD-HHMMSS/log_*.log`
- 监控命令: `bash monitor_training.sh NYUDepthv2 baseline`

---

更新日期: 2025-11-03

