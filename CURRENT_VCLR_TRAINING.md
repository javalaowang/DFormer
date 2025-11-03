# ✅ v-CLR训练确认

## 确认：当前正在运行vCLR训练

**YES！** 确认当前训练的是 `train_wheatlodging_vclr.sh` 脚本

---

## 📊 训练详情

### 配置信息
- **配置文件**: `local_configs.Wheatlodgingdata.DFormerv2_Large_vCLR`
- **✓ use_multi_view_consistency**: True
- **✓ consistency_loss_weight**: 0.1
- **✓ alignment_loss_weight**: 0.05

### 训练状态
- **当前Epoch**: 16/200
- **训练目录**: `checkpoints/Wheatlodgingdata_DFormerv2_L_vCLR_20251028-210238/`
- **开始时间**: 2024-10-28 21:02
- **预计完成**: 2025-10-29 06:21 (约9小时后)

### GPU状态
- **利用率**: 99%
- **内存**: 19.5GB / 24GB
- **设备**: NVIDIA GeForce RTX 3090

### 训练日志
```
Epoch 16/200 Iter 90/90: loss=0.3005 total_loss=0.3158
Avg train time: 57.79s
Avg eval time: 119.77s
```

---

## 📈 与Baseline对比

### 已有的Baseline结果
- **位置**: `checkpoints/Wheatlodgingdata_DFormerv2_L_pretrained_20251024-225443`
- **Best mIoU**: 78.57

### 当前vCLR训练
- **目标**: 提升至 80.57+ (mIoU +2.0)
- **特点**: 启用多视图一致性学习

---

## 🔍 验证命令

```bash
# 查看训练进程
ps aux | grep "train.py" | grep "vCLR"

# 查看训练日志
tail -f vCLR_training.log

# 查看训练结果
ls -lh checkpoints/Wheatlodgingdata_DFormerv2_L_vCLR_20251028-210238/
```

---

**状态**: ✅ v-CLR训练正常运行  
**当前进度**: Epoch 16/200  
**预计剩余**: 8-9小时  
**配置**: ✅ vCLR enabled

