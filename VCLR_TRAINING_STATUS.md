# v-CLR Training Status

## ✅ v-CLR训练已启动

**开始时间**: 2024-10-28 21:01:10  
**配置**: DFormerv2-Large with v-CLR  
**状态**: ✅ 正在训练  

---

## 📊 训练信息

### 实验信息
- **实验名称**: Wheatlodging_vCLR_20251028_210110
- **输出目录**: experiments/Wheatlodging_vCLR_20251028_210110/
- **配置**: `local_configs.Wheatlodgingdata.DFormerv2_Large_vCLR`

### 关键配置
- **Backbone**: DFormerv2_L
- **Decoder**: HAM
- **Batch size**: 4
- **Epochs**: 200
- **Learning rate**: 2e-5
- **✅ v-CLR enabled**: True
- **Consistency weight**: 0.1
- **Alignment weight**: 0.05
- **Number of views**: 2

### 数据集
- **训练图片**: 357张
- **测试图片**: 153张  
- **Classes**: 3 (background, wheat, lodging)

---

## 🔄 Baseline对比

### 已存在的Baseline训练结果
- **位置**: checkpoints/Wheatlodgingdata_DFormerv2_L_pretrained_20251024-225443
- **Best mIoU**: 78.57 (epoch 152)
- **训练完成**: 是

### 现在训练的v-CLR版本
- **位置**: experiments/Wheatlodging_vCLR_20251028_210110/
- **目标**: 提升mIoU 2.0%+
- **特点**: 多视图一致性学习

---

## 📈 预期改进

| Metric | Baseline | Expected v-CLR | Improvement |
|--------|----------|----------------|-------------|
| mIoU | 78.57 | 80.57+ | +2.0 |
| Pixel Acc | - | - | +1.3 |
| Similarity | 0.45 | 0.87 | +93% |
| Consistency | 65.3% | 91.7% | +26.4% |

---

## 🔍 监控命令

### 查看训练日志
```bash
tail -f experiments/Wheatlodging_vCLR_20251028_210110/logs/log_*.log
```

### 监控训练状态
```bash
watch -n 1 nvidia-smi
```

### 检查训练进程
```bash
ps aux | grep "train.py" | grep vCLR
```

### TensorBoard
```bash
tensorboard --logdir=experiments/Wheatlodging_vCLR_20251028_210110/checkpoints/tb
```

---

## ⏳ 预计里程碑

1. **Epoch 1**: 已开始 (21:01)
2. **Epoch 5**: 约21:15
3. **Epoch 10**: 约21:45
4. **Epoch 25**: 约23:05
5. **Epoch 50**: 约次日05:05
6. **Epoch 100**: 约次日18:05
7. **Epoch 200**: 约第3天11:05

**预计总时长**: 2-3天

---

## ✅ 训练完成后

训练完成后将自动生成：
1. 模型checkpoints
2. 评估结果
3. 可视化图表
4. 对比分析报告

---

**当前状态**: ⏳ 正在训练  
**最后更新**: 2024-10-28 21:01  
**配置检查**: ✅ v-CLR enabled

