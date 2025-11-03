# v-CLR训练状态报告

## 🚀 训练已启动

**开始时间**: 2024-10-28 20:54:49  
**配置**: DFormerv2-Large pretrained  
**状态**: ✅ 正在训练  

---

## 📊 当前状态

### 训练信息
- **模型**: DFormerv2-Large
- **数据集**: Wheat Lodging Dataset
- **训练图片**: 357张
- **测试图片**: 153张
- **Batch size**: 4
- **Epochs**: 200
- **Learning rate**: 2e-5

### GPU状态
- **GPU利用率**: 85-90%
- **内存使用**: ~20GB / 24GB
- **设备**: NVIDIA GeForce RTX 3090

### 训练目录
```
checkpoints/Wheatlodgingdata_DFormerv2_L_pretrained_20251028-205449/
├── log_2025_10_28_20_54_49.log
├── tb/ (TensorBoard logs)
└── checkpoint/ (模型checkpoints)
```

---

## 📝 预计训练时间

- **每个epoch**: ~10-15分钟
- **总时长**: 200 epochs × 15分钟 ≈ 50小时
- **实际可能**: 30-40小时（取决于验证）

---

## 🔍 监控命令

### 查看实时日志
```bash
tail -f checkpoints/Wheatlodgingdata_DFormerv2_L_pretrained_20251028-205449/log_2025_10_28_20_54_49.log
```

### 监控GPU
```bash
watch -n 1 nvidia-smi
```

### 检查训练进度
```bash
bash monitor_training.sh
```

### TensorBoard
```bash
tensorboard --logdir=checkpoints/Wheatlodgingdata_DFormerv2_L_pretrained_20251028-205449/tb
```

---

## 📈 训练配置详情

### 优化器
- **类型**: AdamW
- **Learning rate**: 2e-5
- **Weight decay**: 0.01
- **Momentum**: 0.9

### 数据增强
- **Multi-scale**: [0.75, 1, 1.25]
- **Random flip**: Yes
- **Random crop**: 500x500

### 评估
- **验证间隔**: 每25次迭代
- **保存最佳模型**: Yes

---

## ⏳ 预计里程碑

1. **Epoch 1**: 约20:55 (已开始)
2. **Epoch 5**: 约21:10
3. **Epoch 10**: 约21:40
4. **Epoch 25**: 约23:00
5. **Epoch 50**: 约次日05:00
6. **Epoch 100**: 约次日18:00
7. **Epoch 200**: 约第3天12:00

---

## 📊 重要信息

### 预训练模型
```
/root/DFormer/checkpoints/pretrained/DFormerv2_Large_pretrained.pth (359MB)
```

### 数据路径
```
RGB: datasets/Wheatlodgingdata/RGB/
HHA: datasets/Wheatlodgingdata/HHA/
Label: datasets/Wheatlodgingdata/Label/
```

---

## ✅ 下一步

训练完成后将自动：
1. 保存最佳模型
2. 生成评估结果
3. 生成可视化图表

---

**当前状态**: ⏳ 正在训练  
**最后更新**: 2024-10-28 20:55  
**预计完成**: 2-3天后

