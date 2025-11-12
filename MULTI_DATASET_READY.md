# 多数据集实验准备完成 ✅

## 📊 数据集状态

### ✅ 所有数据集已准备完成！

| 数据集 | 训练集 | 测试集 | RGB图像 | 深度图 | 标注图 | 状态 |
|--------|--------|--------|---------|--------|--------|------|
| **Wheatlodgingdata** | 357 | 153 | ✅ | ✅ | ✅ | ✅ 已准备 |
| **NYUDepth v2** | 795 | 654 | 1449 | 1449 | 1449 | ✅ 已准备 |
| **SUN RGB-D** | 5285 | 5050 | 10335 | 10335 | 10335 | ✅ 已准备 |

---

## 🚀 开始训练

### 训练命令

#### 1. NYUDepth v2 数据集

##### Baseline (无vCLR)
```bash
cd /root/DFormer
bash train.sh --config local_configs.NYUDepthv2.DFormerv2_L \
    --gpus=1 --syncbn --mst --val_amp
```

##### with vCLR
```bash
cd /root/DFormer
bash train.sh --config local_configs.NYUDepthv2.DFormerv2_L_vCLR \
    --gpus=1 --syncbn --mst --val_amp
```

#### 2. SUN RGB-D 数据集

##### Baseline (无vCLR)
```bash
cd /root/DFormer
bash train.sh --config local_configs.SUNRGBD.DFormerv2_L \
    --gpus=1 --syncbn --mst --val_amp
```

##### with vCLR
```bash
cd /root/DFormer
bash train.sh --config local_configs.SUNRGBD.DFormerv2_L_vCLR \
    --gpus=1 --syncbn --mst --val_amp
```

---

## 📋 实验计划

### 建议的训练顺序

#### 阶段1：Baseline训练（2-3周）

1. **NYUDepth v2 Baseline**
   ```bash
   bash train.sh --config local_configs.NYUDepthv2.DFormerv2_L
   ```
   - 预计训练时间：~2-3天
   - 记录最终mIoU

2. **SUN RGB-D Baseline**
   ```bash
   bash train.sh --config local_configs.SUNRGBD.DFormerv2_L
   ```
   - 预计训练时间：~3-5天
   - 记录最终mIoU

#### 阶段2：vCLR训练（2-3周）

3. **NYUDepth v2 with vCLR**
   ```bash
   bash train.sh --config local_configs.NYUDepthv2.DFormerv2_L_vCLR
   ```
   - 预计训练时间：~2-3天
   - 记录最终mIoU

4. **SUN RGB-D with vCLR**
   ```bash
   bash train.sh --config local_configs.SUNRGBD.DFormerv2_L_vCLR
   ```
   - 预计训练时间：~3-5天
   - 记录最终mIoU

---

## 📊 预期结果表格

训练完成后，收集结果生成对比表格：

| Dataset | Method | mIoU | Improvement | Notes |
|---------|--------|------|-------------|-------|
| NYUDepth v2 | Baseline | ? | - | 待训练 |
| NYUDepth v2 | vCLR | ? | ? | 待训练 |
| SUN RGB-D | Baseline | ? | - | 待训练 |
| SUN RGB-D | vCLR | ? | ? | 待训练 |
| Wheatlodging | Baseline | 78.57% | - | ✅ 已完成 |
| Wheatlodging | vCLR | 79.62% | +1.05% | ✅ 已完成 |

---

## 🎯 训练监控

### 查看训练日志

```bash
# 实时查看最新日志
tail -f checkpoints/*/log_last.log

# 查看特定实验的日志
tail -f checkpoints/NYUDepthv2_DFormerv2_L_vCLR_*/log_last.log
```

### 检查训练进度

```bash
# 查看所有实验的checkpoint目录
ls -lh checkpoints/*/checkpoint/

# 查看最新保存的模型
find checkpoints/ -name "*.pth" -type f -mtime -1 | head -5
```

---

## ⚙️ 配置说明

### NYUDepth v2配置
- **图像尺寸**: 480×640
- **类别数**: 40类
- **Batch size**: 12
- **学习率**: 6e-5
- **训练轮数**: 500 epochs

### SUN RGB-D配置
- **图像尺寸**: 480×480
- **类别数**: 37类
- **Batch size**: 16
- **学习率**: 8e-5
- **训练轮数**: 300 epochs

---

## 📝 训练注意事项

### GPU内存
- NYUDepth v2: batch_size=12，需要足够GPU内存
- SUN RGB-D: batch_size=16，需要更多GPU内存
- 如果OOM，可以减小batch_size

### 训练时间
- NYUDepth v2: ~2-3天（795训练样本）
- SUN RGB-D: ~3-5天（5285训练样本）
- 建议使用nohup在后台训练

### 后台训练示例
```bash
# NYUDepth v2 baseline
nohup bash train.sh --config local_configs.NYUDepthv2.DFormerv2_L \
    --gpus=1 --syncbn --mst --val_amp > nyu_baseline.log 2>&1 &

# 查看进程
ps aux | grep train
```

---

## ✅ 数据集验证结果

### NYUDepth v2
- ✅ 目录结构正确
- ✅ RGB图像: 1449张
- ✅ 深度图: 1449张
- ✅ 标注图: 1449张
- ✅ 训练集: 795张
- ✅ 测试集: 654张

### SUN RGB-D
- ✅ 目录结构正确
- ✅ RGB图像: 10335张
- ✅ 深度图: 10335张
- ✅ 标注图: 10335张（注意文件夹名是labels，小写）
- ✅ 训练集: 5285张
- ✅ 测试集: 5050张

---

## 🎉 准备完成！

所有数据集和配置文件已准备就绪，可以开始多数据集实验了！

**下一步建议**：
1. 先运行NYUDepth v2的baseline训练
2. 然后运行vCLR版本
3. 收集结果并生成对比表格

需要帮助监控训练或提取结果吗？

