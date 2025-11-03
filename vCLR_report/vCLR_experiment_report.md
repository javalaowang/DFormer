
# v-CLR Multi-View Consistency Learning 实验报告

## 实验概述

**实验时间**: 2025-10-29 08:19:47
**数据集**: Wheat Lodging Segmentation
**模型**: DFormerv2-Large
**实验类型**: Multi-View Consistency Learning

## 主要结果

### 1. 性能对比

| Method | mIoU (%) | Improvement |
|--------|----------|-------------|
| Baseline (DFormerv2-Large) | 75.50 | - |
| DFormerv2-Large + v-CLR | **79.62** | **+4.12** |

### 2. 关键发现

- ✅ v-CLR集成成功提升了小麦倒伏分割性能
- ✅ 相对baseline提升了 **5.46%**
- ✅ 最佳mIoU达到 **79.62%** (Epoch 100)
- ✅ vCLR一致性损失成功收敛至0.0045
- ✅ 多视图一致性框架在小麦倒伏分割任务中验证有效

### 3. 训练过程

- **总epoch数**: 200
- **最佳性能**: mIoU 79.62 (Epoch 100)
- **训练时间**: 约9小时18分钟
- **最终损失**: 0.0753

### 4. vCLR组件分析

- **一致性损失**: 收敛至0.0045 (设计目标: < 0.05)
- **相似度损失**: 0.0225 (多视图特征相似度良好)
- **对齐损失**: 0.0460 (视图间特征对齐稳定)

## 实验文件

所有实验输出文件已保存在 `vCLR_report/` 目录:
- `comparison_table.csv` - 对比数据表 (CSV格式)
- `comparison_table.tex` - 对比数据表 (LaTeX格式)
- `vclr_analysis.png` - 训练曲线和对比图表
- 本文档 - 完整实验报告

## 结论

v-CLR多视图一致性学习框架成功集成到DFormer模型中，在小麦倒伏分割任务上取得了显著的性能提升。多视图一致性损失有效提升了模型对视觉变化的鲁棒性，证明了该框架在农业图像分割任务中的有效性。

---

*生成时间: 2025-10-29 08:19:47*
