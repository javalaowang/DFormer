# 小麦倒伏分割专门实验方案
# Wheat Lodging Segmentation Specialized Experiment Guide

## 🌾 实验概述

本实验方案专门针对小麦倒伏分割任务设计，通过系统性的实验验证不同形状先验方法的有效性。

## 📋 实验设计

### **阶段1: 基线对比实验**
**目标**: 建立性能基准，验证形状先验的必要性

| 实验名称 | 形状类型 | 描述 | 预期效果 |
|---------|---------|------|---------|
| baseline | 无 | 原始DFormer，无形状约束 | 性能基准 |
| ccs_star | 星形 | 星形形状先验 | 验证通用形状约束 |
| wheat_bar | 条状 | 条状形状先验（正常小麦） | 验证条状约束 |
| wheat_diffusion | 扩散 | 扩散形状先验（倒伏小麦） | 验证扩散约束 |

### **阶段2: 形状先验类型对比**
**目标**: 深入分析不同形状先验的适用性

#### **条状形状变体**
- `bar_vertical`: 垂直条状（正常小麦）
- `bar_diagonal`: 对角条状（倾斜小麦）
- `bar_learnable`: 学习条状方向

#### **扩散形状变体**
- `diffusion_small`: 小范围扩散
- `diffusion_large`: 大范围扩散
- `diffusion_learnable`: 学习扩散半径

#### **混合形状**
- `mixed_adaptive`: 自适应混合形状

### **阶段3: 参数敏感性分析**
**目标**: 找到最优参数组合

#### **中心数量分析**
- 1, 2, 3, 5, 7个中心

#### **权重参数分析**
- 轻量级: λ=0.01, μ=0.01
- 中等: λ=0.05, μ=0.05
- 强化: λ=0.1, μ=0.1
- 极强: λ=0.2, μ=0.2

#### **温度参数分析**
- 锐利: τ=0.5
- 标准: τ=1.0
- 平滑: τ=2.0

### **阶段4: 混合策略优化**
**目标**: 结合多种策略，达到最佳性能

#### **渐进式训练**
- 阶段1: 纯语义学习（50 epochs）
- 阶段2: 引入形状约束（50 epochs）
- 阶段3: 强化形状约束（50 epochs）

#### **多尺度形状先验**
- 小尺度: 2中心，半径[5,15]
- 中尺度: 3中心，半径[10,30]
- 大尺度: 2中心，半径[20,50]

## 🚀 使用方法

### **1. 快速测试**
```bash
# 查看实验计划（不实际运行）
bash test_wheat_lodging_quick.sh
```

### **2. 运行单个阶段**
```bash
# 运行阶段1：基线对比实验
python experiments/wheat_lodging_experiment.py --stage stage1_baseline

# 运行阶段2：形状类型对比
python experiments/wheat_lodging_experiment.py --stage stage2_shape_types

# 运行阶段3：参数分析
python experiments/wheat_lodging_experiment.py --stage stage3_parameter_analysis

# 运行阶段4：混合优化
python experiments/wheat_lodging_experiment.py --stage stage4_hybrid_optimization
```

### **3. 运行所有实验**
```bash
# 运行完整实验套件
bash run_wheat_lodging_experiments.sh
```

### **4. 干运行模式**
```bash
# 查看实验计划而不实际训练
python experiments/wheat_lodging_experiment.py --stage stage1_baseline --dry-run
```

## 📊 实验结果分析

### **关键指标**
- **mIoU**: 平均交并比
- **训练时间**: 收敛时间
- **形状一致性**: 分割结果的几何合理性
- **边界质量**: 分割边界的平滑度

### **分析工具**
```bash
# 生成分析报告
python experiments/wheat_lodging_experiment.py --stage stage1_baseline
# 报告将自动保存到 experiments/wheat_lodging/stage1_baseline_report.md
```

### **结果文件结构**
```
experiments/wheat_lodging/
├── stage1_baseline/
│   ├── baseline/
│   ├── ccs_star/
│   ├── wheat_bar/
│   └── wheat_diffusion/
├── stage2_shape_types/
├── stage3_parameter_analysis/
├── stage4_hybrid_optimization/
└── results/
    ├── stage1_baseline_results.json
    ├── stage2_shape_types_results.json
    └── ...
```

## 🎯 预期结果

### **阶段1预期**
- 验证形状先验是否对小麦倒伏分割有效
- 确定哪种基础形状类型最适合

### **阶段2预期**
- 找到最适合小麦倒伏的形状先验类型
- 验证条状vs扩散形状的适用性

### **阶段3预期**
- 确定最优参数组合
- 理解参数对性能的影响

### **阶段4预期**
- 达到最佳分割性能
- 验证混合策略的有效性

## 🔧 自定义实验

### **修改实验参数**
编辑 `experiments/wheat_lodging_experiment.py` 中的 `_define_experiments()` 方法：

```python
def _define_experiments(self):
    return {
        "custom_stage": {
            "name": "自定义实验",
            "description": "您的自定义实验描述",
            "experiments": {
                "custom_exp": {
                    "use_shape_prior": True,
                    "shape_type": "custom",
                    "num_centers": 3,
                    "description": "自定义实验配置"
                }
            }
        }
    }
```

### **添加新的形状类型**
1. 在 `models/wheat_lodging_shape_prior.py` 中实现新的形状类
2. 在实验配置中添加对应的参数
3. 更新配置文件生成逻辑

## 📈 实验监控

### **实时监控**
```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 监控训练日志
tail -f experiments/wheat_lodging/*/log_*.log
```

### **结果可视化**
实验完成后，结果将包含：
- 性能对比图表
- 参数敏感性分析
- 形状约束效果可视化
- 分割结果对比

## 🎉 实验完成后的工作

1. **分析结果**: 查看各阶段的实验结果
2. **选择最佳配置**: 基于性能指标选择最优参数
3. **生成论文图表**: 使用结果生成论文级别的图表
4. **撰写实验报告**: 总结实验发现和结论

## 📚 相关文件

- `models/wheat_lodging_shape_prior.py`: 小麦倒伏形状先验实现
- `experiments/wheat_lodging_experiment.py`: 实验管理器
- `run_wheat_lodging_experiments.sh`: 完整实验运行脚本
- `test_wheat_lodging_quick.sh`: 快速测试脚本

---

**注意**: 完整实验可能需要较长时间（数小时到数天），建议先运行快速测试了解实验计划。
