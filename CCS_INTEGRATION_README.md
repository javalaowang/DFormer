# CCS Shape Prior Integration for DFormer

基于CVPR 2025论文《Convex Combination Star Shape Prior for Data-driven Image Semantic Segmentation》的DFormer集成实现。

## 📋 概述

本项目将凸组合星形(CCS)形状先验集成到DFormer中，专门用于小麦倒伏检测任务。设计考虑了论文实验需求，支持完整的消融实验和对比分析。

## 🏗️ 架构设计

### 核心组件

1. **CCSModule**: 凸组合星形模块
   - 多中心星形场生成
   - 平滑场函数控制
   - Softmax凸组合

2. **CCSIntegrationMixin**: 集成混入类
   - 模块化设计
   - 灵活的开关控制
   - 消融实验友好

3. **DFormerWithCCS**: 集成CCS的DFormer
   - 继承原始EncoderDecoder
   - 保持向后兼容性
   - 支持渐进式增强

### 设计原则

- **模块化**: CCS作为可选插件，不影响原始DFormer
- **可配置**: 支持多种参数组合
- **实验导向**: 便于生成论文级别的对比结果

## 🚀 快速开始

### 1. 基础使用

```python
from models.ccs_integration import DFormerWithCCS
from easydict import EasyDict as edict

# 创建配置
cfg = edict()
cfg.backbone = "DFormer-Base"
cfg.decoder = "ham"
cfg.num_classes = 3
# ... 其他配置

# 创建模型
model = DFormerWithCCS(
    cfg=cfg,
    use_ccs=True,
    ccs_num_centers=5,
    ccs_lambda=0.1,
    ccs_alpha=0.1
)

# 前向传播
output, ccs_details = model(rgb, depth, return_ccs_details=True)
```

### 2. 训练配置

使用预定义的配置文件：

```bash
# 基础CCS训练
bash train_ccs.sh

# 消融实验
bash train_ccs_ablation.sh
```

### 3. 消融实验

```python
from models.ccs_integration import CCSAblationConfig

# 获取消融实验配置
variants = CCSAblationConfig.get_ccs_variants()
baseline = CCSAblationConfig.get_baseline_config()
```

## 📊 消融实验设计

### 实验变体

1. **基线对比**
   - `baseline`: 不使用CCS

2. **中心数量影响**
   - `centers_3`: 3个星形中心
   - `centers_5`: 5个星形中心
   - `centers_7`: 7个星形中心

3. **损失权重影响**
   - `lambda_0.05`: 损失权重0.05
   - `lambda_0.1`: 损失权重0.1
   - `lambda_0.2`: 损失权重0.2

4. **增强权重影响**
   - `alpha_0.05`: 增强权重0.05
   - `alpha_0.1`: 增强权重0.1
   - `alpha_0.2`: 增强权重0.2

5. **中心学习策略**
   - `fixed_centers`: 固定中心位置
   - `learnable_centers`: 学习中心位置

### 运行消融实验

```bash
# 运行完整消融实验
bash train_ccs_ablation.sh

# 分析结果
python utils/generate_ablation_summary.py --experiment_root=experiments/ablation_ccs
```

## 📁 文件结构

```
DFormer/
├── models/
│   ├── ccs_integration.py          # CCS集成主模块
│   ├── shape_priors/
│   │   └── ccs_module.py           # CCS核心模块
│   └── dformer_with_ccs.py         # 原始集成实现
├── local_configs/Wheatlodgingdata/
│   ├── DFormer_Base_CCS.py         # 基础CCS配置
│   └── DFormerv2_L_CCS_Ablation.py # 消融实验配置
├── utils/
│   └── generate_ablation_summary.py # 结果分析工具
├── train_ccs.sh                    # CCS训练脚本
├── train_ccs_ablation.sh           # 消融实验脚本
└── test_ccs_integration.py         # 集成测试脚本
```

## 🔧 配置参数

### CCS参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_ccs` | bool | False | 是否启用CCS |
| `ccs_num_centers` | int | 5 | 星形中心数量 |
| `ccs_lambda` | float | 0.1 | 形状损失权重 |
| `ccs_alpha` | float | 0.1 | 增强权重 |
| `ccs_learnable_centers` | bool | True | 是否学习中心位置 |
| `ccs_temperature` | float | 1.0 | Softmax温度参数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 2e-5 | 学习率（预训练模型） |
| `batch_size` | 2 | 批次大小 |
| `nepochs` | 150 | 训练轮数 |
| `drop_path_rate` | 0.1 | Drop path率 |

## 📈 实验结果分析

### 自动分析工具

消融实验完成后，使用分析工具生成论文级别的结果：

```bash
python utils/generate_ablation_summary.py --experiment_root=experiments/ablation_ccs
```

### 输出文件

- `ablation_summary.csv`: 数值结果汇总
- `ablation_analysis.png`: 可视化图表
- `ablation_report.md`: 论文格式报告

### 关键指标

- **mIoU**: 平均交并比
- **训练时间**: 收敛时间
- **内存使用**: GPU内存占用
- **参数数量**: 模型参数量

## 🧪 测试

运行集成测试：

```bash
python test_ccs_integration.py
```

测试包括：
- CCS模块功能测试
- DFormer集成测试
- 消融配置测试
- 内存使用测试

## 📚 论文引用

如果您使用了本实现，请引用原始论文：

```bibtex
@inproceedings{zhao2025convex,
  title={Convex Combination Star Shape Prior for Data-driven Image Semantic Segmentation},
  author={Zhao, Xinyu and Xie, Jun and Liu, Jun and Chen, Shengzhe},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个实现。

## 📄 许可证

本项目遵循原始DFormer项目的许可证。

## 🔗 相关链接

- [原始DFormer项目](https://github.com/VCIP-RGBD/DFormer)
- [CVPR 2025 CCS论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Convex_Combination_Star_Shape_Prior_for_Data-driven_Image_Semantic_Segmentation_CVPR_2025_paper.pdf)

---

**注意**: 本实现专门针对小麦倒伏检测任务优化，但可以轻松适配其他分割任务。
