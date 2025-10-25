# CCS Shape Prior - Paper Implementation

基于CVPR 2025论文《Convex Combination Star Shape Prior for Data-driven Image Semantic Segmentation》的严谨实现。

## 📚 论文信息

- **标题**: Convex Combination Star Shape Prior for Data-driven Image Semantic Segmentation
- **作者**: Zhao et al.
- **会议**: CVPR 2025
- **实现**: 严格遵循论文的数学公式

## 🧮 数学原理

### 核心公式

1. **星形场函数**:
   ```
   φ(x) = r(θ) - d(x, c)
   ```
   其中 `d(x, c)` 是点x到中心c的距离，`r(θ)` 是角度θ方向的半径函数

2. **凸组合星形**:
   ```
   φ_CCS(x) = Σ_i α_i(x) · φ_i(x)
   ```
   其中 `α_i(x) = softmax(φ_i(x) / τ)`

3. **变分对偶算法**:
   ```
   u* = softmax(f + μ · φ_CCS(x))
   ```

4. **形状损失**:
   ```
   L_shape = ∫_Ω φ_CCS(x) · (1 - u(x)) dx
   ```

### 数学性质

- **凸组合**: Σ_i α_i(x) = 1, α_i(x) ≥ 0
- **可微性**: 处处可微，适合反向传播
- **形状约束**: 通过变分模型将形状先验嵌入神经网络

## 🏗️ 实现架构

### 核心模块

1. **StarShapeField**: 单中心星形场函数
   - 支持固定半径和学习半径
   - 实现角度相关的半径函数

2. **ConvexCombinationStar**: 凸组合星形模块
   - 多中心星形场生成
   - Softmax凸组合
   - 支持固定和学习中心

3. **CCSVariationalModule**: 变分模块
   - 实现变分对偶算法
   - 自适应权重学习
   - 形状约束集成

4. **CCSShapeLoss**: 形状损失函数
   - 基于论文的损失设计
   - 支持监督和无监督

5. **CCSHead**: CCS增强分类头
   - 将CCS约束整合到分类层
   - 支持消融实验

### 集成设计

- **DFormerWithCCSPaper**: 基于论文的DFormer集成
- **模块化设计**: CCS作为可选插件
- **向后兼容**: 不影响原始DFormer功能

## 🚀 快速开始

### 1. 基础使用

```python
from models.dformer_ccs_paper import DFormerWithCCSPaper
from easydict import EasyDict as edict

# 创建配置
cfg = edict()
cfg.backbone = "DFormer-Base"
cfg.decoder = "ham"
cfg.num_classes = 3
# ... 其他配置

# 创建模型
model = DFormerWithCCSPaper(
    cfg=cfg,
    use_ccs=True,
    ccs_num_centers=5,
    ccs_temperature=1.0,
    ccs_variational_weight=0.1,
    ccs_shape_lambda=0.1
)

# 前向传播
output, ccs_details = model(rgb, depth, return_ccs_details=True)
```

### 2. 训练配置

```bash
# 基础CCS训练
bash train_ccs_paper.sh

# 消融实验
bash train_ccs_paper_ablation.sh
```

### 3. 测试验证

```bash
# 运行测试
python test_ccs_paper_implementation.py
```

## 📊 消融实验设计

### 实验变体

1. **基线对比**
   - `baseline`: 不使用CCS

2. **中心数量影响**
   - `centers_3`: 3个星形中心
   - `centers_5`: 5个星形中心
   - `centers_7`: 7个星形中心

3. **温度参数影响**
   - `temp_0.5`: 温度参数0.5
   - `temp_1.0`: 温度参数1.0
   - `temp_2.0`: 温度参数2.0

4. **变分权重影响**
   - `var_0.05`: 变分权重0.05
   - `var_0.1`: 变分权重0.1
   - `var_0.2`: 变分权重0.2

5. **形状损失权重影响**
   - `shape_0.05`: 形状损失权重0.05
   - `shape_0.1`: 形状损失权重0.1
   - `shape_0.2`: 形状损失权重0.2

6. **学习策略对比**
   - `fixed_centers`: 固定中心位置
   - `learnable_centers`: 学习中心位置
   - `fixed_radius`: 固定半径函数
   - `learnable_radius`: 学习半径函数

### 运行消融实验

```bash
# 运行完整消融实验
bash train_ccs_paper_ablation.sh

# 分析结果
python utils/generate_paper_ablation_summary.py --experiment_root=experiments/paper_ablation
```

## 📁 文件结构

```
DFormer/
├── models/
│   ├── ccs_paper_implementation.py      # CCS核心实现
│   └── dformer_ccs_paper.py            # DFormer集成
├── local_configs/Wheatlodgingdata/
│   ├── DFormerv2_L_CCS_Paper.py        # 基础配置
│   └── DFormerv2_L_CCS_Paper_Ablation.py # 消融配置
├── utils/
│   └── generate_paper_ablation_summary.py # 结果分析
├── train_ccs_paper.sh                  # 训练脚本
├── train_ccs_paper_ablation.sh         # 消融脚本
└── test_ccs_paper_implementation.py     # 测试脚本
```

## 🔧 配置参数

### CCS参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_ccs` | bool | False | 是否启用CCS |
| `ccs_num_centers` | int | 5 | 星形中心数量 |
| `ccs_temperature` | float | 1.0 | Softmax温度参数 |
| `ccs_variational_weight` | float | 0.1 | 变分权重 |
| `ccs_shape_lambda` | float | 0.1 | 形状损失权重 |
| `ccs_learnable_centers` | bool | True | 是否学习中心位置 |
| `ccs_learnable_radius` | bool | True | 是否学习半径函数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 2e-5 | 学习率（预训练模型） |
| `batch_size` | 2 | 批次大小 |
| `nepochs` | 150 | 训练轮数 |
| `drop_path_rate` | 0.1 | Drop path率 |

## 📈 实验结果分析

### 自动分析工具

```bash
python utils/generate_paper_ablation_summary.py --experiment_root=experiments/paper_ablation
```

### 输出文件

- `paper_ablation_summary.csv`: 数值结果汇总
- `paper_ablation_analysis.png`: 可视化图表
- `paper_ablation_report.md`: 论文格式报告

### 关键指标

- **mIoU**: 平均交并比
- **训练时间**: 收敛时间
- **数学性质**: 凸组合、可微性验证
- **参数敏感性**: 各参数对性能的影响

## 🧪 测试验证

### 测试内容

1. **数学性质测试**
   - 凸组合性质验证
   - 可微性测试
   - 星形场性质验证

2. **功能测试**
   - 模块功能完整性
   - 集成测试
   - 消融配置测试

3. **性能测试**
   - 内存使用测试
   - 计算效率测试

### 运行测试

```bash
python test_ccs_paper_implementation.py
```

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

## 🔬 实现特点

### 严格遵循论文

1. **数学公式**: 严格按照论文实现所有数学公式
2. **算法流程**: 遵循论文中的算法描述
3. **参数设置**: 基于论文建议的参数范围

### 实验导向设计

1. **消融实验**: 支持完整的消融实验设计
2. **结果分析**: 自动生成论文级别的分析报告
3. **可视化**: 提供丰富的可视化图表

### 工程化考虑

1. **模块化**: 清晰的模块划分
2. **可扩展**: 易于添加新的变体
3. **文档完整**: 详细的代码文档和使用说明

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个实现。

## 📄 许可证

本项目遵循原始DFormer项目的许可证。

## 🔗 相关链接

- [原始DFormer项目](https://github.com/VCIP-RGBD/DFormer)
- [CVPR 2025 CCS论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Convex_Combination_Star_Shape_Prior_for_Data-driven_Image_Semantic_Segmentation_CVPR_2025_paper.pdf)

---

**注意**: 本实现严格遵循CVPR 2025论文的数学公式，确保理论正确性和实验可重复性。



