# vCLR配置说明

## 🔍 vclr_enabled 的配置位置

### 1. 在 `utils/train.py` 中的读取逻辑

```python
# 第184行
vclr_enabled = getattr(config, 'use_multi_view_consistency', False)
```

**说明**:
- `vclr_enabled` 变量是在训练脚本中动态读取的
- 它从配置对象 `config` 中读取 `use_multi_view_consistency` 属性
- 如果配置中没有这个属性，默认值为 `False`

---

### 2. 在配置文件中的设置

#### vCLR配置文件（启用vCLR）

**NYUDepth v2 vCLR配置**:
```python
# local_configs/NYUDepthv2/DFormerv2_L_vCLR.py (第41行)
C.use_multi_view_consistency = True  # ✅ 启用vCLR
```

**SUN RGB-D vCLR配置**:
```python
# local_configs/SUNRGBD/DFormerv2_L_vCLR.py (第41行)
C.use_multi_view_consistency = True  # ✅ 启用vCLR
```

**Wheatlodgingdata vCLR配置**:
```python
# local_configs/Wheatlodgingdata/DFormerv2_Large_vCLR.py (第27行)
C.use_multi_view_consistency = True  # ✅ 启用vCLR
```

#### Baseline配置文件（不启用vCLR）

**NYUDepth v2 Baseline配置**:
```python
# local_configs/NYUDepthv2/DFormerv2_L.py
# 没有设置 use_multi_view_consistency，默认为 False
```

---

## 📋 完整配置流程

```
1. 配置文件
   ↓
   C.use_multi_view_consistency = True
   
2. 训练脚本加载配置
   ↓
   config = import_module(config_path)
   
3. train.py 读取配置
   ↓
   vclr_enabled = getattr(config, 'use_multi_view_consistency', False)
   
4. 根据 vclr_enabled 决定行为
   ↓
   if vclr_enabled:
       # 初始化 vCLR 组件
       # 调用 model(..., return_features=True)
   else:
       # 标准训练
       # 调用 model(..., return_features=False)
```

---

## 🎯 如何启用/禁用 vCLR

### 方法1：使用不同的配置文件

#### 启用vCLR（推荐）
```bash
# 使用 vCLR 配置文件
bash train.sh --config local_configs.NYUDepthv2.DFormerv2_L_vCLR
```

#### 禁用vCLR（baseline）
```bash
# 使用 baseline 配置文件
bash train.sh --config local_configs.NYUDepthv2.DFormerv2_L
```

---

### 方法2：直接修改配置文件

#### 在配置文件中设置
```python
# 启用 vCLR
C.use_multi_view_consistency = True
C.consistency_loss_weight = 0.1
C.alignment_loss_weight = 0.05

# 禁用 vCLR（注释掉或设置为 False）
# C.use_multi_view_consistency = False
```

---

## 📝 相关配置文件位置

### vCLR配置文件
```
local_configs/
├── NYUDepthv2/
│   └── DFormerv2_L_vCLR.py          # ✅ use_multi_view_consistency = True
├── SUNRGBD/
│   └── DFormerv2_L_vCLR.py          # ✅ use_multi_view_consistency = True
└── Wheatlodgingdata/
    └── DFormerv2_Large_vCLR.py      # ✅ use_multi_view_consistency = True
```

### Baseline配置文件
```
local_configs/
├── NYUDepthv2/
│   └── DFormerv2_L.py               # ❌ 未设置（默认False）
├── SUNRGBD/
│   └── DFormerv2_L.py               # ❌ 未设置（默认False）
└── Wheatlodgingdata/
    └── DFormerv2_Large.py           # ❌ 未设置（默认False）
```

---

## 🔧 完整的vCLR配置参数

当 `C.use_multi_view_consistency = True` 时，还需要设置以下参数：

```python
"""vCLR Config"""
# 启用多视图一致性学习
C.use_multi_view_consistency = True

# 一致性损失权重
C.consistency_loss_weight = 0.1  # 一致性损失权重

# 对齐损失权重
C.alignment_loss_weight = 0.05   # 对齐损失权重

# 视图生成设置
C.num_views = 2  # 生成的视图数量

# 一致性损失类型: "cosine_similarity", "mse", "contrastive"
C.consistency_type = "cosine_similarity"

# 几何约束（如果有深度信息）
C.use_geometry_constraint = True

# 实验设置
C.experiment_name = "DFormerv2_vCLR"
C.enable_visualization = True
C.save_experiment_results = True
```

---

## ✅ 验证配置是否生效

### 方法1：查看训练日志

当vCLR启用时，训练日志开头会有：
```
============================================================
Initializing v-CLR Multi-View Consistency Learning
============================================================
✓ ViewConsistencyLoss initialized
```

### 方法2：检查训练代码行为

```python
# train.py 第345行
if vclr_enabled:
    # vCLR模式: 会调用 return_features=True
    seg_loss, features = model(imgs, modal_xs, gts, return_features=True)
else:
    # Baseline模式: 不会传 return_features（默认False）
    loss = model(imgs, modal_xs, gts)
```

---

## 🎯 总结

- **配置位置**: `local_configs/*/DFormerv2_*_vCLR.py` 中设置 `C.use_multi_view_consistency = True`
- **读取位置**: `utils/train.py` 第184行读取配置
- **变量名**: `vclr_enabled`（在train.py中）
- **默认值**: `False`（如果配置文件中没有设置）

---

更新日期: 2025-11-03

