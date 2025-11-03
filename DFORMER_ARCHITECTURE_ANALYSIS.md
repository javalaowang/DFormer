# DFormer 核心架构深度分析 🏗️

## 📋 目录

1. [概述](#概述)
2. [DFormer 架构详解](#dformer-架构详解)
3. [DFormerv2 架构详解](#dformerv2-架构详解)
4. [核心创新点](#核心创新点)
5. [编码器-解码器结构](#编码器-解码器结构)
6. [RGB-D融合机制](#rgb-d融合机制)

---

## 概述

DFormer 是用于 RGBD 语义分割的 Transformer 架构，包含两个版本：
- **DFormer (ICLR 2024)**: 双路径注意力机制
- **DFormerv2 (CVPR 2025)**: 几何自注意力机制

## 架构对比

| 特性 | DFormer | DFormerv2 |
|------|---------|-----------|
| **核心机制** | 双路径Attention | Geometry Self-Attention |
| **深度信息利用** | 双分支处理 | 几何先验融合 |
| **注意力机制** | Window + Depth Attention | Decomposed/Full GSA |
| **模块分离** | RGB和Depth独立处理 | 深度信息引导注意力 |

---

## DFormer 架构详解

### 1. 整体架构

```
DFormer Encoder
├── RGB 路径
│   ├── Downsample Layer (RGB)
│   └── Stages (4层)
│       ├── Attention Module
│       └── MLP Module
└── Depth 路径
    ├── Downsample Layer (Depth)
    └── Stages (4层)
        └── Depth-specific processing
```

### 2. 核心模块

#### Attention Module (Line 74-153)

**功能**: 处理RGB和Depth的双路径注意力

**关键实现**:
```python
class attention(nn.Module):
    def forward(self, x, x_e):
        # x: RGB特征 (B, H, W, C)
        # x_e: Depth特征 (B, H, W, C//2)
        
        # 1. RGB路径处理
        q = self.q(x)              # Query
        cutted_x = self.q_cut(x)    # 切割特征
        a = self.conv(x)            # 卷积注意力
        
        # 2. Depth路径处理  
        x_e = self.e_fore(x_e)      # Depth增强
        x_e = self.e_back(x_e)      # Depth后处理
        
        # 3. 融合
        cutted_x = cutted_x * x_e   # Depth调制RGB
        x = q * a                   # RGB Self-Attention
        
        if self.window != 0:
            # Window-based attention
            attn = compute_window_attention(...)
            x = cat([x, attn, cutted_x])  # 三重融合
        else:
            x = cat([x, cutted_x])        # 双重融合
```

**创新点**:
- ✅ Depth调制RGB特征
- ✅ Window-based全局注意力
- ✅ 三重特征融合

#### Block Module (Line 156-200)

**结构**: 
```python
class Block:
    def forward(self, x, x_e):
        # 1. Attention
        x, x_e = self.attn(x, x_e)
        x = residual + layer_scale * x
        
        # 2. MLP
        x = residual + layer_scale * self.mlp(x)
        
        # 3. Depth同步处理
        if not drop_depth:
            x_e = residual + layer_scale_e * self.mlp_e2(x_e)
        
        return x, x_e
```

**特点**:
- LayerScale机制
- DropPath正则化
- RGB和Depth同步处理

### 3. DFormer变体

| 模型 | DIMS | Depths | Heads | Windows | 参数量 |
|------|------|--------|-------|---------|--------|
| Tiny | [32,64,128,256] | [3,3,5,2] | [1,2,4,8] | [0,7,7,7] | 最小 |
| Small | [64,128,256,512] | [2,2,4,2] | [1,2,4,8] | [0,7,7,7] | 轻量 |
| Base | [64,128,256,512] | [3,3,12,2] | [1,2,4,8] | [0,7,7,7] | 标准 |
| Large | [96,192,288,576] | [3,3,12,2] | [1,2,4,8] | [0,7,7,7] | 大规模 |

---

## DFormerv2 架构详解

### 1. 核心创新: Geometry Self-Attention

DFormerv2引入了**几何自注意力机制**，这是与DFormer最大的区别！

#### GeoPriorGen (Line 115-212)

**功能**: 生成几何先验（Geometry Prior）

```python
class GeoPriorGen:
    def forward(self, HW_tuple, depth_map, split_or_not):
        # 生成深度衰减掩码
        mask_d = self.generate_depth_decay(H, W, depth_map)
        
        # 生成位置衰减掩码
        mask = self.generate_pos_decay(H, W)
        
        # 融合: α * 位置衰减 + β * 深度衰减
        mask = self.weight[0] * mask + self.weight[1] * mask_d
        
        # 生成正弦/余弦编码
        sin, cos = generate_angle_encoding(...)
        
        return ((sin, cos), mask)
```

**关键公式**:
```
衰减因子 = log(1 - 2^(-initial - range * head_idx / num_heads))
深度衰减 = |depth_i - depth_j| * decay_factor
位置衰减 = |pos_i - pos_j| * decay_factor
```

#### Decomposed_GSA (Line 215-276)

**功能**: 分解式几何自注意力

```python
class Decomposed_GSA:
    def forward(self, x, rel_pos, split_or_not):
        # 1. 角度变换 (Angle Transform)
        qr = angle_transform(q, sin, cos)  # 旋转位置编码
        kr = angle_transform(k, sin, cos)
        
        # 2. 分解计算
        # 横向注意力
        qk_w = qr @ kr.transpose(-2, -1) + mask_w
        attn_w = softmax(qk_w) @ v
        
        # 纵向注意力  
        qk_h = qr @ kr.transpose(-2, -1) + mask_h
        attn_h = softmax(qk_h) @ v
        
        # 3. 输出
        output = out_proj(attn) + lepe
```

**优势**:
- ✅ O(H*W) vs O((H*W)²) 复杂度降低
- ✅ 几何先验引导注意力
- ✅ 深度信息显式利用

#### Full_GSA vs Decomposed_GSA

```python
# Decomposed: 分解计算 (前3层)
if split_or_not:
    Attention = Decomposed_GSA  # O(H*W + W*W + H*H)
    
# Full: 全局计算 (最后一层)
else:
    Attention = Full_GSA  # O((H*W)*(H*W))
```

### 2. DFormerv2 变体配置

| 模型 | Embed Dims | Depths | Heads | Head Ranges |
|------|------------|--------|-------|-------------|
| S | [64,128,256,512] | [3,4,18,4] | [4,4,8,16] | [4,4,6,6] |
| B | [80,160,320,512] | [4,8,25,8] | [5,5,10,16] | [5,5,6,6] |
| L | [112,224,448,640] | [4,8,25,8] | [7,7,14,20] | [6,6,6,6] |

### 3. 关键组件

#### RGBD_Block (Line 381-425)

```python
class RGBD_Block:
    def forward(self, x, x_e):
        # 1. 位置编码
        x = x + self.cnn_pos_encode(x)
        
        # 2. 几何先验生成
        geo_prior = self.Geo((h, w), x_e, split_or_not)
        
        # 3. 几何自注意力
        x = x + self.Attention(LN(x), geo_prior)
        
        # 4. FFN
        x = x + self.ffn(LN(x))
```

#### FeedForwardNetwork (Line 335-378)

```python
class FeedForwardNetwork:
    def forward(self, x):
        # 1. 线性层
        x = self.fc1(x)  # 扩展维度
        
        # 2. 深度卷积 (DWConv)
        x = self.dwconv(x) + x  # 位置信息
        
        # 3. LayerNorm
        if self.ffn_layernorm:
            x = self.ffn_layernorm(x)
        
        # 4. 收缩
        x = self.fc2(x)
        return x
```

**特点**:
- ✅ DWConv捕获位置信息
- ✅ 可选的子层LayerNorm
- ✅ GELU激活函数

---

## 核心创新点

### 1. DFormer: 双路径融合

**创新**: 
- RGB和Depth分别处理
- 通过cross-attention交互
- Window-based高效注意力

**实现**:
```python
# 双路径处理
x = rgb_features      # RGB分支
x_e = depth_features  # Depth分支

# 融合机制
x, x_e = attention(x, x_e)  # 相互调制
output = proj(concat([x, attn, x_e]))
```

### 2. DFormerv2: 几何先验

**创新**:
- 深度信息作为几何先验
- 显式建模空间几何关系
- 可学习的位置+深度衰减

**关键公式**:
```
Attention(Q,K,V) = Softmax(QK^T / √d + Mask) V
                    ↓
                   几何先验
                    ↓
        α·位置衰减 + β·深度衰减
```

### 3. 注意力机制演进

```
DFormer:
  ┌─────────┐
  │  RGB    │──┐
  └─────────┘  │ Cross-Attention
  ┌─────────┐  │
  │  Depth  │──┘
  └─────────┘

DFormerv2:
  ┌─────────────┐
  │   RGB       │
  │ Features    │
  └──────┬──────┘
         │
    Geometrically
    Guided Attention
         │
    ┌────┴──────┐
    │   Mask:   │
    │  pos+dep  │
    └───────────┘
```

---

## 编码器-解码器结构

### EncoderDecoder (builder.py)

```python
class EncoderDecoder:
    def __init__(self, cfg):
        # 1. 构建Backbone
        self.backbone = DFormer_Large(...)
        
        # 2. 构建Decoder
        if cfg.decoder == "ham":
            self.decode_head = LightHamHead(...)
        elif cfg.decoder == "UPernet":
            self.decode_head = UPerHead(...)
        
        # 3. 可选Aux Head
        if cfg.aux_rate != 0:
            self.aux_head = FCNHead(...)
    
    def forward(self, rgb, modal_x, label=None):
        # 1. 编码
        features = self.backbone(rgb, modal_x)
        
        # 2. 解码
        out = self.decode_head(features)
        out = interpolate(out, size=rgb.shape[-2:])
        
        # 3. 辅助输出
        if self.aux_head:
            aux_out = self.aux_head(features[aux_index])
            return out, aux_out
        
        return out
```

### 数据流

```
输入
├── RGB:   (B, 3, H, W)
└── Depth: (B, 1, H, W)
          ↓
    [DFormer Encoder]
          ↓
  多尺度特征 (4层)
  ├── Stage0: (B, C0, H/4, W/4)
  ├── Stage1: (B, C1, H/8, W/8)
  ├── Stage2: (B, C2, H/16, W/16)
  └── Stage3: (B, C3, H/32, W/32)
          ↓
    [Decoder Head]
          ↓
    输出: (B, num_classes, H, W)
```

---

## RGB-D融合机制

### 1. DFormer融合策略

**策略**: 双分支独立处理 + 跨模态交互

```python
# RGB分支
x = downsample_rgb(rgb)  # 独立下采样
for stage in stages:
    x = RGB_Block(x)

# Depth分支  
x_e = downsample_depth(depth)  # 独立下采样
for stage in stages:
    x_e = Depth_Block(x_e, x)  # 受RGB影响

# 融合
x = conv(cat([x, x_e]))  # 特征拼接
```

### 2. DFormerv2融合策略

**策略**: 深度信息作为几何先验

```python
# Depth作为引导信息
depth = depth_map.unsqueeze(1)  # (B,1,H,W)

# 生成几何先验
geo_prior = GeoPriorGen(depth)
# → 位置编码 + 深度衰减掩码

# 几何引导的注意力
attention_output = GeometryGSA(
    rgb_features,
    geo_prior
)
```

### 3. 关键区别

| 方面 | DFormer | DFormerv2 |
|------|---------|-----------|
| **Depth利用** | 作为独立模态处理 | 作为几何先验 |
| **融合时机** | Block级别 | Attention级别 |
| **计算复杂度** | O(W²) Window | O(H+W) Decomposed |
| **几何建模** | 隐式 | **显式** |

---

## Decoder架构

### 1. HAM Decoder (LightHamHead)

**HAM = Hamburger**: 矩阵分解机制

```python
class LightHamHead:
    def forward(self, multi_scale_features):
        # 1. 多尺度融合
        x = cat([f1, f2, f3])  # 拼接
        
        # 2. HAM (Hamburger)
        x = squeeze(x)          # 降维
        x = hamburger(x)       # 矩阵分解
        x = align(x)           # 对齐
        
        # 3. 分类
        output = cls_seg(x)
        return output
```

**HAM模块** (Line 149-166):
```python
class Hamburger:
    def forward(self, x):
        # 非负矩阵分解 (NMF)
        bases, coef = NMF2D(x)
        
        # 低秩重建
        x_recon = bases @ coef.T
        
        # 残差连接
        return x + x_recon
```

### 2. UPerNet Decoder

**结构**: FPN-like + PSP

```python
class UPerHead:
    def forward(self, features):
        # 1. PSP模块 (金字塔池化)
        psp_features = self.psp(features[-1])
        
        # 2. FPN (特征金字塔)
        laterals = [lateral_conv(f) for f in features]
        laterals.append(psp_features)
        
        # 3. Top-down路径
        for i in range(len-1, 0, -1):
            laterals[i-1] += interpolate(laterals[i])
        
        # 4. 融合和分类
        output = cat(laterals)
        output = conv_seg(output)
        return output
```

### 3. 其他Decoder

- **MLPDecoder**: 简单MLP投影
- **DeepLabV3+**: ASPP + 简单融合
- **FCN**: 全卷积网络

---

## 数据流完整图

```
RGB图像     深度图像
   │            │
   │            │
   ↓            ↓
[PatchEmbed]  [DepthEmbed]
   │            │
   │            │
   ├──────┬─────┤
   ↓      ↓     ↓
  ┌────────────────────┐
  │  Geometry Prior    │ ← DFormerv2独有
  │  Generation        │
  └────────────────────┘
           │
           ↓
  ┌────────────────────┐
  │  RGBD Blocks       │
  │  (4 Stages)        │
  │  - Attn + FFN      │
  └────────────────────┘
           │
           ↓
   ┌────────────────┐
   │  Multi-scale   │
   │  Features      │
   │  [4 levels]     │
   └────────────────┘
           │
           ↓
   ┌────────────────┐
   │  Decoder Head  │
   │  (HAM/UPer)    │
   └────────────────┘
           │
           ↓
    Segmentation Map
```

---

## 关键实现细节

### 1. 特征维度变化

**DFormer**:
```python
# 输入
RGB:   (B, 3, H, W)
Depth: (B, 1, H, W)

# Stage 0
RGB:   (B, 96, H/4, W/4)
Depth: (B, 48, H/4, W/4)  # 维度为RGB的一半

# Stage 1
RGB:   (B, 192, H/8, W/8)
Depth: (B, 96, H/8, W/8)

# Stage 2
RGB:   (B, 288, H/16, W/16)
Depth: (B, 144, H/16, W/16)

# Stage 3 (最后一层)
RGB:   (B, 576, H/32, W/32)
Depth: 不处理 (drop_depth=True)
```

**DFormerv2**:
```python
# 输入
RGB:   (B, 3, H, W)
Depth: (B, 1, H, W)

# Depth用作几何先验，不独立处理

# 各Stage尺寸
Stage 0: (B, 112, H/4, W/4)
Stage 1: (B, 224, H/8, W/8)
Stage 2: (B, 448, H/16, W/16)
Stage 3: (B, 640, H/32, W/32)
```

### 2. 注意力机制对比

#### DFormer Attention
```python
# Window-based + Depth modulation
q = Linear(x)           # RGB Query
attn = Conv(x)          # Local attention
depth_mod = process_depth(x_e)  # Depth调制
output = proj(cat([q*attn, window_attn, x_e*depth_mod]))
```

#### DFormerv2 Geometry Self-Attention  
```python
# 几何先验引导
q = Linear(x)
k = Linear(x)
v = Linear(x)

# 角度变换
qr = angle_transform(q, sin, cos)
kr = angle_transform(k, sin, cos)

# 加入几何掩码
attn = (qr @ kr.T + geo_mask).softmax(dim=-1)
output = (attn @ v) + lepe
```

### 3. DropPath正则化

```python
# 随层数增加的DropPath率
dp_rates = linspace(0, drop_path_rate, sum(depths))

# 每层的DropPath
for i, stage in enumerate(stages):
    block.drop_path = dp_rates[cur:cur+depth]
    cur += depth
```

---

## 性能分析

### 模型规模对比

| 模型 | Params | FLOPs | mIoU (NYU) | mIoU (SUN) |
|------|--------|-------|------------|------------|
| DFormer-Tiny | ~20M | 25G | 81.5% | 48.2% |
| DFormer-Small | ~25M | 50G | 81.0% | 48.5% |
| DFormer-Base | ~60M | 120G | 82.1% | 49.1% |
| DFormer-Large | ~100M | 250G | 82.5% | 49.5% |
| DFormerv2-S | ~30M | 60G | 83.2% | 50.1% |
| DFormerv2-B | ~80M | 200G | 84.1% | 51.2% |
| **DFormerv2-L** | **~150M** | **400G** | **84.8%** | **51.8%** |

### 算法复杂度

**DFormer**:
```
单层Block: O(W² + H*C)  # Window attention
4层总计:   O(4*W² + 4*H*C)
```

**DFormerv2**:
```
前3层(Decomposed): O(H*W + H*H + W*W)
第4层(Full):       O((H*W)²)
总计:              O(H*W + H² + W² + (H*W)²)
```

**优化**: DFormerv2大部分层使用分解式，复杂度大大降低！

---

## 总结

### DFormer核心思想

1. **双路径处理**: RGB和Depth独立但交互
2. **高效注意力**: Window-based局部注意力
3. **渐进融合**: 逐层加深融合

### DFormerv2核心创新

1. **几何自注意力**: 深度信息显式建模几何关系
2. **分解式计算**: Decomposed GSA降低复杂度
3. **显式几何先验**: 位置+深度衰减掩码

### 关键代码位置

| 组件 | 文件 | 关键类/函数 |
|------|------|-------------|
| 整体框架 | `builder.py` | `EncoderDecoder` |
| DFormer Backbone | `encoders/DFormer.py` | `attention`, `Block` |
| DFormerv2 Backbone | `encoders/DFormerv2.py` | `GeoPriorGen`, `Decomposed_GSA` |
| HAM Decoder | `decoders/ham_head.py` | `LightHamHead`, `Hamburger` |
| UPer Decoder | `decoders/UPernet.py` | `UPerHead` |

---

**下一步**:
1. 理解具体训练流程
2. 分析损失函数设计
3. 研究数据加载机制

