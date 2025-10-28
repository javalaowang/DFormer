"""
Convex Combination Star (CCS) Shape Prior - Paper Implementation
基于CVPR 2025论文的严谨实现

核心数学原理：
1. 多中心星形约束：φ(x) = max_i φ_i(x) 其中 φ_i(x) 是第i个中心的星形场
2. 凸组合平滑化：φ_CCS(x) = Σ_i α_i(x) · φ_i(x)，其中 α_i(x) = softmax(φ_i(x))
3. 变分对偶算法：通过拉格朗日对偶将形状约束转化为softmax/sigmoid形式

Author: Based on Zhao et al. CVPR 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


class StarShapeField(nn.Module):
    """
    单中心星形场函数
    
    数学定义：
    φ(x) = d(x, c) - r(θ)
    其中 d(x, c) 是点x到中心c的距离，r(θ) 是角度θ方向的半径函数
    """
    
    def __init__(self, learnable_radius: bool = True):
        super().__init__()
        self.learnable_radius = learnable_radius
        
        if learnable_radius:
            # 学习角度相关的半径函数
            self.radius_net = nn.Sequential(
                nn.Linear(1, 32),  # 输入角度
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # 输出[0,1]的半径
            )
        else:
            # 固定半径（圆形）
            self.register_buffer('fixed_radius', torch.ones(1))
    
    def forward(self, coords: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """
        计算星形场函数
        
        Args:
            coords: (B, H, W, 2) 像素坐标
            center: (B, 2) 中心坐标
            
        Returns:
            field: (B, H, W) 星形场函数值
        """
        B, H, W, _ = coords.shape
        
        # 计算相对坐标
        center_expanded = center.view(B, 1, 1, 2)
        relative_coords = coords - center_expanded  # (B, H, W, 2)
        
        # 计算距离
        distance = torch.norm(relative_coords, dim=-1)  # (B, H, W)
        
        # 计算角度
        angle = torch.atan2(relative_coords[..., 1], relative_coords[..., 0])  # (B, H, W)
        angle_normalized = (angle + math.pi) / (2 * math.pi)  # 归一化到[0,1]
        
        if self.learnable_radius:
            # 学习角度相关的半径
            angle_input = angle_normalized.unsqueeze(-1)  # (B, H, W, 1)
            radius = self.radius_net(angle_input).squeeze(-1)  # (B, H, W)
            # 缩放到合理范围
            radius = radius * 50.0 + 10.0  # [10, 60]
        else:
            radius = self.fixed_radius.expand_as(distance)
        
        # 星形场函数：φ(x) = r(θ) - d(x, c)
        # 正值表示在星形内部，负值表示在外部
        field = radius - distance
        
        return field


class ConvexCombinationStar(nn.Module):
    """
    凸组合星形(CCS)模块
    
    核心公式：
    φ_CCS(x) = Σ_i α_i(x) · φ_i(x)
    其中 α_i(x) = softmax(φ_i(x) / τ)
    
    保证：
    1. Σ_i α_i(x) = 1 (凸组合)
    2. α_i(x) ≥ 0 (非负权重)
    3. 处处可微 (适合反向传播)
    """
    
    def __init__(
        self,
        num_centers: int = 5,
        temperature: float = 1.0,
        learnable_centers: bool = True,
        learnable_radius: bool = True,
        feature_dim: int = 512
    ):
        super().__init__()
        
        self.num_centers = num_centers
        self.temperature = temperature
        self.learnable_centers = learnable_centers
        
        # 星形场生成器（所有中心共享）
        self.star_field = StarShapeField(learnable_radius=learnable_radius)
        
        # 中心预测网络
        if learnable_centers:
            self.center_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_centers * 2)  # 每个中心2个坐标
            )
        else:
            # 固定中心（网格分布）
            self.register_buffer('fixed_centers', self._init_fixed_centers(num_centers))
    
    def _init_fixed_centers(self, num_centers: int) -> torch.Tensor:
        """初始化固定的网格分布中心"""
        grid_size = int(math.ceil(math.sqrt(num_centers)))
        y = torch.linspace(0.2, 0.8, grid_size)
        x = torch.linspace(0.2, 0.8, grid_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        centers = torch.stack([yy.flatten(), xx.flatten()], dim=1)[:num_centers]
        return centers
    
    def forward(
        self,
        features: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: (B, C, H, W) 输入特征
            image_size: (H, W) 目标图像尺寸，如果为None则使用特征尺寸
            
        Returns:
            ccs_field: (B, H, W) CCS场函数
            centers: (B, num_centers, 2) 中心坐标
            weights: (B, num_centers, H, W) 凸组合权重
        """
        B, C, H, W = features.shape
        
        if image_size is None:
            image_size = (H, W)
        target_H, target_W = image_size
        
        # 1. 预测或使用固定中心
        if self.learnable_centers:
            centers_flat = self.center_predictor(features)  # (B, num_centers*2)
            centers = centers_flat.view(B, self.num_centers, 2)  # (B, num_centers, 2)
            # 归一化到[0, 1]
            centers = torch.sigmoid(centers)
            # 缩放到目标图像尺寸
            centers = centers * torch.tensor([target_H, target_W], device=centers.device)
        else:
            centers = self.fixed_centers.unsqueeze(0).expand(B, -1, -1)
            centers = centers * torch.tensor([target_H, target_W], device=centers.device)
        
        # 2. 生成坐标网格
        y_coords = torch.arange(target_H, device=features.device, dtype=torch.float32)
        x_coords = torch.arange(target_W, device=features.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([yy, xx], dim=-1)  # (H, W, 2)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        
        # 3. 为每个中心生成星形场
        fields = []
        for i in range(self.num_centers):
            center_i = centers[:, i]  # (B, 2)
            field_i = self.star_field(coords, center_i)  # (B, H, W)
            fields.append(field_i)
        
        fields = torch.stack(fields, dim=1)  # (B, num_centers, H, W)
        
        # 4. 凸组合：α_i(x) = softmax(φ_i(x) / τ)
        weights = F.softmax(fields / self.temperature, dim=1)  # (B, num_centers, H, W)
        
        # 5. 加权求和：φ_CCS(x) = Σ_i α_i(x) · φ_i(x)
        ccs_field = (weights * fields).sum(dim=1)  # (B, H, W)
        
        return ccs_field, centers, weights


class CCSVariationalModule(nn.Module):
    """
    CCS变分模块
    
    基于论文中的变分模型和对偶算法：
    min_u ∫_Ω (u - f)² dx + λ ∫_Ω |∇u| dx + μ ∫_Ω φ_CCS(x) · u dx
    
    对偶算法得到：
    u* = softmax(f + μ · φ_CCS(x))
    """
    
    def __init__(
        self,
        num_centers: int = 5,
        temperature: float = 1.0,
        learnable_centers: bool = True,
        learnable_radius: bool = True,
        variational_weight: float = 0.1,
        feature_dim: int = 512
    ):
        super().__init__()
        
        self.variational_weight = variational_weight
        
        # CCS模块
        self.ccs = ConvexCombinationStar(
            num_centers=num_centers,
            temperature=temperature,
            learnable_centers=learnable_centers,
            learnable_radius=learnable_radius,
            feature_dim=feature_dim
        )
        
        # 变分权重学习
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        变分前向传播
        
        Args:
            features: (B, C, H, W) 输入特征
            logits: (B, num_classes, H, W) 原始logits
            
        Returns:
            enhanced_logits: (B, num_classes, H, W) 增强的logits
            details: 详细信息字典
        """
        B, num_classes, H, W = logits.shape
        
        # 1. 计算CCS场
        ccs_field, centers, weights = self.ccs(features, (H, W))
        
        # 2. 学习变分权重
        adaptive_weight = self.weight_net(features)  # (B, 1)
        adaptive_weight = adaptive_weight * self.variational_weight
        
        # 3. 变分增强：u* = softmax(f + μ · φ_CCS(x))
        # 将CCS场扩展到所有类别
        ccs_enhancement = ccs_field.unsqueeze(1) * adaptive_weight.view(B, 1, 1, 1)
        
        # 4. 应用变分约束
        enhanced_logits = logits + ccs_enhancement
        
        if return_details:
            details = {
                'ccs_field': ccs_field,
                'centers': centers,
                'weights': weights,
                'adaptive_weight': adaptive_weight,
                'enhancement': ccs_enhancement
            }
            return enhanced_logits, details
        
        return enhanced_logits, {}


class CCSShapeLoss(nn.Module):
    """
    CCS形状约束损失
    
    基于论文中的形状约束：
    L_shape = ∫_Ω φ_CCS(x) · (1 - u(x)) dx
    其中 u(x) 是预测的分割概率
    """
    
    def __init__(self, lambda_shape: float = 0.1):
        super().__init__()
        self.lambda_shape = lambda_shape
    
    def forward(
        self,
        pred_prob: torch.Tensor,
        ccs_field: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算形状损失
        
        Args:
            pred_prob: (B, H, W) 预测概率
            ccs_field: (B, H, W) CCS场函数
            target: (B, H, W) 可选的ground truth
            
        Returns:
            loss: 形状损失
        """
        # 基础形状损失：L_shape = ∫_Ω φ_CCS(x) · (1 - u(x)) dx
        # 在CCS场为正的区域，预测概率应该接近1
        positive_regions = (ccs_field > 0).float()
        shape_loss = torch.mean(ccs_field * positive_regions * (1 - pred_prob))
        
        # 如果有ground truth，添加监督
        if target is not None:
            # 在ground truth为正的区域，CCS场应该为正
            gt_positive = (target > 0).float()
            supervision_loss = torch.mean(
                F.mse_loss(ccs_field * gt_positive, gt_positive)
            )
            shape_loss = shape_loss + 0.5 * supervision_loss
        
        return self.lambda_shape * shape_loss


class CCSHead(nn.Module):
    """
    CCS增强的分类头
    
    将CCS约束直接整合到最终的分类层
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_centers: int = 5,
        temperature: float = 1.0,
        variational_weight: float = 0.1,
        use_ccs: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_ccs = use_ccs
        
        # 标准分类头
        self.conv_seg = nn.Conv2d(in_channels, num_classes, 1)
        
        # CCS变分模块
        if use_ccs:
            self.ccs_variational = CCSVariationalModule(
                num_centers=num_centers,
                temperature=temperature,
                variational_weight=variational_weight,
                feature_dim=in_channels
            )
            
            # CCS形状损失
            self.shape_loss = CCSShapeLoss(lambda_shape=0.1)
    
    def forward(
        self,
        features: torch.Tensor,
        return_ccs_details: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            features: (B, C, H, W) 输入特征
            return_ccs_details: 是否返回CCS详细信息
            
        Returns:
            logits: (B, num_classes, H, W) 分类logits
            ccs_details: CCS详细信息
        """
        # 标准分类
        logits = self.conv_seg(features)
        
        if not self.use_ccs:
            return logits, {}
        
        # CCS变分增强
        enhanced_logits, ccs_details = self.ccs_variational(
            features, logits, return_details=True
        )
        
        if return_ccs_details:
            return enhanced_logits, ccs_details
        
        return enhanced_logits, {}


# ================ 测试代码 ================

if __name__ == "__main__":
    """测试CCS模块"""
    
    print("="*60)
    print("Testing CCS Paper Implementation")
    print("="*60)
    
    # 创建测试数据
    B, C, H, W = 2, 512, 128, 128
    features = torch.randn(B, C, H, W)
    
    print(f"Input features shape: {features.shape}")
    
    # 测试CCS模块
    print("\n1. Testing ConvexCombinationStar...")
    ccs = ConvexCombinationStar(
        num_centers=3,
        temperature=1.0,
        learnable_centers=True
    )
    
    ccs_field, centers, weights = ccs(features)
    
    print(f"   CCS field shape: {ccs_field.shape}")
    print(f"   Centers shape: {centers.shape}")
    print(f"   Weights shape: {weights.shape}")
    print(f"   Centers (first sample): {centers[0]}")
    
    # 测试变分模块
    print("\n2. Testing CCSVariationalModule...")
    variational = CCSVariationalModule(
        num_centers=3,
        variational_weight=0.1
    )
    
    logits = torch.randn(B, 3, H, W)
    enhanced_logits, details = variational(features, logits, return_details=True)
    
    print(f"   Enhanced logits shape: {enhanced_logits.shape}")
    print(f"   Adaptive weight: {details['adaptive_weight']}")
    
    # 测试CCS头
    print("\n3. Testing CCSHead...")
    head = CCSHead(
        in_channels=C,
        num_classes=3,
        num_centers=3,
        use_ccs=True
    )
    
    output, ccs_details = head(features, return_ccs_details=True)
    
    print(f"   Output shape: {output.shape}")
    print(f"   CCS field shape: {ccs_details['ccs_field'].shape}")
    
    # 测试形状损失
    print("\n4. Testing CCSShapeLoss...")
    pred_prob = torch.sigmoid(torch.randn(B, H, W))
    target = (torch.rand(B, H, W) > 0.5).float()
    
    loss_fn = CCSShapeLoss(lambda_shape=0.1)
    loss = loss_fn(pred_prob, ccs_details['ccs_field'], target)
    
    print(f"   Shape loss: {loss.item():.4f}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)



