"""
小麦倒伏形状先验模块
专门针对小麦倒伏分割任务设计的形状约束

设计理念：
1. 条状形状先验：适合正常小麦的垂直条状特征
2. 扩散形状先验：适合倒伏小麦的扩散特征
3. 混合形状先验：结合条状和扩散特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Dict


class BarShapeField(nn.Module):
    """
    条状形状场函数
    适合正常小麦的垂直条状特征
    """
    
    def __init__(self, learnable_orientation: bool = True):
        super().__init__()
        self.learnable_orientation = learnable_orientation
        
        if learnable_orientation:
            # 学习条状方向
            self.orientation_net = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh()  # 输出[-1, 1]的角度
            )
        else:
            self.register_buffer('fixed_orientation', torch.tensor(0.0))
    
    def forward(self, coords: torch.Tensor, center: torch.Tensor, width: float = 10.0) -> torch.Tensor:
        """
        计算条状场函数
        
        Args:
            coords: (B, H, W, 2) 像素坐标
            center: (B, 2) 中心坐标
            width: 条状宽度
            
        Returns:
            field: (B, H, W) 条状场函数值
        """
        B, H, W, _ = coords.shape
        
        # 计算相对坐标
        center_expanded = center.view(B, 1, 1, 2)
        relative_coords = coords - center_expanded  # (B, H, W, 2)
        
        if self.learnable_orientation:
            # 学习条状方向
            orientation = self.orientation_net(relative_coords) * math.pi  # (B, H, W)
        else:
            orientation = self.fixed_orientation
        
        # 计算到条状轴的距离
        # 条状轴方向向量
        axis_direction = torch.stack([
            torch.cos(orientation),
            torch.sin(orientation)
        ], dim=-1)  # (B, H, W, 2)
        
        # 投影到条状轴
        projection = torch.sum(relative_coords * axis_direction, dim=-1)  # (B, H, W)
        
        # 计算垂直距离
        perpendicular_distance = torch.norm(
            relative_coords - projection.unsqueeze(-1) * axis_direction, 
            dim=-1
        )  # (B, H, W)
        
        # 条状场函数：在条状内部为正
        field = width - perpendicular_distance
        
        return field


class DiffusionShapeField(nn.Module):
    """
    扩散形状场函数
    适合倒伏小麦的扩散特征
    """
    
    def __init__(self, learnable_radius: bool = True):
        super().__init__()
        self.learnable_radius = learnable_radius
        
        if learnable_radius:
            # 学习扩散半径
            self.radius_net = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.register_buffer('fixed_radius', torch.ones(1) * 30.0)
    
    def forward(self, coords: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """
        计算扩散场函数
        
        Args:
            coords: (B, H, W, 2) 像素坐标
            center: (B, 2) 中心坐标
            
        Returns:
            field: (B, H, W) 扩散场函数值
        """
        B, H, W, _ = coords.shape
        
        # 计算相对坐标
        center_expanded = center.view(B, 1, 1, 2)
        relative_coords = coords - center_expanded  # (B, H, W, 2)
        
        # 计算距离
        distance = torch.norm(relative_coords, dim=-1)  # (B, H, W)
        
        if self.learnable_radius:
            # 学习扩散半径
            radius = self.radius_net(relative_coords).squeeze(-1) * 50.0 + 10.0  # (B, H, W)
        else:
            radius = self.fixed_radius.expand_as(distance)
        
        # 扩散场函数：距离越近值越大
        field = torch.exp(-distance / (radius + 1e-6))
        
        return field


class WheatLodgingShapePrior(nn.Module):
    """
    小麦倒伏形状先验模块
    结合条状和扩散形状特征
    """
    
    def __init__(
        self,
        num_centers: int = 5,
        use_bar_shape: bool = True,
        use_diffusion_shape: bool = True,
        learnable_centers: bool = True,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.num_centers = num_centers
        self.use_bar_shape = use_bar_shape
        self.use_diffusion_shape = use_diffusion_shape
        self.learnable_centers = learnable_centers
        self.temperature = temperature
        
        # 形状场生成器
        if use_bar_shape:
            self.bar_field = BarShapeField(learnable_orientation=True)
        if use_diffusion_shape:
            self.diffusion_field = DiffusionShapeField(learnable_radius=True)
        
        # 中心预测网络
        if learnable_centers:
            self.center_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_centers * 2)
            )
        else:
            self.register_buffer('fixed_centers', self._init_fixed_centers(num_centers))
        
        # 形状权重学习网络
        self.shape_weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 条状和扩散的权重
            nn.Softmax(dim=-1)
        )
    
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            features: (B, C, H, W) 输入特征
            image_size: (H, W) 目标图像尺寸
            
        Returns:
            combined_field: (B, H, W) 组合形状场
            centers: (B, num_centers, 2) 中心坐标
            weights: (B, num_centers, H, W) 凸组合权重
            details: 详细信息字典
        """
        B, C, H, W = features.shape
        
        if image_size is None:
            image_size = (H, W)
        target_H, target_W = image_size
        
        # 1. 预测中心
        if self.learnable_centers:
            centers_flat = self.center_predictor(features)
            centers = centers_flat.view(B, self.num_centers, 2)
            centers = torch.sigmoid(centers) * torch.tensor([target_H, target_W], device=centers.device)
        else:
            centers = self.fixed_centers.unsqueeze(0).expand(B, -1, -1)
            centers = centers * torch.tensor([target_H, target_W], device=centers.device)
        
        # 2. 生成坐标网格
        y_coords = torch.arange(target_H, device=features.device, dtype=torch.float32)
        x_coords = torch.arange(target_W, device=features.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([yy, xx], dim=-1)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 3. 学习形状权重
        shape_weights = self.shape_weight_net(features)  # (B, 2)
        bar_weight = shape_weights[:, 0:1]  # (B, 1)
        diffusion_weight = shape_weights[:, 1:2]  # (B, 1)
        
        # 4. 为每个中心生成形状场
        all_fields = []
        
        for i in range(self.num_centers):
            center_i = centers[:, i]  # (B, 2)
            center_fields = []
            
            if self.use_bar_shape:
                bar_field = self.bar_field(coords, center_i)  # (B, H, W)
                center_fields.append(bar_field)
            
            if self.use_diffusion_shape:
                diffusion_field = self.diffusion_field(coords, center_i)  # (B, H, W)
                center_fields.append(diffusion_field)
            
            if len(center_fields) == 2:
                # 组合条状和扩散场
                combined_field = bar_weight.view(B, 1, 1) * center_fields[0] + \
                               diffusion_weight.view(B, 1, 1) * center_fields[1]
            else:
                combined_field = center_fields[0]
            
            all_fields.append(combined_field)
        
        all_fields = torch.stack(all_fields, dim=1)  # (B, num_centers, H, W)
        
        # 5. 凸组合
        weights = F.softmax(all_fields / self.temperature, dim=1)
        combined_field = (weights * all_fields).sum(dim=1)
        
        # 详细信息
        details = {
            'bar_weight': bar_weight,
            'diffusion_weight': diffusion_weight,
            'shape_weights': shape_weights,
            'individual_fields': all_fields
        }
        
        return combined_field, centers, weights, details


class WheatLodgingShapeLoss(nn.Module):
    """
    小麦倒伏形状损失
    结合条状和扩散形状约束
    """
    
    def __init__(self, lambda_shape: float = 0.1, lambda_consistency: float = 0.05):
        super().__init__()
        self.lambda_shape = lambda_shape
        self.lambda_consistency = lambda_consistency
    
    def forward(
        self,
        pred_prob: torch.Tensor,
        shape_field: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算形状损失
        
        Args:
            pred_prob: (B, H, W) 预测概率
            shape_field: (B, H, W) 形状场函数
            target: (B, H, W) 可选的ground truth
            
        Returns:
            loss: 形状损失
        """
        # 基础形状损失
        positive_regions = (shape_field > 0).float()
        shape_loss = torch.mean(shape_field * positive_regions * (1 - pred_prob))
        
        # 一致性损失：在形状场为正的区域，预测应该接近1
        consistency_loss = F.mse_loss(
            pred_prob * positive_regions,
            positive_regions
        )
        
        total_loss = self.lambda_shape * shape_loss + self.lambda_consistency * consistency_loss
        
        # 如果有ground truth，添加监督
        if target is not None:
            gt_positive = (target > 0).float()
            supervision_loss = F.mse_loss(shape_field * gt_positive, gt_positive)
            total_loss += 0.5 * supervision_loss
        
        return total_loss


# ================ 测试代码 ================

if __name__ == "__main__":
    """测试小麦倒伏形状先验模块"""
    
    print("="*60)
    print("Testing Wheat Lodging Shape Prior")
    print("="*60)
    
    # 创建测试数据
    B, C, H, W = 2, 512, 128, 128
    features = torch.randn(B, C, H, W)
    
    print(f"Input features shape: {features.shape}")
    
    # 测试小麦倒伏形状先验
    shape_prior = WheatLodgingShapePrior(
        num_centers=3,
        use_bar_shape=True,
        use_diffusion_shape=True,
        learnable_centers=True
    )
    
    shape_field, centers, weights, details = shape_prior(features)
    
    print(f"✓ Shape field shape: {shape_field.shape}")
    print(f"✓ Centers shape: {centers.shape}")
    print(f"✓ Weights shape: {weights.shape}")
    print(f"✓ Bar weight: {details['bar_weight']}")
    print(f"✓ Diffusion weight: {details['diffusion_weight']}")
    
    # 测试形状损失
    pred_prob = torch.sigmoid(torch.randn(B, H, W))
    target = (torch.rand(B, H, W) > 0.5).float()
    
    loss_fn = WheatLodgingShapeLoss()
    loss = loss_fn(pred_prob, shape_field, target)
    
    print(f"✓ Shape loss: {loss.item():.4f}")
    
    print("\n" + "="*60)
    print("✓ Wheat Lodging Shape Prior test completed!")
    print("="*60)
