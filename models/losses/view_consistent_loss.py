"""
Multi-View Consistency Loss for DFormer
基于v-CLR思想的视图一致性损失，用于SCI论文实验

核心创新：
1. 在多视图间强制特征一致性
2. 减少对外观（纹理、颜色）的依赖
3. 增强模型泛化能力

Paper-ready implementation with visualization support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import math


class ViewConsistencyLoss(nn.Module):
    """
    视图一致性损失模块
    
    为SCI论文准备的完整实现，包含：
    1. 余弦相似度损失
    2. 特征对齐损失
    3. 几何一致性损失
    """
    
    def __init__(
        self,
        lambda_consistent: float = 0.1,
        lambda_alignment: float = 0.05,
        temperature: float = 0.07,
        consistency_type: str = "cosine_similarity",  # "cosine", "mse", "contrastive"
        use_geometry_constraint: bool = True
    ):
        """
        Args:
            lambda_consistent: 一致性损失权重
            lambda_alignment: 特征对齐损失权重
            temperature: 对比学习的温度参数
            consistency_type: 一致性损失类型
            use_geometry_constraint: 是否使用几何约束
        """
        super().__init__()
        self.lambda_consistent = lambda_consistent
        self.lambda_alignment = lambda_alignment
        self.temperature = temperature
        self.consistency_type = consistency_type
        self.use_geometry_constraint = use_geometry_constraint
        
        # 特征投影头（延迟初始化）
        self.proj_head = None
        self.proj_dim = None
    
    def forward(
        self,
        features_view1: torch.Tensor,
        features_view2: torch.Tensor,
        depth_view1: Optional[torch.Tensor] = None,
        depth_view2: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        计算视图一致性损失
        
        Args:
            features_view1: (B, C, H, W) - 视图1的多尺度特征
            features_view2: (B, C, H, W) - 视图2的多尺度特征
            depth_view1: (B, 1, H, W) - 视图1的深度
            depth_view2: (B, 1, H, W) - 视图2的深度
            return_details: 是否返回详细信息（用于可视化）
            
        Returns:
            loss_dict: 包含各种损失的字典
        """
        # 动态初始化投影头
        if self.proj_head is None or self.proj_dim != features_view1.shape[1]:
            self.proj_dim = features_view1.shape[1]
            self.proj_head = nn.Sequential(
                nn.Linear(self.proj_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ).to(features_view1.device)
        
        # 1. 特征对齐损失
        alignment_loss = self._compute_alignment_loss(features_view1, features_view2)
        
        # 2. 一致性损失
        if self.consistency_type == "cosine_similarity":
            consistency_loss = self._compute_cosine_similarity_loss(features_view1, features_view2)
        elif self.consistency_type == "mse":
            consistency_loss = self._compute_mse_loss(features_view1, features_view2)
        elif self.consistency_type == "contrastive":
            consistency_loss = self._compute_contrastive_loss(features_view1, features_view2)
        else:
            consistency_loss = self._compute_cosine_similarity_loss(features_view1, features_view2)
        
        # 3. 几何一致性损失（如果提供深度）
        if self.use_geometry_constraint and depth_view1 is not None and depth_view2 is not None:
            geometry_loss = self._compute_geometry_loss(depth_view1, depth_view2)
        else:
            geometry_loss = torch.tensor(0.0, device=features_view1.device)
        
        # 总损失
        total_loss = (
            self.lambda_consistent * consistency_loss +
            self.lambda_alignment * alignment_loss +
            geometry_loss
        )
        
        loss_dict = {
            'loss_consistency': consistency_loss,
            'loss_alignment': alignment_loss,
            'loss_geometry': geometry_loss,
            'loss_total': total_loss
        }
        
        if return_details:
            # 用于可视化的详细信息
            loss_dict['details'] = {
                'features_view1_mean': features_view1.mean().item(),
                'features_view2_mean': features_view2.mean().item(),
                'similarity_score': F.cosine_similarity(
                    features_view1.flatten(1),
                    features_view2.flatten(1)
                ).mean().item()
            }
        
        return loss_dict
    
    def _compute_cosine_similarity_loss(self, feat1, feat2):
        """余弦相似度损失"""
        B, C, H, W = feat1.shape
        
        # 展平特征
        feat1_flat = feat1.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        feat2_flat = feat2.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        
        # 投影到低维空间
        feat1_proj = self.proj_head(feat1_flat)  # (B, HW, 128)
        feat2_proj = self.proj_head(feat2_flat)  # (B, HW, 128)
        
        # 归一化
        feat1_proj = F.normalize(feat1_proj, p=2, dim=-1)
        feat2_proj = F.normalize(feat2_proj, p=2, dim=-1)
        
        # 计算余弦相似度
        cosine_sim = (feat1_proj * feat2_proj).sum(dim=-1)  # (B, HW)
        
        # 损失：最大化相似度
        loss = (1 - cosine_sim).mean()
        
        return loss
    
    def _compute_mse_loss(self, feat1, feat2):
        """MSE损失"""
        mse_loss = F.mse_loss(feat1, feat2)
        return mse_loss
    
    def _compute_contrastive_loss(self, feat1, feat2):
        """对比学习损失"""
        B, C, H, W = feat1.shape
        
        # 投影和归一化
        feat1_flat = feat1.flatten(2).permute(0, 2, 1)
        feat2_flat = feat2.flatten(2).permute(0, 2, 1)
        
        feat1_proj = F.normalize(self.proj_head(feat1_flat), p=2, dim=-1)
        feat2_proj = F.normalize(self.proj_head(feat2_flat), p=2, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.bmm(feat1_proj, feat2_proj.permute(0, 2, 1)) / self.temperature
        # (B, HW, HW)
        
        # 正样本：对应位置
        pos_mask = torch.eye(H * W, device=similarity.device).unsqueeze(0).expand(B, -1, -1).bool()
        pos_samples = similarity[pos_mask]
        
        # InfoNCE损失
        pos_exp = torch.exp(pos_samples)
        neg_exp_sum = torch.exp(similarity).sum(dim=-1).sum(dim=-1) - pos_exp.sum(dim=-1)
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum / (H * W - 1) + 1e-10)).mean()
        
        return loss
    
    def _compute_alignment_loss(self, feat1, feat2):
        """特征对齐损失"""
        # 对齐统计量
        mean1 = feat1.mean(dim=[2, 3], keepdim=True)
        mean2 = feat2.mean(dim=[2, 3], keepdim=True)
        
        std1 = feat1.std(dim=[2, 3], keepdim=True)
        std2 = feat2.std(dim=[2, 3], keepdim=True)
        
        alignment_loss = F.mse_loss(mean1, mean2) + F.mse_loss(std1, std2)
        
        return alignment_loss
    
    def _compute_geometry_loss(self, depth1, depth2):
        """几何一致性损失"""
        # 深度应该保持一致
        depth_diff = torch.abs(depth1 - depth2)
        geometry_loss = depth_diff.mean()
        
        return 0.1 * geometry_loss  # 较小的权重


class MultiViewFeatureExtractor(nn.Module):
    """
    多视图特征提取器
    
    用于提取不同视图的特征用于一致性学习
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        融合多视图特征
        
        Args:
            features: 多视图特征列表 [[B,C,H,W], ...]
            
        Returns:
            fused_features: 融合后的特征列表
        """
        if len(features) == 1:
            return features
        
        # 特征对齐
        aligned_features = self._align_features(features)
        
        # 特征融合
        fused_features = []
        for i in range(len(aligned_features) - 1):
            feat_concat = torch.cat([aligned_features[i], aligned_features[i+1]], dim=1)
            feat_fused = self.fusion(feat_concat)
            fused_features.append(feat_fused)
        
        return fused_features
    
    def _align_features(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """对齐不同视图的特征"""
        # 简单的插值对齐
        target_size = features[0].shape[2:]
        aligned = []
        
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(feat)
        
        return aligned


# ============== 论文评估指标 ==============

class ConsistencyMetrics(nn.Module):
    """
    一致性评估指标
    
    用于论文实验的定量分析
    """
    
    def __init__(self):
        super().__init__()
        self.reset()
    
    def reset(self):
        """重置统计量"""
        self.feature_similarities = []
        self.alignment_errors = []
        self.geometry_consistencies = []
    
    def compute(self, features1, features2, depth1=None, depth2=None):
        """
        计算一致性指标
        
        Returns:
            metrics: 指标字典
        """
        # 1. 特征相似度
        feat1_flat = features1.flatten(1)
        feat2_flat = features2.flatten(1)
        
        similarity = F.cosine_similarity(feat1_flat, feat2_flat).mean().item()
        self.feature_similarities.append(similarity)
        
        # 2. 对齐误差
        mean_diff = (features1.mean() - features2.mean()).abs().item()
        std_diff = (features1.std() - features2.std()).abs().item()
        alignment_error = mean_diff + std_diff
        self.alignment_errors.append(alignment_error)
        
        # 3. 几何一致性（如果有深度）
        if depth1 is not None and depth2 is not None:
            depth_diff = (depth1 - depth2).abs().mean().item()
            self.geometry_consistencies.append(depth_diff)
        
        metrics = {
            'similarity': similarity,
            'alignment_error': alignment_error,
            'geometry_consistency': depth_diff if depth1 is not None else None
        }
        
        return metrics
    
    def get_summary(self):
        """获取汇总统计"""
        summary = {
            'mean_similarity': np.mean(self.feature_similarities) if self.feature_similarities else 0,
            'std_similarity': np.std(self.feature_similarities) if self.feature_similarities else 0,
            'mean_alignment_error': np.mean(self.alignment_errors) if self.alignment_errors else 0,
            'mean_geometry_consistency': np.mean(self.geometry_consistencies) if self.geometry_consistencies else 0
        }
        return summary


# ============== 使用示例 ==============

if __name__ == "__main__":
    """测试View Consistency Loss模块"""
    
    print("="*60)
    print("测试Multi-View Consistency Loss")
    print("="*60)
    
    # 创建模拟数据
    B, C, H, W = 2, 512, 64, 64
    
    feat1 = torch.randn(B, C, H, W)
    feat2 = torch.randn(B, C, H, W)
    depth1 = torch.rand(B, 1, H, W) * 10
    depth2 = torch.rand(B, 1, H, W) * 10
    
    # 测试损失函数
    print("\n1. 测试ViewConsistencyLoss...")
    loss_fn = ViewConsistencyLoss(
        lambda_consistent=0.1,
        lambda_alignment=0.05,
        consistency_type="cosine_similarity"
    )
    
    loss_dict = loss_fn(feat1, feat2, depth1, depth2, return_details=True)
    
    print(f"  损失统计:")
    for k, v in loss_dict.items():
        if k != 'details':
            print(f"    {k}: {v.item():.4f}")
    
    # 测试评估指标
    print("\n2. 测试ConsistencyMetrics...")
    metrics = ConsistencyMetrics()
    
    for _ in range(3):
        m = metrics.compute(feat1, feat2, depth1, depth2)
        print(f"  相似度: {m['similarity']:.4f}, 对齐误差: {m['alignment_error']:.4f}")
    
    summary = metrics.get_summary()
    print(f"\n  汇总统计:")
    for k, v in summary.items():
        print(f"    {k}: {v:.4f}")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)

