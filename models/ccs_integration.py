"""
CCS Shape Prior Integration for DFormer
论文实验专用：支持消融实验和对比分析

设计原则：
1. 模块化设计：CCS作为可选插件，不影响原始DFormer
2. 消融实验友好：支持多种开关和参数组合
3. 论文实验导向：便于生成对比结果和分析报告

Author: Based on Zhao et al. CVPR 2025
Implementation for DFormer wheat lodging segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.builder import EncoderDecoder
from models.shape_priors.ccs_module import CCSModule, CCSShapeLoss


class CCSIntegrationMixin:
    """
    CCS集成混入类
    提供CCS相关的功能，可以被任何模型继承
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ccs_initialized = False
    
    def init_ccs_module(
        self,
        feature_dim: int,
        num_centers: int = 5,
        learnable_centers: bool = True,
        ccs_lambda: float = 0.1,
        temperature: float = 1.0
    ):
        """初始化CCS模块"""
        self.ccs_module = CCSModule(
            num_centers=num_centers,
            feature_dim=feature_dim,
            learnable_centers=learnable_centers,
            temperature=temperature
        )
        
        self.ccs_loss_fn = CCSShapeLoss(lambda_shape=ccs_lambda)
        self._ccs_initialized = True
        
        # 实验记录
        self.ccs_config = {
            'num_centers': num_centers,
            'learnable_centers': learnable_centers,
            'ccs_lambda': ccs_lambda,
            'temperature': temperature
        }
    
    def apply_ccs_constraint(
        self,
        features: torch.Tensor,
        output: torch.Tensor,
        return_details: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        应用CCS形状约束
        
        Args:
            features: (B, C, H, W) - 特征图
            output: (B, num_classes, H, W) - 原始输出
            return_details: 是否返回详细信息
            
        Returns:
            enhanced_output: CCS增强的输出
            details: 详细信息（如果return_details=True）
        """
        if not self._ccs_initialized:
            return output if not return_details else (output, {})
        
        B, C, H, W = features.shape
        
        # 上采样特征到输出尺寸
        if features.shape[-2:] != output.shape[-2:]:
            features_upsampled = F.interpolate(
                features, size=output.shape[-2:], 
                mode='bilinear', align_corners=False
            )
        else:
            features_upsampled = features
        
        # 计算CCS场
        ccs_field, centers, weights = self.ccs_module(
            features_upsampled,
            return_centers=True,
            return_weights=True
        )
        
        # CCS引导的分类增强
        ccs_enhancement = self._compute_ccs_enhancement(ccs_field, output)
        
        # 组合输出
        enhanced_output = output + ccs_enhancement
        
        if return_details:
            details = {
                'ccs_field': ccs_field,
                'centers': centers,
                'weights': weights,
                'enhancement': ccs_enhancement
            }
            return enhanced_output, details
        
        return enhanced_output
    
    def _compute_ccs_enhancement(
        self, 
        ccs_field: torch.Tensor, 
        output: torch.Tensor
    ) -> torch.Tensor:
        """计算CCS增强项"""
        # 将CCS场转换为与输出相同维度的增强
        ccs_enhanced = ccs_field.unsqueeze(1)  # (B, 1, H, W)
        
        # 简单的线性变换到输出维度
        if not hasattr(self, 'ccs_projection'):
            self.ccs_projection = nn.Conv2d(1, output.shape[1], 1).to(output.device)
        
        enhancement = self.ccs_projection(ccs_enhanced)
        
        # 自适应权重
        alpha = getattr(self, 'ccs_alpha', 0.1)
        return alpha * enhancement
    
    def compute_ccs_loss(
        self,
        pred_output: torch.Tensor,
        ccs_details: Dict,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算CCS形状损失"""
        if not self._ccs_initialized or ccs_details is None:
            return torch.tensor(0.0, device=pred_output.device)
        
        ccs_field = ccs_details['ccs_field']
        
        # 获取预测概率
        pred_prob = F.softmax(pred_output, dim=1)
        
        # 对于小麦倒伏检测，取正类概率
        if pred_output.shape[1] == 3:  # 背景、正常、倒伏
            pred_mask = pred_prob[:, 2]  # 倒伏类别
        else:
            pred_mask = pred_prob.max(dim=1)[0]
        
        return self.ccs_loss_fn(pred_mask, ccs_field, target)


class DFormerWithCCS(EncoderDecoder, CCSIntegrationMixin):
    """
    集成CCS形状先验的DFormer
    
    设计特点：
    1. 继承原始EncoderDecoder，保持兼容性
    2. 通过Mixin添加CCS功能
    3. 支持灵活的开关控制
    """
    
    def __init__(
        self,
        cfg=None,
        criterion=nn.CrossEntropyLoss(reduction="none", ignore_index=255),
        norm_layer=nn.BatchNorm2d,
        syncbn=False,
        # CCS参数
        use_ccs: bool = False,
        ccs_num_centers: int = 5,
        ccs_lambda: float = 0.1,
        ccs_alpha: float = 0.1,
        ccs_learnable_centers: bool = True,
        ccs_temperature: float = 1.0
    ):
        # 初始化原始DFormer
        super().__init__(cfg, criterion, norm_layer, syncbn)
        
        # CCS配置
        self.use_ccs = use_ccs
        self.ccs_alpha = ccs_alpha
        
        # 初始化CCS模块
        if use_ccs:
            # 获取最后一层特征维度
            if hasattr(self, 'channels'):
                feature_dim = self.channels[-1]
            else:
                # 根据backbone确定特征维度
                if 'Large' in cfg.backbone:
                    feature_dim = 576
                elif 'Base' in cfg.backbone or 'Small' in cfg.backbone:
                    feature_dim = 512
                else:
                    feature_dim = 256
            
            self.init_ccs_module(
                feature_dim=feature_dim,
                num_centers=ccs_num_centers,
                learnable_centers=ccs_learnable_centers,
                ccs_lambda=ccs_lambda,
                temperature=ccs_temperature
            )
    
    def forward(self, rgb, modal_x=None, label=None, return_ccs_details=False):
        """
        前向传播
        
        Args:
            rgb: RGB图像
            modal_x: 深度图像
            label: 标签（训练时）
            return_ccs_details: 是否返回CCS详细信息
            
        Returns:
            训练时: loss
            推理时: output 或 (output, ccs_details)
        """
        # 原始DFormer前向传播
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        
        # 应用CCS约束
        ccs_details = None
        if self.use_ccs:
            # 获取最后一层特征
            features = self.backbone(rgb, modal_x)
            if isinstance(features, tuple):
                features = features[0]
            last_features = features[-1]
            
            # 应用CCS约束
            out, ccs_details = self.apply_ccs_constraint(
                last_features, out, return_details=True
            )
        
        # 计算损失（训练时）
        if label is not None:
            # 主损失
            loss = self.criterion(out, label.long())[label.long() != self.cfg.background].mean()
            
            # 辅助损失
            if self.aux_head:
                aux_loss = self.criterion(aux_fm, label.long())[label.long() != self.cfg.background].mean()
                loss += self.aux_rate * aux_loss
            
            # CCS形状损失
            if self.use_ccs and ccs_details is not None:
                ccs_loss = self.compute_ccs_loss(out, ccs_details, label)
                loss += ccs_loss
            
            return loss
        
        # 推理时返回
        if return_ccs_details:
            return out, ccs_details
        return out


# ================ 消融实验配置 ================

class CCSAblationConfig:
    """CCS消融实验配置"""
    
    @staticmethod
    def get_baseline_config():
        """基线配置：不使用CCS"""
        return {
            'use_ccs': False,
            'ccs_num_centers': 0,
            'ccs_lambda': 0.0,
            'ccs_alpha': 0.0
        }
    
    @staticmethod
    def get_ccs_variants():
        """CCS变体配置"""
        return {
            # 不同中心数量
            'ccs_centers_3': {'use_ccs': True, 'ccs_num_centers': 3, 'ccs_lambda': 0.1, 'ccs_alpha': 0.1},
            'ccs_centers_5': {'use_ccs': True, 'ccs_num_centers': 5, 'ccs_lambda': 0.1, 'ccs_alpha': 0.1},
            'ccs_centers_7': {'use_ccs': True, 'ccs_num_centers': 7, 'ccs_lambda': 0.1, 'ccs_alpha': 0.1},
            
            # 不同损失权重
            'ccs_lambda_0.05': {'use_ccs': True, 'ccs_num_centers': 5, 'ccs_lambda': 0.05, 'ccs_alpha': 0.1},
            'ccs_lambda_0.1': {'use_ccs': True, 'ccs_num_centers': 5, 'ccs_lambda': 0.1, 'ccs_alpha': 0.1},
            'ccs_lambda_0.2': {'use_ccs': True, 'ccs_num_centers': 5, 'ccs_lambda': 0.2, 'ccs_alpha': 0.1},
            
            # 不同增强权重
            'ccs_alpha_0.05': {'use_ccs': True, 'ccs_num_centers': 5, 'ccs_lambda': 0.1, 'ccs_alpha': 0.05},
            'ccs_alpha_0.1': {'use_ccs': True, 'ccs_num_centers': 5, 'ccs_lambda': 0.1, 'ccs_alpha': 0.1},
            'ccs_alpha_0.2': {'use_ccs': True, 'ccs_num_centers': 5, 'ccs_lambda': 0.1, 'ccs_alpha': 0.2},
            
            # 固定vs学习中心
            'ccs_fixed_centers': {'use_ccs': True, 'ccs_num_centers': 5, 'ccs_lambda': 0.1, 'ccs_alpha': 0.1, 'ccs_learnable_centers': False},
            'ccs_learnable_centers': {'use_ccs': True, 'ccs_num_centers': 5, 'ccs_lambda': 0.1, 'ccs_alpha': 0.1, 'ccs_learnable_centers': True},
        }
    
    @staticmethod
    def get_component_ablation():
        """组件消融实验"""
        return {
            'baseline': {'use_ccs': False},
            'ccs_field_only': {'use_ccs': True, 'ccs_lambda': 0.0, 'ccs_alpha': 0.1},  # 只用场约束
            'ccs_loss_only': {'use_ccs': True, 'ccs_lambda': 0.1, 'ccs_alpha': 0.0},   # 只用损失约束
            'ccs_full': {'use_ccs': True, 'ccs_lambda': 0.1, 'ccs_alpha': 0.1},        # 完整CCS
        }


# ================ 实验管理工具 ================

class CCSExperimentManager:
    """CCS实验管理器"""
    
    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        self.experiments = {}
    
    def create_experiment_configs(self, experiment_name: str, variants: Dict[str, Dict]):
        """创建实验配置"""
        configs = {}
        
        for variant_name, ccs_params in variants.items():
            config_name = f"{experiment_name}_{variant_name}"
            
            # 读取基础配置
            config_content = self._read_base_config()
            
            # 添加CCS参数
            ccs_config = self._format_ccs_config(ccs_params)
            config_content += ccs_config
            
            # 保存配置
            config_path = f"local_configs/Wheatlodgingdata/{config_name}.py"
            self._save_config(config_path, config_content)
            
            configs[config_name] = config_path
        
        return configs
    
    def _read_base_config(self) -> str:
        """读取基础配置文件"""
        with open(self.base_config_path, 'r') as f:
            return f.read()
    
    def _format_ccs_config(self, ccs_params: Dict) -> str:
        """格式化CCS配置"""
        config_lines = [
            "\n# CCS Shape Prior Configuration",
            f"C.use_ccs = {ccs_params.get('use_ccs', False)}",
            f"C.ccs_num_centers = {ccs_params.get('ccs_num_centers', 5)}",
            f"C.ccs_lambda = {ccs_params.get('ccs_lambda', 0.1)}",
            f"C.ccs_alpha = {ccs_params.get('ccs_alpha', 0.1)}",
            f"C.ccs_learnable_centers = {ccs_params.get('ccs_learnable_centers', True)}",
            f"C.ccs_temperature = {ccs_params.get('ccs_temperature', 1.0)}",
        ]
        return "\n".join(config_lines)
    
    def _save_config(self, path: str, content: str):
        """保存配置文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)


# ================ 使用示例 ================

if __name__ == "__main__":
    """测试CCS集成"""
    
    print("="*60)
    print("Testing CCS Integration for DFormer")
    print("="*60)
    
    from easydict import EasyDict as edict
    
    # 创建测试配置
    cfg = edict()
    cfg.backbone = "DFormer-Base"
    cfg.pretrained_model = None
    cfg.decoder = "ham"
    cfg.decoder_embed_dim = 512
    cfg.num_classes = 3
    cfg.background = 255
    cfg.drop_path_rate = 0.1
    cfg.bn_eps = 1e-3
    cfg.bn_momentum = 0.1
    cfg.aux_rate = 0.4
    
    # 测试基线模型
    print("\n1. Testing Baseline DFormer...")
    baseline_model = DFormerWithCCS(cfg, use_ccs=False)
    
    # 测试CCS增强模型
    print("\n2. Testing DFormer with CCS...")
    ccs_model = DFormerWithCCS(
        cfg, 
        use_ccs=True,
        ccs_num_centers=3,
        ccs_lambda=0.1,
        ccs_alpha=0.1
    )
    
    # 创建测试数据
    B, H, W = 2, 128, 128
    rgb = torch.randn(B, 3, H, W)
    depth = torch.randn(B, 3, H, W)
    label = torch.randint(0, 3, (B, H, W))
    
    print(f"\nInput shapes: RGB={rgb.shape}, Depth={depth.shape}, Label={label.shape}")
    
    # 测试前向传播
    print("\n3. Testing Forward Pass...")
    
    # 基线模型
    baseline_output = baseline_model(rgb, depth)
    print(f"   Baseline output shape: {baseline_output.shape}")
    
    # CCS模型
    ccs_output, ccs_details = ccs_model(rgb, depth, return_ccs_details=True)
    print(f"   CCS output shape: {ccs_output.shape}")
    print(f"   CCS field shape: {ccs_details['ccs_field'].shape}")
    print(f"   CCS centers shape: {ccs_details['centers'].shape}")
    
    # 测试训练模式
    print("\n4. Testing Training Mode...")
    ccs_model.train()
    loss = ccs_model(rgb, depth, label)
    print(f"   Training loss: {loss.item():.4f}")
    
    print("\n" + "="*60)
    print("✓ CCS Integration Test Completed!")
    print("="*60)
