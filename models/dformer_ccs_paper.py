"""
DFormer with CCS Shape Prior - Paper Implementation
基于CVPR 2025论文的严谨实现

集成策略：
1. 在decoder输出后应用CCS变分约束
2. 使用论文中的对偶算法形式
3. 保持与原始DFormer的兼容性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.builder import EncoderDecoder
from models.ccs_paper_implementation import CCSHead, CCSShapeLoss


class DFormerWithCCSPaper(nn.Module):
    """
    基于论文的DFormer-CCS集成
    
    设计原则：
    1. 严格遵循论文的数学公式
    2. 保持与原始DFormer的兼容性
    3. 支持消融实验
    """
    
    def __init__(
        self,
        cfg,
        criterion=nn.CrossEntropyLoss(reduction="none", ignore_index=255),
        norm_layer=nn.BatchNorm2d,
        syncbn=False,
        # CCS参数
        use_ccs: bool = False,
        ccs_num_centers: int = 5,
        ccs_temperature: float = 1.0,
        ccs_variational_weight: float = 0.1,
        ccs_shape_lambda: float = 0.1,
        ccs_learnable_centers: bool = True,
        ccs_learnable_radius: bool = True
    ):
        super().__init__()
        
        self.cfg = cfg
        self.use_ccs = use_ccs
        
        # 原始DFormer模型
        self.dformer = EncoderDecoder(
            cfg=cfg,
            criterion=criterion,
            norm_layer=norm_layer,
            syncbn=syncbn
        )
        
        # CCS增强
        if use_ccs:
            # 获取最后一层特征维度
            if hasattr(self.dformer, 'channels'):
                feature_dim = self.dformer.channels[-1]
            else:
                # 根据backbone确定特征维度
                if 'Large' in cfg.backbone:
                    feature_dim = 576
                elif 'Base' in cfg.backbone or 'Small' in cfg.backbone:
                    feature_dim = 512
                else:
                    feature_dim = 256
            
            # CCS增强的分类头
            self.ccs_head = CCSHead(
                in_channels=feature_dim,
                num_classes=cfg.num_classes,
                num_centers=ccs_num_centers,
                temperature=ccs_temperature,
                variational_weight=ccs_variational_weight,
                use_ccs=True
            )
            
            # CCS形状损失
            self.ccs_shape_loss = CCSShapeLoss(lambda_shape=ccs_shape_lambda)
            
            # 记录CCS配置
            self.ccs_config = {
                'num_centers': ccs_num_centers,
                'temperature': ccs_temperature,
                'variational_weight': ccs_variational_weight,
                'shape_lambda': ccs_shape_lambda,
                'learnable_centers': ccs_learnable_centers,
                'learnable_radius': ccs_learnable_radius
            }
    
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
        B, _, H, W = rgb.shape
        
        # 1. DFormer backbone特征提取
        features = self.dformer.backbone(rgb, modal_x)
        
        # features是一个列表 [f1, f2, f3, f4] 或 ([f1, f2, f3, f4], depth_features)
        if isinstance(features, tuple):
            features = features[0]
        
        # 2. 原始decoder
        decoder_output = self.dformer.decode_head.forward(features)
        
        # 上采样到原始尺寸
        output = F.interpolate(
            decoder_output, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 3. CCS增强（如果启用）
        ccs_details = {}
        if self.use_ccs:
            # 在最后一层特征上应用CCS
            last_features = features[-1]
            
            # 上采样特征到输出尺寸
            last_features_upsampled = F.interpolate(
                last_features,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            
            # CCS变分增强
            enhanced_output, ccs_details = self.ccs_head(
                last_features_upsampled,
                return_ccs_details=True
            )
            
            # 组合输出（论文中的对偶算法形式）
            # 使用较小的权重，保持DFormer的主导地位
            alpha = 0.1  # 可调参数
            output = output + alpha * (enhanced_output - output)
        
        # 4. 辅助头（如果有）
        aux_output = None
        if self.dformer.aux_head:
            aux_output = self.dformer.aux_head(features[self.dformer.aux_index])
            aux_output = F.interpolate(
                aux_output, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 5. 计算损失（训练时）
        if label is not None:
            # 主损失
            main_loss = self.dformer.criterion(output, label.long())
            main_loss = main_loss[label.long() != self.cfg.background].mean()
            
            total_loss = main_loss
            
            # 辅助损失
            if aux_output is not None:
                aux_loss = self.dformer.criterion(aux_output, label.long())
                aux_loss = aux_loss[label.long() != self.cfg.background].mean()
                total_loss += self.dformer.aux_rate * aux_loss
            
            # CCS形状损失
            if self.use_ccs and ccs_details:
                # 获取预测概率
                pred_prob = F.softmax(output, dim=1)
                
                # 对于小麦倒伏检测，取正类概率
                if output.shape[1] == 3:  # 背景、正常、倒伏
                    pred_mask = pred_prob[:, 2]  # 倒伏类别
                else:
                    pred_mask = pred_prob.max(dim=1)[0]
                
                ccs_field = ccs_details['ccs_field']
                ccs_loss = self.ccs_shape_loss(pred_mask, ccs_field, label)
                total_loss += ccs_loss
            
            return total_loss
        
        # 推理时返回
        if return_ccs_details:
            return output, ccs_details
        return output
    
    def get_ccs_config(self):
        """获取CCS配置"""
        return self.ccs_config if self.use_ccs else {}


# ================ 消融实验配置 ================

class CCSAblationConfigPaper:
    """基于论文的CCS消融实验配置"""
    
    @staticmethod
    def get_baseline_config():
        """基线配置：不使用CCS"""
        return {
            'use_ccs': False,
            'ccs_num_centers': 0,
            'ccs_temperature': 1.0,
            'ccs_variational_weight': 0.0,
            'ccs_shape_lambda': 0.0
        }
    
    @staticmethod
    def get_ccs_variants():
        """CCS变体配置"""
        return {
            # 不同中心数量
            'centers_3': {
                'use_ccs': True, 'ccs_num_centers': 3, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.1
            },
            'centers_5': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.1
            },
            'centers_7': {
                'use_ccs': True, 'ccs_num_centers': 7, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.1
            },
            
            # 不同温度参数
            'temp_0.5': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 0.5,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.1
            },
            'temp_1.0': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.1
            },
            'temp_2.0': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 2.0,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.1
            },
            
            # 不同变分权重
            'var_0.05': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.05, 'ccs_shape_lambda': 0.1
            },
            'var_0.1': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.1
            },
            'var_0.2': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.2, 'ccs_shape_lambda': 0.1
            },
            
            # 不同形状损失权重
            'shape_0.05': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.05
            },
            'shape_0.1': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.1
            },
            'shape_0.2': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.2
            },
            
            # 中心学习策略
            'fixed_centers': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.1,
                'ccs_learnable_centers': False
            },
            'learnable_centers': {
                'use_ccs': True, 'ccs_num_centers': 5, 'ccs_temperature': 1.0,
                'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.1,
                'ccs_learnable_centers': True
            },
        }
    
    @staticmethod
    def get_component_ablation():
        """组件消融实验"""
        return {
            'baseline': {'use_ccs': False},
            'ccs_field_only': {
                'use_ccs': True, 'ccs_variational_weight': 0.0, 'ccs_shape_lambda': 0.1
            },
            'variational_only': {
                'use_ccs': True, 'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.0
            },
            'ccs_full': {
                'use_ccs': True, 'ccs_variational_weight': 0.1, 'ccs_shape_lambda': 0.1
            },
        }


# ================ 使用示例 ================

if __name__ == "__main__":
    """测试DFormerWithCCSPaper"""
    
    from easydict import EasyDict as edict
    
    print("="*60)
    print("Testing DFormer with CCS Paper Implementation")
    print("="*60)
    
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
    
    # 创建模型
    print("\nCreating model...")
    model = DFormerWithCCSPaper(
        cfg=cfg,
        use_ccs=True,
        ccs_num_centers=3,
        ccs_temperature=1.0,
        ccs_variational_weight=0.1,
        ccs_shape_lambda=0.1
    )
    
    print(f"✓ Model created with CCS")
    print(f"  CCS config: {model.get_ccs_config()}")
    
    # 创建测试数据
    B, H, W = 2, 128, 128
    rgb = torch.randn(B, 3, H, W)
    depth = torch.randn(B, 3, H, W)
    label = torch.randint(0, 3, (B, H, W))
    
    print(f"\nInput shapes:")
    print(f"  RGB: {rgb.shape}")
    print(f"  Depth: {depth.shape}")
    print(f"  Label: {label.shape}")
    
    # 测试推理模式
    print("\nTesting inference mode...")
    model.eval()
    
    with torch.no_grad():
        output, ccs_details = model(rgb, depth, return_ccs_details=True)
    
    print(f"✓ Inference successful")
    print(f"  Output shape: {output.shape}")
    print(f"  CCS field shape: {ccs_details['ccs_field'].shape}")
    print(f"  Centers shape: {ccs_details['centers'].shape}")
    
    # 测试训练模式
    print("\nTesting training mode...")
    model.train()
    
    try:
        loss = model(rgb, depth, label)
        print(f"✓ Training successful")
        print(f"  Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✓ All tests completed!")
    print("="*60)



