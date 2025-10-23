"""
DFormer with CCS Shape Prior
将凸组合星形(CCS)约束集成到DFormer中

使用方法:
    from models.dformer_with_ccs import DFormerWithCCS
    
    model = DFormerWithCCS(
        cfg=config,
        use_ccs=True,
        num_centers=5
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.builder import EncoderDecoder
from models.shape_priors import CCSModule, CCSShapeLoss, CCSSoftmaxHead


class DFormerWithCCS(nn.Module):
    """
    集成CCS形状先验的DFormer
    
    改进点:
    1. 在decoder输出后添加CCS约束
    2. 添加CCS形状损失
    3. 使用CCS引导的分类头
    """
    def __init__(
        self,
        cfg,
        use_ccs: bool = True,
        num_centers: int = 5,
        ccs_lambda: float = 0.1,
        norm_layer=nn.BatchNorm2d,
        criterion=nn.CrossEntropyLoss(reduction="none", ignore_index=255),
        syncbn=False
    ):
        super().__init__()
        
        self.cfg = cfg
        self.use_ccs = use_ccs
        self.num_centers = num_centers
        
        # 原始DFormer模型
        self.dformer = EncoderDecoder(
            cfg=cfg,
            criterion=criterion,
            norm_layer=norm_layer,
            syncbn=syncbn
        )
        
        # CCS模块
        if use_ccs:
            # 在最后一层特征图上应用CCS
            if hasattr(self.dformer, 'channels'):
                last_channel = self.dformer.channels[-1]
            else:
                # 默认值
                if 'Large' in cfg.backbone:
                    last_channel = 576
                elif 'Base' in cfg.backbone or 'Small' in cfg.backbone:
                    last_channel = 512
                else:
                    last_channel = 256
            
            self.ccs_module = CCSModule(
                num_centers=num_centers,
                feature_dim=last_channel,
                learnable_centers=True
            )
            
            self.ccs_loss_fn = CCSShapeLoss(lambda_shape=ccs_lambda)
            
            # CCS增强的分类层
            self.ccs_head = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, cfg.num_classes, 1)
            )
    
    def forward(self, rgb, modal_x=None, label=None):
        """
        前向传播
        
        Args:
            rgb: (B, 3, H, W) - RGB图像
            modal_x: (B, 3, H, W) - Depth图像（3通道）
            label: (B, H, W) - 标签（训练时）
            
        Returns:
            如果训练: loss
            如果推理: output
        """
        B, _, H, W = rgb.shape
        
        # 1. DFormer backbone特征提取
        features = self.dformer.backbone(rgb, modal_x)
        
        # features是一个列表 [f1, f2, f3, f4] 或 ([f1, f2, f3, f4], depth_features)
        if isinstance(features, tuple):
            features = features[0]
        
        # 2. Decoder
        decoder_output = self.dformer.decode_head.forward(features)
        
        # 上采样到原始尺寸
        output = F.interpolate(
            decoder_output, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 3. 如果使用CCS，添加形状约束
        if self.use_ccs:
            # 在最后一层特征上计算CCS
            last_features = features[-1]
            
            # 上采样特征到输出尺寸
            last_features_upsampled = F.interpolate(
                last_features,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            
            # 计算CCS场
            ccs_field, centers = self.ccs_module(
                last_features_upsampled,
                return_centers=True
            )
            
            # CCS引导的分类
            ccs_logits = self.ccs_head(ccs_field.unsqueeze(1))
            
            # 计算当前权重（自适应或固定）
            if self.adaptive_alpha:
                alpha = self._get_adaptive_alpha()
            else:
                alpha = self.ccs_alpha
            
            # 渐进式增强：DFormer主导，CCS辅助
            # output = decoder_output + α * ccs_logits
            # 保证DFormer的预测占主导（α << 1）
            output = output + alpha * ccs_logits
        else:
            ccs_field = None
            centers = None
        
        # 4. 辅助头（如果有）
        if self.dformer.aux_head:
            aux_output = self.dformer.aux_head(features[self.dformer.aux_index])
            aux_output = F.interpolate(
                aux_output, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            aux_output = None
        
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
            if self.use_ccs and ccs_field is not None:
                # 获取预测的概率
                pred_prob = F.softmax(output, dim=1)
                
                # 对于二分类，取正类概率
                if self.cfg.num_classes == 2:
                    pred_mask = pred_prob[:, 1]
                else:
                    # 多分类，取最大概率类
                    pred_mask = pred_prob.max(dim=1)[0]
                
                ccs_loss = self.ccs_loss_fn(pred_mask, ccs_field, label)
                total_loss += ccs_loss
            
            return total_loss
        
        # 推理时返回输出
        if self.use_ccs:
            return output, ccs_field, centers
        else:
            return output
    
    def _get_adaptive_alpha(self):
        """
        自适应调整CCS权重
        
        训练策略:
        - 前20% epochs: α=0 (纯DFormer，建立语义基础)
        - 20-50% epochs: α线性增长 (渐进引入CCS)
        - 50%+ epochs: α=max_alpha (完整CCS约束)
        """
        epoch = self.current_epoch
        total = getattr(self.cfg, 'nepochs', 300)
        
        if epoch < total * 0.2:
            # 阶段1: 纯DFormer
            return 0.0
        elif epoch < total * 0.5:
            # 阶段2: 渐进增加
            progress = (epoch - total * 0.2) / (total * 0.3)
            return self.ccs_alpha * progress
        else:
            # 阶段3: 完整约束
            return self.ccs_alpha
    
    def set_epoch(self, epoch):
        """设置当前epoch（用于自适应权重）"""
        self.current_epoch = epoch


# ================ 使用示例 ================

if __name__ == "__main__":
    """
    测试DFormerWithCCS
    """
    from easydict import EasyDict as edict
    
    print("="*60)
    print("Testing DFormer with CCS")
    print("="*60)
    
    # 创建模拟配置
    cfg = edict()
    cfg.backbone = "DFormer-Base"
    cfg.pretrained_model = None
    cfg.decoder = "ham"
    cfg.decoder_embed_dim = 512
    cfg.num_classes = 3  # 背景、正常、倒伏
    cfg.background = 255
    cfg.drop_path_rate = 0.1
    cfg.bn_eps = 1e-3
    cfg.bn_momentum = 0.1
    cfg.aux_rate = 0.4
    
    # 创建模型
    print("\nCreating model...")
    model = DFormerWithCCS(
        cfg=cfg,
        use_ccs=True,
        num_centers=3,
        ccs_lambda=0.1
    )
    
    print(f"✓ Model created with CCS (num_centers=3)")
    
    # 创建模拟输入
    B, H, W = 2, 480, 640
    rgb = torch.randn(B, 3, H, W)
    depth = torch.randn(B, 3, H, W)
    label = torch.randint(0, 3, (B, H, W))
    
    print(f"\nInput shapes:")
    print(f"  RGB: {rgb.shape}")
    print(f"  Depth: {depth.shape}")
    print(f"  Label: {label.shape}")
    
    # 测试训练模式
    print("\nTesting training mode...")
    model.train()
    
    try:
        loss = model(rgb, depth, label)
        print(f"✓ Training forward pass successful")
        print(f"  Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Training forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试推理模式
    print("\nTesting inference mode...")
    model.eval()
    
    try:
        with torch.no_grad():
            output, ccs_field, centers = model(rgb, depth)
        
        print(f"✓ Inference forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  CCS field shape: {ccs_field.shape}")
        print(f"  Centers shape: {centers.shape}")
        print(f"  Centers (first sample): {centers[0]}")
    except Exception as e:
        print(f"✗ Inference forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✓ All tests completed!")
    print("="*60)

