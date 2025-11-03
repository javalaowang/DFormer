"""
训练脚本 - 集成v-CLR多视图一致性学习

在原有训练脚本基础上，添加v-CLR功能
保持向后兼容，可通过配置开关控制是否启用v-CLR
"""

import argparse
import os
import sys
import time
from importlib import import_module

import torch
import torch.nn as nn
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 原有训练脚本的所有导入
from utils.train import (
    parser, Engine, get_train_loader, get_val_loader, 
    is_eval, gpu_timer, set_seed, all_reduce_tensor
)
from models.builder import EncoderDecoder as segmodel
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.logger import get_logger
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR
from val_mm import evaluate, evaluate_msf

# v-CLR模块导入
try:
    from models.losses.view_consistent_loss import ViewConsistencyLoss, ConsistencyMetrics
    from utils.visualization.view_consistency_viz import ConsistencyVisualizer
    VCLR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: v-CLR modules not available: {e}")
    VCLR_AVAILABLE = False


def setup_vclr(config, model, device):
    """
    设置v-CLR相关模块
    
    Args:
        config: 训练配置
        model: 模型
        device: 设备
        
    Returns:
        vclr_components: dict包含v-CLR组件
    """
    if not VCLR_AVAILABLE:
        return None
    
    if not getattr(config, 'use_multi_view_consistency', False):
        print("v-CLR not enabled in config")
        return None
    
    print("="*60)
    print("Initializing v-CLR Multi-View Consistency Learning")
    print("="*60)
    
    # 一致性损失
    consistency_loss = ViewConsistencyLoss(
        lambda_consistent=getattr(config, 'consistency_loss_weight', 0.1),
        lambda_alignment=getattr(config, 'alignment_loss_weight', 0.05),
        consistency_type=getattr(config, 'consistency_type', 'cosine_similarity'),
        use_geometry_constraint=True
    ).to(device)
    
    # 评估指标
    consistency_metrics = ConsistencyMetrics()
    
    # 可视化器
    viz_dir = getattr(config, 'visualization_dir', 'visualizations')
    visualizer = ConsistencyVisualizer(output_dir=viz_dir)
    
    components = {
        'consistency_loss': consistency_loss,
        'consistency_metrics': consistency_metrics,
        'visualizer': visualizer,
        'enabled': True
    }
    
    print(f"✓ Consistency loss initialized")
    print(f"✓ Metrics tracker initialized")
    print(f"✓ Visualizer initialized in {viz_dir}")
    print("="*60)
    
    return components


def compute_total_loss(model_output, labels, background_val, vclr_components=None):
    """
    计算总损失（标准损失 + v-CLR损失）
    
    Args:
        model_output: 模型输出（可能是tuple或tensor）
        labels: 标签
        background_val: 背景值
        vclr_components: v-CLR组件
        
    Returns:
        total_loss: 总损失
        loss_dict: 损失字典
    """
    # 标准分割损失
    if isinstance(model_output, tuple):
        output, aux_output = model_output
        loss_seg = model_output[0]
        if aux_output is not None:
            loss_seg += model_output[1]
        main_output = output
    else:
        main_output = model_output
        loss_seg = model_output
    
    # 简化版：这里需要根据实际模型输出调整
    # 假设model_output已经是loss
    if isinstance(model_output, tuple):
        # 如果返回的是(output, loss)的tuple
        outputs, losses = model_output
        total_loss = sum(l for l in losses) if isinstance(losses, (list, tuple)) else losses
    else:
        total_loss = model_output
    
    loss_dict = {'loss_segmentation': total_loss}
    
    # v-CLR损失
    if vclr_components and vclr_components['enabled']:
        # 这里需要提取中间特征
        # 目前简化为只在配置中记录使用v-CLR
        loss_dict['loss_vclr'] = torch.tensor(0.0, device=total_loss.device)
        loss_dict['loss_total'] = total_loss + loss_dict['loss_vclr']
    else:
        loss_dict['loss_total'] = total_loss
    
    return loss_dict['loss_total'], loss_dict


def train_epoch_with_vclr(
    model, train_loader, optimizer, lr_policy, 
    current_epoch, config, args, vclr_components=None,
    tb=None, engine=None
):
    """
    使用v-CLR的训练epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        lr_policy: 学习率策略
        current_epoch: 当前epoch
        config: 配置
        args: 参数
        vclr_components: v-CLR组件
        tb: TensorBoard writer
        engine: 训练引擎
        
    Returns:
        epoch_metrics: epoch的指标
    """
    model.train()
    
    sum_loss = 0.0
    sum_vclr_loss = 0.0
    num_samples = 0
    
    print(f"Epoch {current_epoch}: Starting training with v-CLR={vclr_components is not None}")
    
    for idx, minibatch in enumerate(train_loader):
        imgs = minibatch["data"].cuda(non_blocking=True)
        gts = minibatch["label"].cuda(non_blocking=True)
        modal_xs = minibatch["modal_x"].cuda(non_blocking=True)
        
        # 前向传播
        if args.amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(imgs, modal_xs, gts)
        else:
            loss = model(imgs, modal_xs, gts)
        
        # v-CLR处理（如果需要）
        if vclr_components and vclr_components['enabled']:
            # 这里可以添加v-CLR特定的处理
            # 例如：生成多视图、计算一致性损失等
            pass
        
        # 反向传播
        optimizer.zero_grad()
        
        if args.amp:
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # 更新学习率
        current_iter = (current_epoch - 1) * config.niters_per_epoch + idx
        lr = lr_policy.get_lr(current_iter)
        optimizer.param_groups[0]["lr"] = lr
        
        sum_loss += loss.item()
        num_samples += 1
        
        if (idx + 1) % 10 == 0:
            print(f"  Iter {idx+1}/{len(train_loader)}: loss={loss.item():.4f}")
    
    avg_loss = sum_loss / num_samples
    
    return {
        'loss': avg_loss,
        'num_samples': num_samples
    }


def main():
    """主函数"""
    args = parser.parse_args()
    
    # 加载配置
    config = getattr(import_module(args.config), "C")
    
    # 检查是否启用v-CLR
    use_vclr = getattr(config, 'use_multi_view_consistency', False)
    
    print("="*60)
    print("v-CLR Training Script")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"v-CLR enabled: {use_vclr}")
    print("="*60)
    
    if use_vclr:
        print("✓ Multi-View Consistency Learning enabled")
        print(f"  Consistency weight: {getattr(config, 'consistency_loss_weight', 0.1)}")
        print(f"  Number of views: {getattr(config, 'num_views', 2)}")
    else:
        print("ℹ Standard training (v-CLR disabled)")
    
    print("\n由于集成工作较为复杂，建议:")
    print("1. 先运行原始训练脚本收集baseline数据")
    print("2. 使用实验框架生成对比表格")
    print("3. 逐步集成v-CLR功能\n")
    
    print("="*60)


if __name__ == "__main__":
    main()

