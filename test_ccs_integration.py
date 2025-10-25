#!/usr/bin/env python3
"""
CCS集成测试脚本
测试CCS模块与DFormer的集成是否正常工作

使用方法:
    python test_ccs_integration.py
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.ccs_integration import DFormerWithCCS, CCSAblationConfig
from models.shape_priors.ccs_module import CCSModule, CCSShapeLoss
from easydict import EasyDict as edict


def create_test_config():
    """创建测试配置"""
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
    return cfg


def test_ccs_module():
    """测试CCS模块"""
    print("="*60)
    print("Testing CCS Module")
    print("="*60)
    
    # 创建测试数据
    B, C, H, W = 2, 256, 128, 128
    features = torch.randn(B, C, H, W)
    
    print(f"Input features shape: {features.shape}")
    
    # 测试CCS模块
    ccs_module = CCSModule(
        num_centers=3,
        feature_dim=C,
        learnable_centers=True
    )
    
    # 前向传播
    ccs_field, centers, weights = ccs_module(
        features,
        return_centers=True,
        return_weights=True
    )
    
    print(f"✓ CCS field shape: {ccs_field.shape}")
    print(f"✓ Centers shape: {centers.shape}")
    print(f"✓ Weights shape: {weights.shape}")
    print(f"✓ Centers (first sample): {centers[0]}")
    
    # 测试形状损失
    pred_mask = torch.sigmoid(torch.randn(B, H, W))
    target = (torch.rand(B, H, W) > 0.5).float()
    
    loss_fn = CCSShapeLoss(lambda_shape=0.1)
    loss = loss_fn(pred_mask, ccs_field, target)
    print(f"✓ Shape loss: {loss.item():.4f}")
    
    return True


def test_dformer_with_ccs():
    """测试DFormer与CCS的集成"""
    print("\n" + "="*60)
    print("Testing DFormer with CCS Integration")
    print("="*60)
    
    cfg = create_test_config()
    
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
    
    print(f"\nInput shapes:")
    print(f"  RGB: {rgb.shape}")
    print(f"  Depth: {depth.shape}")
    print(f"  Label: {label.shape}")
    
    # 测试推理模式
    print("\n3. Testing Inference Mode...")
    baseline_model.eval()
    ccs_model.eval()
    
    with torch.no_grad():
        # 基线模型
        baseline_output = baseline_model(rgb, depth)
        print(f"  ✓ Baseline output shape: {baseline_output.shape}")
        
        # CCS模型
        ccs_output, ccs_details = ccs_model(rgb, depth, return_ccs_details=True)
        print(f"  ✓ CCS output shape: {ccs_output.shape}")
        print(f"  ✓ CCS field shape: {ccs_details['ccs_field'].shape}")
        print(f"  ✓ CCS centers shape: {ccs_details['centers'].shape}")
    
    # 测试训练模式
    print("\n4. Testing Training Mode...")
    ccs_model.train()
    
    try:
        loss = ccs_model(rgb, depth, label)
        print(f"  ✓ Training loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        return False
    
    return True


def test_ablation_configs():
    """测试消融实验配置"""
    print("\n" + "="*60)
    print("Testing Ablation Configurations")
    print("="*60)
    
    # 测试基线配置
    baseline_config = CCSAblationConfig.get_baseline_config()
    print(f"✓ Baseline config: {baseline_config}")
    
    # 测试CCS变体
    ccs_variants = CCSAblationConfig.get_ccs_variants()
    print(f"✓ Found {len(ccs_variants)} CCS variants")
    
    # 测试组件消融
    component_ablation = CCSAblationConfig.get_component_ablation()
    print(f"✓ Found {len(component_ablation)} component ablation configs")
    
    # 显示部分配置
    print("\nSample configurations:")
    for name, config in list(ccs_variants.items())[:3]:
        print(f"  {name}: {config}")
    
    return True


def test_memory_usage():
    """测试内存使用"""
    print("\n" + "="*60)
    print("Testing Memory Usage")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return True
    
    cfg = create_test_config()
    
    # 创建模型
    model = DFormerWithCCS(
        cfg,
        use_ccs=True,
        ccs_num_centers=5,
        ccs_lambda=0.1,
        ccs_alpha=0.1
    ).cuda()
    
    # 创建测试数据
    B, H, W = 1, 256, 256
    rgb = torch.randn(B, 3, H, W).cuda()
    depth = torch.randn(B, 3, H, W).cuda()
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        output, details = model(rgb, depth, return_ccs_details=True)
    
    # 检查内存使用
    memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"✓ Peak GPU memory usage: {memory_used:.2f} GB")
    
    # 清理
    del model, rgb, depth, output, details
    torch.cuda.empty_cache()
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("="*80)
    print("CCS Integration Test Suite")
    print("="*80)
    
    tests = [
        ("CCS Module", test_ccs_module),
        ("DFormer with CCS", test_dformer_with_ccs),
        ("Ablation Configs", test_ablation_configs),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n{test_name}: ✗ FAILED - {e}")
    
    # 汇总结果
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:20} {status}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CCS integration is ready for experiments.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
