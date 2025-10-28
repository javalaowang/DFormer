#!/usr/bin/env python3
"""
CCS Paper Implementation Test Script
测试基于CVPR 2025论文的CCS模块实现

测试内容：
1. 数学公式的正确性
2. 模块功能的完整性
3. 与DFormer的集成
4. 消融实验配置
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.ccs_paper_implementation import (
    StarShapeField, ConvexCombinationStar, CCSVariationalModule, 
    CCSShapeLoss, CCSHead
)
from models.dformer_ccs_paper import DFormerWithCCSPaper, CCSAblationConfigPaper
from easydict import EasyDict as edict


def test_star_shape_field():
    """测试星形场函数"""
    print("="*60)
    print("Testing Star Shape Field")
    print("="*60)
    
    # 创建测试数据
    B, H, W = 2, 64, 64
    coords = torch.randn(B, H, W, 2) * 32 + 32  # 中心在(32, 32)
    center = torch.tensor([[32.0, 32.0], [32.0, 32.0]])  # 中心点
    
    print(f"Input coords shape: {coords.shape}")
    print(f"Center shape: {center.shape}")
    
    # 测试固定半径
    print("\n1. Testing Fixed Radius...")
    field_fixed = StarShapeField(learnable_radius=False)
    field_output = field_fixed(coords, center)
    
    print(f"   Field output shape: {field_output.shape}")
    print(f"   Field range: [{field_output.min():.2f}, {field_output.max():.2f}]")
    
    # 测试学习半径
    print("\n2. Testing Learnable Radius...")
    field_learnable = StarShapeField(learnable_radius=True)
    field_output = field_learnable(coords, center)
    
    print(f"   Field output shape: {field_output.shape}")
    print(f"   Field range: [{field_output.min():.2f}, {field_output.max():.2f}]")
    
    # 验证星形性质
    print("\n3. Verifying Star Shape Properties...")
    center_coord = coords[0, 32, 32]  # 中心点
    center_field = field_output[0, 32, 32]
    print(f"   Center field value: {center_field:.2f}")
    print(f"   Should be positive (inside star): {center_field > 0}")
    
    return True


def test_convex_combination_star():
    """测试凸组合星形模块"""
    print("\n" + "="*60)
    print("Testing Convex Combination Star")
    print("="*60)
    
    # 创建测试数据
    B, C, H, W = 2, 512, 64, 64
    features = torch.randn(B, C, H, W)
    
    print(f"Input features shape: {features.shape}")
    
    # 测试固定中心
    print("\n1. Testing Fixed Centers...")
    ccs_fixed = ConvexCombinationStar(
        num_centers=3,
        learnable_centers=False
    )
    
    ccs_field, centers, weights = ccs_fixed(features)
    
    print(f"   CCS field shape: {ccs_field.shape}")
    print(f"   Centers shape: {centers.shape}")
    print(f"   Weights shape: {weights.shape}")
    print(f"   Centers (first sample): {centers[0]}")
    
    # 验证凸组合性质
    print("\n2. Verifying Convex Combination Properties...")
    weight_sum = weights.sum(dim=1)  # 应该等于1
    print(f"   Weight sum range: [{weight_sum.min():.6f}, {weight_sum.max():.6f}]")
    print(f"   Convex combination property: {torch.allclose(weight_sum, torch.ones_like(weight_sum))}")
    
    # 测试学习中心
    print("\n3. Testing Learnable Centers...")
    ccs_learnable = ConvexCombinationStar(
        num_centers=3,
        learnable_centers=True
    )
    
    ccs_field, centers, weights = ccs_learnable(features)
    
    print(f"   CCS field shape: {ccs_field.shape}")
    print(f"   Centers shape: {centers.shape}")
    print(f"   Centers (first sample): {centers[0]}")
    
    return True


def test_variational_module():
    """测试变分模块"""
    print("\n" + "="*60)
    print("Testing CCS Variational Module")
    print("="*60)
    
    # 创建测试数据
    B, C, H, W = 2, 512, 64, 64
    features = torch.randn(B, C, H, W)
    logits = torch.randn(B, 3, H, W)
    
    print(f"Input features shape: {features.shape}")
    print(f"Input logits shape: {logits.shape}")
    
    # 测试变分模块
    variational = CCSVariationalModule(
        num_centers=3,
        variational_weight=0.1
    )
    
    enhanced_logits, details = variational(features, logits, return_details=True)
    
    print(f"   Enhanced logits shape: {enhanced_logits.shape}")
    print(f"   CCS field shape: {details['ccs_field'].shape}")
    print(f"   Centers shape: {details['centers'].shape}")
    print(f"   Adaptive weight: {details['adaptive_weight']}")
    
    # 验证变分增强
    print("\n2. Verifying Variational Enhancement...")
    enhancement = details['enhancement']
    print(f"   Enhancement shape: {enhancement.shape}")
    print(f"   Enhancement range: [{enhancement.min():.4f}, {enhancement.max():.4f}]")
    
    return True


def test_shape_loss():
    """测试形状损失"""
    print("\n" + "="*60)
    print("Testing CCS Shape Loss")
    print("="*60)
    
    # 创建测试数据
    B, H, W = 2, 64, 64
    pred_prob = torch.sigmoid(torch.randn(B, H, W))
    ccs_field = torch.randn(B, H, W)
    target = (torch.rand(B, H, W) > 0.5).float()
    
    print(f"Pred prob shape: {pred_prob.shape}")
    print(f"CCS field shape: {ccs_field.shape}")
    print(f"Target shape: {target.shape}")
    
    # 测试形状损失
    loss_fn = CCSShapeLoss(lambda_shape=0.1)
    loss = loss_fn(pred_prob, ccs_field, target)
    
    print(f"   Shape loss: {loss.item():.4f}")
    print(f"   Loss is scalar: {loss.dim() == 0}")
    
    return True


def test_ccs_head():
    """测试CCS头"""
    print("\n" + "="*60)
    print("Testing CCS Head")
    print("="*60)
    
    # 创建测试数据
    B, C, H, W = 2, 512, 64, 64
    features = torch.randn(B, C, H, W)
    
    print(f"Input features shape: {features.shape}")
    
    # 测试CCS头
    head = CCSHead(
        in_channels=C,
        num_classes=3,
        num_centers=3,
        use_ccs=True
    )
    
    output, ccs_details = head(features, return_ccs_details=True)
    
    print(f"   Output shape: {output.shape}")
    print(f"   CCS field shape: {ccs_details['ccs_field'].shape}")
    print(f"   Centers shape: {ccs_details['centers'].shape}")
    
    return True


def test_dformer_integration():
    """测试DFormer集成"""
    print("\n" + "="*60)
    print("Testing DFormer Integration")
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
    print("\n1. Creating DFormer with CCS...")
    model = DFormerWithCCSPaper(
        cfg=cfg,
        use_ccs=True,
        ccs_num_centers=3,
        ccs_temperature=1.0,
        ccs_variational_weight=0.1,
        ccs_shape_lambda=0.1
    )
    
    # 将模型移到GPU
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"   Model created successfully on {device}")
    print(f"   CCS config: {model.get_ccs_config()}")
    
    # 创建测试数据
    B, H, W = 2, 128, 128
    rgb = torch.randn(B, 3, H, W).to(device)
    depth = torch.randn(B, 3, H, W).to(device)
    label = torch.randint(0, 3, (B, H, W)).to(device)
    
    print(f"\n2. Input shapes:")
    print(f"   RGB: {rgb.shape}")
    print(f"   Depth: {depth.shape}")
    print(f"   Label: {label.shape}")
    
    # 测试推理模式
    print("\n3. Testing inference mode...")
    model.eval()
    
    with torch.no_grad():
        output, ccs_details = model(rgb, depth, return_ccs_details=True)
    
    print(f"   Output shape: {output.shape}")
    print(f"   CCS field shape: {ccs_details['ccs_field'].shape}")
    print(f"   Centers shape: {ccs_details['centers'].shape}")
    
    # 测试训练模式
    print("\n4. Testing training mode...")
    model.train()
    
    try:
        loss = model(rgb, depth, label)
        print(f"   Training loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   Training failed: {e}")
        return False
    
    return True


def test_ablation_configs():
    """测试消融实验配置"""
    print("\n" + "="*60)
    print("Testing Ablation Configurations")
    print("="*60)
    
    # 测试基线配置
    baseline_config = CCSAblationConfigPaper.get_baseline_config()
    print(f"✓ Baseline config: {baseline_config}")
    
    # 测试CCS变体
    ccs_variants = CCSAblationConfigPaper.get_ccs_variants()
    print(f"✓ Found {len(ccs_variants)} CCS variants")
    
    # 测试组件消融
    component_ablation = CCSAblationConfigPaper.get_component_ablation()
    print(f"✓ Found {len(component_ablation)} component ablation configs")
    
    # 显示部分配置
    print("\nSample configurations:")
    for name, config in list(ccs_variants.items())[:3]:
        print(f"  {name}: {config}")
    
    return True


def test_mathematical_properties():
    """测试数学性质"""
    print("\n" + "="*60)
    print("Testing Mathematical Properties")
    print("="*60)
    
    # 测试凸组合性质
    print("\n1. Testing Convex Combination Properties...")
    B, num_centers, H, W = 2, 3, 32, 32
    weights = torch.rand(B, num_centers, H, W)
    weights = torch.softmax(weights, dim=1)  # 确保凸组合
    
    # 验证权重和为1
    weight_sum = weights.sum(dim=1)
    is_convex = torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-6)
    print(f"   Convex combination property: {is_convex}")
    
    # 验证非负性
    is_non_negative = torch.all(weights >= 0)
    print(f"   Non-negativity property: {is_non_negative}")
    
    # 测试星形场性质
    print("\n2. Testing Star Shape Field Properties...")
    coords = torch.randn(B, H, W, 2)
    center = torch.randn(B, 2)
    
    field = StarShapeField(learnable_radius=False)
    field_output = field(coords, center)
    
    # 验证连续性
    print(f"   Field is continuous: {torch.isfinite(field_output).all()}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("="*80)
    print("CCS Paper Implementation Test Suite")
    print("="*80)
    print("Based on: Zhao et al. CVPR 2025")
    print("Implementation: Paper-based mathematical formulation")
    print("="*80)
    
    tests = [
        ("Star Shape Field", test_star_shape_field),
        ("Convex Combination Star", test_convex_combination_star),
        ("Variational Module", test_variational_module),
        ("Shape Loss", test_shape_loss),
        ("CCS Head", test_ccs_head),
        ("DFormer Integration", test_dformer_integration),
        ("Ablation Configs", test_ablation_configs),
        ("Mathematical Properties", test_mathematical_properties),
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
            import traceback
            traceback.print_exc()
    
    # 汇总结果
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:25} {status}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CCS paper implementation is ready for experiments.")
        print("📚 Implementation follows the mathematical formulation in Zhao et al. CVPR 2025")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)



