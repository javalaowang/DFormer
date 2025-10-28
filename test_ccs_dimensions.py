#!/usr/bin/env python3
"""
测试CCS模块的维度问题
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ccs_paper_implementation import CCSHead

def test_ccs_dimensions():
    """测试CCS模块的维度兼容性"""
    
    print("="*60)
    print("Testing CCS Module Dimensions")
    print("="*60)
    
    # 测试不同的输入维度
    test_dims = [256, 512, 640, 576]
    
    for dim in test_dims:
        print(f"\nTesting with input dimension: {dim}")
        
        try:
            # 创建CCS模块
            ccs_head = CCSHead(
                in_channels=dim,
                num_classes=3,
                num_centers=5,
                temperature=1.0,
                variational_weight=0.1,
                use_ccs=True
            )
            
            # 创建测试数据
            B, H, W = 2, 64, 64
            features = torch.randn(B, dim, H, W)
            
            print(f"  Input features shape: {features.shape}")
            
            # 前向传播
            with torch.no_grad():
                logits, ccs_details = ccs_head(features, return_ccs_details=True)
            
            print(f"  ✓ Success! Output shape: {logits.shape}")
            print(f"  CCS field shape: {ccs_details['ccs_field'].shape}")
            print(f"  Centers shape: {ccs_details['centers'].shape}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Dimension test completed!")
    print("="*60)

if __name__ == "__main__":
    test_ccs_dimensions()
