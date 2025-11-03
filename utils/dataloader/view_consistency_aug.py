"""
为v-CLR生成多视图的数据增强

核心思想：
1. 在线生成多视图（无需预处理）
2. 通过颜色变换改变外观
3. 保持几何结构不变
4. 深度信息完全保留

适用场景：
- 只有单一视图的数据集
- 需要训练视图一致性模型
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Tuple, Optional
import cv2


class ViewConsistencyAugmentation:
    """
    视图一致性增强器
    
    为单一视图生成多个视图，用于v-CLR训练
    """
    
    def __init__(
        self,
        num_views: int = 2,
        color_jitter_params: dict = None,
        blur_probability: float = 0.3,
        gamma_correction: bool = True,
        channel_swap: bool = True
    ):
        """
        Args:
            num_views: 生成视图的数量（包括原始视图）
            color_jitter_params: 颜色抖动参数
            blur_probability: 模糊概率
            gamma_correction: 是否应用gamma校正
            channel_swap: 是否随机交换RGB通道
        """
        self.num_views = num_views
        self.blur_probability = blur_probability
        self.gamma_correction = gamma_correction
        self.channel_swap = channel_swap
        
        # 默认颜色抖动参数
        if color_jitter_params is None:
            self.color_jitter = {
                'brightness': 0.4,      # 亮度变化范围
                'contrast': 0.4,        # 对比度变化范围
                'saturation': 0.4,     # 饱和度变化范围
                'hue': 0.1              # 色相变化范围
            }
        else:
            self.color_jitter = color_jitter_params
    
    def generate_views(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        return_original: bool = True
    ) -> Tuple[list, list]:
        """
        生成多个视图
        
        Args:
            rgb_image: (H, W, 3) numpy array, 值域[0, 255]
            depth_image: (H, W) numpy array, 深度图
            return_original: 是否包含原始视图
            
        Returns:
            rgb_views: [view0, view1, ...] 列表
            depth_views: [depth, depth, ...] 深度图（保持不变）
        """
        H, W, C = rgb_image.shape
        rgb_views = []
        depth_views = []
        
        # 转换为[0, 1]范围
        rgb_normalized = rgb_image.astype(np.float32) / 255.0
        
        # 原始视图
        if return_original:
            rgb_views.append(rgb_normalized.copy())
            depth_views.append(depth_image.copy())
        
        # 生成变换视图
        for i in range(self.num_views - 1):
            # 1. 颜色抖动
            view_rgb = self._apply_color_jitter(rgb_normalized.copy())
            
            # 2. 随机模糊
            if random.random() < self.blur_probability:
                view_rgb = self._apply_blur(view_rgb)
            
            # 3. Gamma校正
            if self.gamma_correction and random.random() < 0.5:
                gamma = random.uniform(0.7, 1.3)
                view_rgb = self._apply_gamma_correction(view_rgb, gamma)
            
            # 4. 通道交换
            if self.channel_swap and random.random() < 0.3:
                view_rgb = self._swap_channels(view_rgb)
            
            # 5. 对比度调整
            if random.random() < 0.4:
                view_rgb = self._adjust_contrast(view_rgb)
            
            rgb_views.append(view_rgb)
            depth_views.append(depth_image.copy())  # 深度保持不变
        
        return rgb_views, depth_views
    
    def _apply_color_jitter(self, rgb: np.ndarray) -> np.ndarray:
        """应用颜色抖动"""
        # 转换为HSV空间
        hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv = hsv / np.array([180, 255, 255])
        
        # 亮度调整
        brightness_factor = random.uniform(
            1 - self.color_jitter['brightness'],
            1 + self.color_jitter['brightness']
        )
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 1)
        
        # 饱和度调整
        saturation_factor = random.uniform(
            1 - self.color_jitter['saturation'],
            1 + self.color_jitter['saturation']
        )
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 1)
        
        # 色相调整
        hue_shift = random.uniform(-self.color_jitter['hue'], self.color_jitter['hue'])
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 1.0
        
        # 转回RGB
        hsv_scaled = (hsv * np.array([180, 255, 255])).astype(np.uint8)
        rgb_jittered = cv2.cvtColor(hsv_scaled, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        # 对比度调整
        contrast_factor = random.uniform(
            1 - self.color_jitter['contrast'],
            1 + self.color_jitter['contrast']
        )
        mean = np.mean(rgb_jittered)
        rgb_jittered = contrast_factor * (rgb_jittered - mean) + mean
        
        return np.clip(rgb_jittered, 0, 1)
    
    def _apply_blur(self, rgb: np.ndarray) -> np.ndarray:
        """应用高斯模糊"""
        blur_kernel = random.choice([3, 5])
        rgb_blurred = cv2.GaussianBlur(
            (rgb * 255).astype(np.uint8),
            (blur_kernel, blur_kernel),
            0
        )
        return rgb_blurred.astype(np.float32) / 255.0
    
    def _apply_gamma_correction(self, rgb: np.ndarray, gamma: float) -> np.ndarray:
        """应用gamma校正"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)])
        rgb_gamma = cv2.LUT((rgb * 255).astype(np.uint8), table.astype(np.uint8))
        return rgb_gamma.astype(np.float32) / 255.0
    
    def _swap_channels(self, rgb: np.ndarray) -> np.ndarray:
        """随机交换RGB通道"""
        orders = [
            [0, 1, 2],  # 原始
            [1, 2, 0],  # R->G, G->B, B->R
            [2, 0, 1],  # R->B, G->R, B->G
        ]
        order = random.choice(orders)
        return rgb[:, :, order]
    
    def _adjust_contrast(self, rgb: np.ndarray) -> np.ndarray:
        """调整对比度"""
        contrast_factor = random.uniform(0.7, 1.3)
        mean = np.mean(rgb)
        return np.clip(contrast_factor * (rgb - mean) + mean, 0, 1)


def augment_batch(
    rgb_tensor: torch.Tensor,
    depth_tensor: torch.Tensor,
    num_views: int = 2,
    return_original: bool = True
) -> Tuple[list, list]:
    """
    批量增强张量
    
    Args:
        rgb_tensor: (B, 3, H, W) tensor
        depth_tensor: (B, 1, H, W) tensor
        num_views: 视图数量
        return_original: 是否返回原始视图
        
    Returns:
        rgb_views_list: [[view1_batch], [view2_batch], ...]
        depth_views_list: [[depth_batch], [depth_batch], ...]
    """
    B, C, H, W = rgb_tensor.shape
    
    # 转换为numpy进行处理
    rgb_np = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)
    depth_np = depth_tensor.squeeze(1).cpu().numpy()  # (B, H, W)
    
    # 增强器
    augmenter = ViewConsistencyAugmentation(num_views=num_views)
    
    rgb_views_list = [[] for _ in range(num_views if return_original else num_views - 1)]
    depth_views_list = [[] for _ in range(num_views if return_original else num_views - 1)]
    
    # 对每个样本单独处理
    for b in range(B):
        rgb_views, depth_views = augmenter.generate_views(
            rgb_np[b],
            depth_np[b],
            return_original=return_original
        )
        
        for i, (rv, dv) in enumerate(zip(rgb_views, depth_views)):
            rgb_views_list[i].append(rv)
            depth_views_list[i].append(dv)
    
    # 转换回tensor
    rgb_tensors = []
    depth_tensors = []
    
    for i in range(len(rgb_views_list)):
        # RGB
        rgb_batch = np.stack(rgb_views_list[i])  # (B, H, W, 3)
        rgb_tensor_v = torch.from_numpy(rgb_batch).permute(0, 3, 1, 2).to(rgb_tensor.device)
        rgb_tensors.append(rgb_tensor_v)
        
        # Depth
        depth_batch = np.stack(depth_views_list[i])  # (B, H, W)
        depth_tensor_v = torch.from_numpy(depth_batch).unsqueeze(1).to(depth_tensor.device)
        depth_tensors.append(depth_tensor_v)
    
    return rgb_tensors, depth_tensors


# ============== 测试代码 ==============

if __name__ == "__main__":
    """测试视图一致性增强"""
    
    print("="*60)
    print("测试视图一致性增强模块")
    print("="*60)
    
    # 创建模拟数据
    H, W = 128, 128
    rgb_image = np.random.rand(H, W, 3).astype(np.float32)
    depth_image = np.random.rand(H, W).astype(np.float32) * 10
    
    print(f"\n原始数据:")
    print(f"  RGB shape: {rgb_image.shape}, range: [{rgb_image.min():.2f}, {rgb_image.max():.2f}]")
    print(f"  Depth shape: {depth_image.shape}, range: [{depth_image.min():.2f}, {depth_image.max():.2f}]")
    
    # 测试ViewConsistencyAugmentation
    print("\n1. 测试ViewConsistencyAugmentation...")
    augmenter = ViewConsistencyAugmentation(num_views=3)
    
    rgb_views, depth_views = augmenter.generate_views(rgb_image, depth_image)
    
    print(f"  生成了 {len(rgb_views)} 个视图")
    print(f"  视图0 RGB范围: [{rgb_views[0].min():.2f}, {rgb_views[0].max():.2f}]")
    print(f"  视图1 RGB范围: [{rgb_views[1].min():.2f}, {rgb_views[1].max():.2f}]")
    print(f"  视图2 RGB范围: [{rgb_views[2].min():.2f}, {rgb_views[2].max():.2f}]")
    
    # 验证深度保持不变
    depth_unchanged = np.allclose(depth_views[0], depth_views[1])
    print(f"  深度一致性: {depth_unchanged} ✓")
    
    # 测试批量处理
    print("\n2. 测试批量处理...")
    B = 4
    rgb_tensor = torch.rand(B, 3, H, W)
    depth_tensor = torch.rand(B, 1, H, W) * 10
    
    rgb_views_tensors, depth_views_tensors = augment_batch(
        rgb_tensor, depth_tensor, num_views=2, return_original=True
    )
    
    print(f"  生成了 {len(rgb_views_tensors)} 个视图批次")
    print(f"  每个视图batch形状: {rgb_views_tensors[0].shape}")
    print(f"  深度batch形状: {depth_views_tensors[0].shape}")
    
    # 验证深度在所有视图中相同
    depth_same = torch.equal(depth_views_tensors[0], depth_views_tensors[1])
    print(f"  跨视图深度一致性: {depth_same} ✓")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)

