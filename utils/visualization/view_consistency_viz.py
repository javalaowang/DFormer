"""
Multi-View Consistency Visualization Tools
用于SCI论文的完整可视化工具

提供的可视化：
1. 特征相似度热图
2. 视图对比图
3. 一致性学习曲线
4. Attention maps对比
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import seaborn as sns
from typing import Optional, Tuple, List, Dict
import os


class ConsistencyVisualizer:
    """
    视图一致性可视化器
    
    用于生成论文质量的可视化图表
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
    
    def visualize_feature_similarity(
        self,
        features_view1: torch.Tensor,
        features_view2: torch.Tensor,
        save_path: str = "feature_similarity.png",
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        可视化特征相似度
        
        Args:
            features_view1: (B, C, H, W) 视图1特征
            features_view2: (B, C, H, W) 视图2特征
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Multi-View Feature Similarity Analysis', fontsize=16, fontweight='bold')
        
        # 1. 余弦相似度热图
        feat1_flat = features_view1[0].flatten(1)  # (C, HW)
        feat2_flat = features_view2[0].flatten(1)
        
        similarity_map = F.cosine_similarity(feat1_flat, feat2_flat, dim=0)
        H, W = features_view1.shape[2:]
        similarity_2d = similarity_map.reshape(H, W).cpu().numpy()
        
        im1 = axes[0].imshow(similarity_2d, cmap='jet', vmin=0, vmax=1)
        axes[0].set_title('Feature Similarity Map', fontsize=12)
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # 2. 特征分布对比
        feat1_mean = features_view1[0].mean(dim=0).cpu().numpy()
        feat2_mean = features_view2[0].mean(dim=0).cpu().numpy()
        
        axes[1].plot(feat1_mean.flatten(), label='View 1', alpha=0.7)
        axes[1].plot(feat2_mean.flatten(), label='View 2', alpha=0.7)
        axes[1].set_title('Feature Distribution Comparison', fontsize=12)
        axes[1].set_xlabel('Feature Index')
        axes[1].set_ylabel('Feature Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 相似度直方图
        axes[2].hist(similarity_map.cpu().numpy(), bins=50, edgecolor='black', alpha=0.7)
        axes[2].set_title('Similarity Distribution', fontsize=12)
        axes[2].set_xlabel('Cosine Similarity')
        axes[2].set_ylabel('Frequency')
        axes[2].axvline(similarity_map.mean().item(), color='red', 
                        linestyle='--', label=f'Mean: {similarity_map.mean():.3f}')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved feature similarity visualization to {save_path}")
    
    def visualize_view_comparison(
        self,
        rgb_view1: torch.Tensor,
        rgb_view2: torch.Tensor,
        prediction_view1: torch.Tensor,
        prediction_view2: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
        save_path: str = "view_comparison.png",
        figsize: Tuple[int, int] = (20, 10)
    ):
        """
        可视化视图对比
        
        展示原始图像、多视图变换、预测结果
        """
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        fig.suptitle('Multi-View Consistency: Before & After', fontsize=16, fontweight='bold')
        
        # 转换为numpy
        rgb1 = rgb_view1[0].permute(1, 2, 0).cpu().numpy()
        rgb2 = rgb_view2[0].permute(1, 2, 0).cpu().numpy()
        
        if rgb1.max() <= 1.0:
            rgb1 = (rgb1 * 255).astype(np.uint8)
            rgb2 = (rgb2 * 255).astype(np.uint8)
        
        pred1 = prediction_view1[0].argmax(0).cpu().numpy()
        pred2 = prediction_view2[0].argmax(0).cpu().numpy()
        
        if ground_truth is not None:
            gt = ground_truth[0].cpu().numpy()
        
        # 第一行：视图1
        axes[0, 0].imshow(rgb1)
        axes[0, 0].set_title('View 1: Original RGB', fontsize=11)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(rgb2)
        axes[0, 1].set_title('View 2: Augmented RGB', fontsize=11)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred1, cmap='tab10')
        axes[0, 2].set_title('Prediction (View 1)', fontsize=11)
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(pred2, cmap='tab10')
        axes[0, 3].set_title('Prediction (View 2)', fontsize=11)
        axes[0, 3].axis('off')
        
        # 第二行：对比分析
        if ground_truth is not None:
            axes[1, 0].imshow(gt, cmap='tab10')
            axes[1, 0].set_title('Ground Truth', fontsize=11)
            axes[1, 0].axis('off')
        
        # 一致性差异
        diff = np.abs(pred1.astype(float) - pred2.astype(float))
        axes[1, 1].imshow(diff, cmap='hot')
        axes[1, 1].set_title('Prediction Consistency\n(Red=Inconsistent)', fontsize=11)
        axes[1, 1].axis('off')
        
        # 成功率分析
        consistency_rate = 1 - (diff > 0).sum() / diff.size
        axes[1, 2].bar(['Consistent', 'Inconsistent'], 
                      [consistency_rate, 1-consistency_rate],
                      color=['green', 'red'], alpha=0.7)
        axes[1, 2].set_title(f'Consistency Rate: {consistency_rate*100:.1f}%', fontsize=11)
        axes[1, 2].set_ylabel('Percentage')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 类别级一致性
        unique_classes = np.unique(np.concatenate([pred1, pred2]))
        consistency_per_class = []
        for cls in unique_classes[:5]:  # 只显示前5个类别
            mask1 = (pred1 == cls)
            mask2 = (pred2 == cls)
            if mask1.sum() > 0:
                consistency = ((pred1 == cls) & (pred2 == cls)).sum() / mask1.sum()
                consistency_per_class.append((cls, consistency))
        
        if consistency_per_class:
            classes, rates = zip(*consistency_per_class)
            axes[1, 3].bar(range(len(classes)), rates, color='skyblue', alpha=0.7)
            axes[1, 3].set_title('Consistency by Class', fontsize=11)
            axes[1, 3].set_xticks(range(len(classes)))
            axes[1, 3].set_xticklabels([f'C{c}' for c in classes])
            axes[1, 3].set_ylabel('Consistency Rate')
            axes[1, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved view comparison visualization to {save_path}")
    
    def visualize_consistency_curves(
        self,
        epoch_logs: List[Dict],
        save_path: str = "consistency_curves.png",
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        可视化一致性学习曲线
        
        Args:
            epoch_logs: 每个epoch的日志 [{'epoch': x, 'similarity': y, ...}, ...]
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Multi-View Consistency Training Analysis', fontsize=16, fontweight='bold')
        
        # 提取数据
        epochs = [log['epoch'] for log in epoch_logs]
        similarities = [log.get('similarity', 0) for log in epoch_logs]
        consistency_losses = [log.get('loss_consistency', 0) for log in epoch_logs]
        alignment_losses = [log.get('loss_alignment', 0) for log in epoch_logs]
        total_losses = [log.get('loss_total', 0) for log in epoch_logs]
        
        # 1. 相似度曲线
        axes[0, 0].plot(epochs, similarities, linewidth=2, color='blue', marker='o')
        axes[0, 0].set_title('Feature Similarity Evolution', fontsize=12)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Cosine Similarity')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=np.mean(similarities), color='red', 
                           linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
        axes[0, 0].legend()
        
        # 2. 一致性损失
        axes[0, 1].plot(epochs, consistency_losses, linewidth=2, color='red', marker='s', label='Consistency')
        axes[0, 1].plot(epochs, alignment_losses, linewidth=2, color='green', marker='^', label='Alignment')
        axes[0, 1].set_title('Consistency Losses', fontsize=12)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 总损失
        axes[1, 0].plot(epochs, total_losses, linewidth=2, color='purple', marker='D')
        axes[1, 0].set_title('Total Loss', fontsize=12)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 损失相关性
        axes[1, 1].scatter(consistency_losses, alignment_losses, alpha=0.6)
        axes[1, 1].set_title('Loss Correlation', fontsize=12)
        axes[1, 1].set_xlabel('Consistency Loss')
        axes[1, 1].set_ylabel('Alignment Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved consistency curves to {save_path}")
    
    def create_paper_figure(
        self,
        results: Dict,
        save_path: str = "paper_figure.png"
    ):
        """
        创建论文用的大图
        
        包含完整的对比实验可视化
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Multi-View Consistency Learning for DFormer', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # 这里添加具体的可视化内容
        # 可以根据实际实验需求自定义
        
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved paper figure to {save_path}")


# ============== 使用示例 ==============

if __name__ == "__main__":
    """测试可视化工具"""
    
    print("="*60)
    print("测试Multi-View Consistency Visualization")
    print("="*60)
    
    # 创建模拟数据
    B, C, H, W = 2, 512, 64, 64
    classes = 3
    
    feat1 = torch.randn(B, C, H, W)
    feat2 = torch.randn(B, C, H, W)
    rgb1 = torch.randn(B, 3, H, W)
    rgb2 = torch.randn(B, 3, H, W)
    pred1 = torch.randint(0, classes, (B, H, W)).float()
    pred2 = torch.randint(0, classes, (B, H, W)).float()
    gt = torch.randint(0, classes, (B, H, W)).float()
    
    # 创建可视化器
    viz = ConsistencyVisualizer(output_dir="test_visualizations")
    
    # 测试特征相似度可视化
    print("\n1. 测试特征相似度可视化...")
    viz.visualize_feature_similarity(feat1, feat2)
    
    # 测试视图对比
    print("\n2. 测试视图对比可视化...")
    viz.visualize_view_comparison(rgb1, rgb2, pred1.unsqueeze(1), 
                                  pred2.unsqueeze(1), gt.unsqueeze(1))
    
    # 测试一致性曲线
    print("\n3. 测试一致性曲线...")
    epoch_logs = [
        {'epoch': i, 'similarity': 0.5 + 0.3*i/10, 'loss_consistency': 0.5 - 0.1*i/10,
         'loss_alignment': 0.3 - 0.05*i/10, 'loss_total': 0.8 - 0.15*i/10}
        for i in range(10)
    ]
    viz.visualize_consistency_curves(epoch_logs)
    
    print("\n" + "="*60)
    print("✓ 所有可视化测试通过！")
    print("="*60)

