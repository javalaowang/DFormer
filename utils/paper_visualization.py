#!/usr/bin/env python3
"""
论文级别的可视化工具
用于生成符合顶级会议和期刊标准的图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import cv2
from PIL import Image
import os

# 设置论文级别的matplotlib参数
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.major.size': 5,
    'ytick.minor.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': False,
    'legend.loc': 'best',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# 设置seaborn样式
sns.set_style("whitegrid", {
    'axes.grid': True,
    'grid.color': '0.8',
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True
})

class PaperVisualization:
    """论文级别的可视化类"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义论文级别的颜色方案
        self.colors = {
            'primary': '#1f77b4',      # 蓝色
            'secondary': '#ff7f0e',    # 橙色
            'success': '#2ca02c',      # 绿色
            'danger': '#d62728',       # 红色
            'warning': '#ff7f0e',      # 橙色
            'info': '#17a2b8',         # 青色
            'light': '#f8f9fa',        # 浅灰色
            'dark': '#343a40',         # 深灰色
            'muted': '#6c757d'         # 静音色
        }
        
        # 定义线型
        self.line_styles = ['-', '--', '-.', ':']
        
        # 定义标记样式
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    def create_training_curves_figure(self, metrics_data, save_name="training_curves"):
        """创建训练曲线图 - 论文标准"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Performance Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        metrics = metrics_data['metrics_history']
        
        # 1. 损失曲线
        ax1 = axes[0, 0]
        epochs = range(len(metrics['train_loss']))
        ax1.plot(epochs, metrics['train_loss'], label='Training', 
                linewidth=2.5, color=self.colors['primary'], marker='o', markersize=4, markevery=5)
        ax1.plot(epochs, metrics['val_loss'], label='Validation', 
                linewidth=2.5, color=self.colors['secondary'], marker='s', markersize=4, markevery=5)
        ax1.set_title('(a) Loss Curves', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        
        # 2. mIoU曲线
        ax2 = axes[0, 1]
        ax2.plot(epochs, metrics['train_miou'], label='Training', 
                linewidth=2.5, color=self.colors['success'], marker='o', markersize=4, markevery=5)
        ax2.plot(epochs, metrics['val_miou'], label='Validation', 
                linewidth=2.5, color=self.colors['danger'], marker='s', markersize=4, markevery=5)
        ax2.set_title('(b) mIoU Curves', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('mIoU (%)', fontsize=12)
        ax2.legend(fontsize=11, loc='lower right')
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        
        # 3. 准确率曲线
        ax3 = axes[1, 0]
        ax3.plot(epochs, metrics['train_acc'], label='Training', 
                linewidth=2.5, color=self.colors['info'], marker='o', markersize=4, markevery=5)
        ax3.plot(epochs, metrics['val_acc'], label='Validation', 
                linewidth=2.5, color=self.colors['warning'], marker='s', markersize=4, markevery=5)
        ax3.set_title('(c) Accuracy Curves', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Accuracy (%)', fontsize=12)
        ax3.legend(fontsize=11, loc='lower right')
        ax3.grid(True, alpha=0.3, linewidth=0.5)
        
        # 4. 学习率调度
        ax4 = axes[1, 1]
        ax4.plot(epochs, metrics['learning_rate'], 
                linewidth=2.5, color=self.colors['dark'], marker='o', markersize=4, markevery=5)
        ax4.set_title('(d) Learning Rate Schedule', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Learning Rate', fontsize=12)
        ax4.grid(True, alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparison_figure(self, comparison_data, save_name="model_comparison"):
        """创建模型对比图"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 性能对比柱状图
        models = list(comparison_data.keys())
        mious = [comparison_data[model]['mIoU'] for model in models]
        accuracies = [comparison_data[model]['Accuracy'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, mious, width, label='mIoU (%)', 
                       color=self.colors['primary'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, accuracies, width, label='Accuracy (%)', 
                       color=self.colors['secondary'], alpha=0.8)
        
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Performance (%)', fontsize=12)
        ax1.set_title('(a) Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 训练时间对比
        training_times = [comparison_data[model]['Training Time (hours)'] for model in models]
        bars3 = ax2.bar(models, training_times, color=self.colors['success'], alpha=0.8)
        
        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_ylabel('Training Time (hours)', fontsize=12)
        ax2.set_title('(b) Training Time Comparison', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_ablation_study_figure(self, ablation_data, save_name="ablation_study"):
        """创建消融研究图"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 组件贡献分析
        components = list(ablation_data['components'].keys())
        contributions = list(ablation_data['components'].values())
        
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['success'], 
                 self.colors['danger'], self.colors['warning']]
        
        wedges, texts, autotexts = ax1.pie(contributions, labels=components, autopct='%1.1f%%',
                                          colors=colors[:len(components)], startangle=90)
        ax1.set_title('(a) Component Contribution Analysis', fontsize=14, fontweight='bold')
        
        # 2. 消融研究结果
        variants = list(ablation_data['variants'].keys())
        performances = list(ablation_data['variants'].values())
        
        bars = ax2.bar(variants, performances, color=self.colors['info'], alpha=0.8)
        ax2.set_xlabel('Model Variants', fontsize=12)
        ax2.set_ylabel('mIoU (%)', fontsize=12)
        ax2.set_title('(b) Ablation Study Results', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_qualitative_results_figure(self, image_paths, predictions, ground_truths, 
                                        save_name="qualitative_results"):
        """创建定性结果展示图"""
        
        n_samples = len(image_paths)
        fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 12))
        
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_samples):
            # 原始图像
            img = Image.open(image_paths[i])
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'(a{i+1}) Input Image', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            
            # 预测结果
            pred = predictions[i]
            axes[1, i].imshow(pred, cmap='viridis')
            axes[1, i].set_title(f'(b{i+1}) Prediction', fontsize=12, fontweight='bold')
            axes[1, i].axis('off')
            
            # 真实标签
            gt = ground_truths[i]
            axes[2, i].imshow(gt, cmap='viridis')
            axes[2, i].set_title(f'(c{i+1}) Ground Truth', fontsize=12, fontweight='bold')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_error_analysis_figure(self, error_data, save_name="error_analysis"):
        """创建错误分析图"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 类别错误分布
        classes = list(error_data['class_errors'].keys())
        errors = list(error_data['class_errors'].values())
        
        bars = ax1.bar(classes, errors, color=self.colors['danger'], alpha=0.8)
        ax1.set_xlabel('Classes', fontsize=12)
        ax1.set_ylabel('Error Rate (%)', fontsize=12)
        ax1.set_title('(a) Class-wise Error Analysis', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 错误类型分布
        error_types = list(error_data['error_types'].keys())
        error_counts = list(error_data['error_types'].values())
        
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['success'], 
                 self.colors['warning']]
        
        wedges, texts, autotexts = ax2.pie(error_counts, labels=error_types, autopct='%1.1f%%',
                                          colors=colors[:len(error_types)], startangle=90)
        ax2.set_title('(b) Error Type Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_confusion_matrix_figure(self, confusion_matrix, class_names, save_name="confusion_matrix"):
        """创建混淆矩阵图"""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 计算百分比
        cm_percent = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        # 创建热力图
        im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # 设置标签
        ax.set(xticks=np.arange(cm_percent.shape[1]),
               yticks=np.arange(cm_percent.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title='Confusion Matrix (%)',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # 添加数值标签
        thresh = cm_percent.max() / 2.
        for i in range(cm_percent.shape[0]):
            for j in range(cm_percent.shape[1]):
                ax.text(j, i, f'{cm_percent[i, j]:.1f}%',
                       ha="center", va="center",
                       color="white" if cm_percent[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """示例用法"""
    visualizer = PaperVisualization("paper_figures")
    
    # 示例数据
    metrics_data = {
        'metrics_history': {
            'train_loss': [2.5, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7],
            'val_loss': [2.6, 2.1, 1.9, 1.7, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8],
            'train_miou': [45, 52, 58, 63, 67, 71, 74, 76, 78, 79],
            'val_miou': [44, 51, 57, 62, 66, 70, 73, 75, 77, 78],
            'train_acc': [65, 72, 78, 83, 87, 90, 92, 94, 95, 96],
            'val_acc': [64, 71, 77, 82, 86, 89, 91, 93, 94, 95],
            'learning_rate': [0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
        }
    }
    
    # 生成训练曲线图
    visualizer.create_training_curves_figure(metrics_data)

if __name__ == "__main__":
    main()


