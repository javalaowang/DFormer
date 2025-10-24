import argparse
import datetime
import os
import pprint
import random
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from val_mm import evaluate, evaluate_msf

from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import configure_optimizers, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.pyt_utils import all_reduce_tensor

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

class EnhancedMetricsLogger:
    """增强的指标记录器，用于论文级别的分析"""
    
    def __init__(self, output_dir, experiment_name):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_miou': [],
            'val_miou': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': [],
            'memory_usage': []
        }
        self.best_metrics = {}
        self.start_time = time.time()
        
        # 创建输出目录
        os.makedirs(f"{output_dir}/metrics", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{output_dir}/analysis", exist_ok=True)
        
    def log_epoch_metrics(self, epoch, train_metrics, val_metrics, lr, epoch_time, memory_usage):
        """记录每个epoch的指标"""
        self.metrics_history['train_loss'].append(train_metrics.get('loss', 0))
        self.metrics_history['val_loss'].append(val_metrics.get('loss', 0))
        self.metrics_history['train_miou'].append(train_metrics.get('miou', 0))
        self.metrics_history['val_miou'].append(val_metrics.get('miou', 0))
        self.metrics_history['train_acc'].append(train_metrics.get('acc', 0))
        self.metrics_history['val_acc'].append(val_metrics.get('acc', 0))
        self.metrics_history['learning_rate'].append(lr)
        self.metrics_history['epoch_time'].append(epoch_time)
        self.metrics_history['memory_usage'].append(memory_usage)
        
        # 更新最佳指标
        if val_metrics.get('miou', 0) > self.best_metrics.get('best_miou', 0):
            self.best_metrics.update({
                'best_miou': val_metrics.get('miou', 0),
                'best_epoch': epoch,
                'best_train_loss': train_metrics.get('loss', 0),
                'best_val_loss': val_metrics.get('loss', 0)
            })
    
    def save_metrics_json(self):
        """保存指标为JSON格式"""
        metrics_data = {
            'experiment_name': self.experiment_name,
            'total_training_time': time.time() - self.start_time,
            'best_metrics': self.best_metrics,
            'metrics_history': self.metrics_history,
            'summary_statistics': self._calculate_summary_stats()
        }
        
        with open(f"{self.output_dir}/metrics/training_metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def _calculate_summary_stats(self):
        """计算汇总统计信息"""
        stats = {}
        for metric, values in self.metrics_history.items():
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1]
                }
        return stats
    
    def generate_training_curves(self):
        """生成训练曲线图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Analysis - {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # 损失曲线
        axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # mIoU曲线
        axes[0, 1].plot(self.metrics_history['train_miou'], label='Train mIoU', linewidth=2)
        axes[0, 1].plot(self.metrics_history['val_miou'], label='Val mIoU', linewidth=2)
        axes[0, 1].set_title('mIoU Curves', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mIoU (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 2].plot(self.metrics_history['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 2].plot(self.metrics_history['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 2].set_title('Accuracy Curves', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 学习率曲线
        axes[1, 0].plot(self.metrics_history['learning_rate'], linewidth=2, color='red')
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 训练时间
        axes[1, 1].plot(self.metrics_history['epoch_time'], linewidth=2, color='green')
        axes[1, 1].set_title('Epoch Training Time', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 内存使用
        axes[1, 2].plot(self.metrics_history['memory_usage'], linewidth=2, color='purple')
        axes[1, 2].set_title('Memory Usage', fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Memory (GB)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualizations/training_curves.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_dir}/analysis/paper_training_curves.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_performance_summary(self):
        """生成性能汇总表"""
        summary_data = []
        
        # 最佳性能
        summary_data.append({
            'Metric': 'Best mIoU',
            'Value': f"{self.best_metrics.get('best_miou', 0):.2f}%",
            'Epoch': self.best_metrics.get('best_epoch', 0)
        })
        
        # 最终性能
        if self.metrics_history['val_miou']:
            summary_data.append({
                'Metric': 'Final mIoU',
                'Value': f"{self.metrics_history['val_miou'][-1]:.2f}%",
                'Epoch': len(self.metrics_history['val_miou'])
            })
        
        # 训练时间
        total_time = time.time() - self.start_time
        summary_data.append({
            'Metric': 'Total Training Time',
            'Value': f"{total_time/3600:.2f} hours",
            'Epoch': '-'
        })
        
        # 平均epoch时间
        if self.metrics_history['epoch_time']:
            avg_epoch_time = np.mean(self.metrics_history['epoch_time'])
            summary_data.append({
                'Metric': 'Average Epoch Time',
                'Value': f"{avg_epoch_time:.2f} seconds",
                'Epoch': '-'
            })
        
        # 保存为CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(f"{self.output_dir}/analysis/performance_summary.csv", index=False)
        
        # 保存为LaTeX表格
        latex_table = df.to_latex(index=False, escape=False)
        with open(f"{self.output_dir}/analysis/performance_summary.tex", 'w') as f:
            f.write(latex_table)
        
        return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--gpus", default=2, type=int, help="used gpu number")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--epochs", default=0)
    parser.add_argument("--show_image", "-s", default=False, action="store_true")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--continue_fpath")
    parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mst", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--val_amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pad_SUNRGBD", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_seed", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--local-rank", default=0)
    
    # 增强参数
    parser.add_argument("--experiment_name", default="default_experiment")
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--enable_visualization", default=False, action="store_true")
    parser.add_argument("--enable_metrics_analysis", default=False, action="store_true")
    parser.add_argument("--enable_detailed_logging", default=False, action="store_true")
    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--val_interval", default=5, type=int)
    parser.add_argument("--log_interval", default=20, type=int)
    
    args = parser.parse_args()
    
    # 初始化增强指标记录器
    if args.enable_metrics_analysis:
        metrics_logger = EnhancedMetricsLogger(args.output_dir, args.experiment_name)
    
    # 这里应该调用原始的train.py逻辑，但添加增强的指标记录
    # 由于原始train.py比较复杂，这里提供一个框架
    print(f"Enhanced training started for experiment: {args.experiment_name}")
    print(f"Output directory: {args.output_dir}")
    
    # 模拟训练过程（实际应该调用原始训练逻辑）
    # 这里只是展示如何集成增强功能
    
    if args.enable_metrics_analysis:
        # 在训练循环中调用
        # metrics_logger.log_epoch_metrics(epoch, train_metrics, val_metrics, lr, epoch_time, memory_usage)
        
        # 训练完成后生成分析
        metrics_logger.save_metrics_json()
        metrics_logger.generate_training_curves()
        metrics_logger.generate_performance_summary()
        
        print("Enhanced analysis completed!")
        print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()


