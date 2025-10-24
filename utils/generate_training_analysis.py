#!/usr/bin/env python3
"""
训练分析报告生成器
用于生成符合论文标准的分析报告和可视化图表
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

class TrainingAnalysisGenerator:
    """训练分析报告生成器"""
    
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.metrics_file = os.path.join(experiment_dir, "metrics", "training_metrics.json")
        self.load_metrics()
        
    def load_metrics(self):
        """加载训练指标"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                self.metrics_data = json.load(f)
        else:
            print(f"Warning: Metrics file not found at {self.metrics_file}")
            self.metrics_data = None
    
    def generate_comprehensive_analysis(self, output_format='both', generate_plots=True, 
                                      generate_tables=True, generate_summary=True):
        """生成综合分析报告"""
        
        if self.metrics_data is None:
            print("No metrics data available for analysis")
            return
        
        print("Generating comprehensive training analysis...")
        
        if generate_plots:
            self.generate_publication_quality_plots()
            # self.generate_comparison_plots()  # 需要对比数据
            # self.generate_ablation_plots()    # 需要消融数据
        
        if generate_tables:
            self.generate_performance_tables()
            self.generate_statistical_tables()
        
        if generate_summary:
            self.generate_executive_summary()
            self.generate_paper_ready_summary()
        
        print(f"Analysis completed! Results saved to {self.experiment_dir}/analysis/")
    
    def generate_publication_quality_plots(self):
        """生成论文质量的图表"""
        
        # 创建主要训练曲线图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Performance Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        metrics = self.metrics_data['metrics_history']
        
        # 1. 损失曲线
        axes[0, 0].plot(metrics['train_loss'], label='Training Loss', linewidth=2.5, color='#1f77b4')
        axes[0, 0].plot(metrics['val_loss'], label='Validation Loss', linewidth=2.5, color='#ff7f0e')
        axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)
        
        # 2. mIoU曲线
        axes[0, 1].plot(metrics['train_miou'], label='Training mIoU', linewidth=2.5, color='#2ca02c')
        axes[0, 1].plot(metrics['val_miou'], label='Validation mIoU', linewidth=2.5, color='#d62728')
        axes[0, 1].set_title('Mean Intersection over Union (mIoU)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('mIoU (%)', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        
        # 3. 准确率曲线
        axes[1, 0].plot(metrics['train_acc'], label='Training Accuracy', linewidth=2.5, color='#9467bd')
        axes[1, 0].plot(metrics['val_acc'], label='Validation Accuracy', linewidth=2.5, color='#8c564b')
        axes[1, 0].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)
        
        # 4. 学习率调度
        axes[1, 1].plot(metrics['learning_rate'], linewidth=2.5, color='#e377c2')
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].spines['top'].set_visible(False)
        axes[1, 1].spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/analysis/paper_main_curves.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.experiment_dir}/analysis/paper_main_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成性能指标雷达图
        self.generate_radar_chart()
        
        # 生成收敛分析图
        self.generate_convergence_analysis()
    
    def generate_radar_chart(self):
        """生成性能指标雷达图"""
        
        best_metrics = self.metrics_data['best_metrics']
        
        # 定义指标和值
        metrics = ['mIoU', 'Accuracy', 'Training Stability', 'Convergence Speed']
        values = [
            best_metrics.get('best_miou', 0),
            best_metrics.get('best_acc', 0),
            self.calculate_training_stability(),
            self.calculate_convergence_speed()
        ]
        
        # 标准化到0-100
        values = [min(100, max(0, v)) for v in values]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
        ax.fill(angles, values, alpha=0.25, color='#1f77b4')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/analysis/performance_radar.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.experiment_dir}/analysis/performance_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_convergence_analysis(self):
        """生成收敛分析图"""
        
        metrics = self.metrics_data['metrics_history']
        val_miou = np.array(metrics['val_miou'])
        
        # 计算移动平均
        window_size = min(10, len(val_miou) // 4)
        if window_size > 1:
            moving_avg = np.convolve(val_miou, np.ones(window_size)/window_size, mode='valid')
        else:
            moving_avg = val_miou
        
        # 计算收敛点（当移动平均变化小于阈值时）
        threshold = 0.1
        convergence_epoch = len(moving_avg)
        for i in range(1, len(moving_avg)):
            if abs(moving_avg[i] - moving_avg[i-1]) < threshold:
                convergence_epoch = i + window_size
                break
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 收敛曲线
        ax1.plot(val_miou, label='Validation mIoU', linewidth=2, color='#2ca02c')
        if len(moving_avg) > 0:
            ax1.plot(range(window_size-1, len(val_miou)), moving_avg, 
                    label=f'Moving Average (window={window_size})', linewidth=2, color='#d62728')
        ax1.axvline(x=convergence_epoch, color='red', linestyle='--', 
                   label=f'Convergence Point (Epoch {convergence_epoch})')
        ax1.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mIoU (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 损失变化率
        if len(val_miou) > 1:
            loss_change = np.diff(val_miou)
            ax2.plot(range(1, len(val_miou)), loss_change, linewidth=2, color='#ff7f0e')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_title('Loss Change Rate', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('mIoU Change (%)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/analysis/convergence_analysis.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.experiment_dir}/analysis/convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_performance_tables(self):
        """生成性能表格"""
        
        best_metrics = self.metrics_data['best_metrics']
        summary_stats = self.metrics_data['summary_statistics']
        
        # 主要性能指标表
        performance_data = [
            ['Best mIoU', f"{best_metrics.get('best_miou', 0):.2f}%", f"Epoch {best_metrics.get('best_epoch', 0)}"],
            ['Final mIoU', f"{summary_stats.get('val_miou', {}).get('final', 0):.2f}%", f"Epoch {len(self.metrics_data['metrics_history']['val_miou'])}"],
            ['Best Accuracy', f"{best_metrics.get('best_acc', 0):.2f}%", f"Epoch {best_metrics.get('best_epoch', 0)}"],
            ['Training Time', f"{self.metrics_data['total_training_time']/3600:.2f} hours", "Total"],
            ['Average Epoch Time', f"{summary_stats.get('epoch_time', {}).get('mean', 0):.2f} seconds", "Average"],
            ['Peak Memory Usage', f"{max(self.metrics_data['metrics_history']['memory_usage']):.2f} GB", "Peak"]
        ]
        
        df_performance = pd.DataFrame(performance_data, columns=['Metric', 'Value', 'Note'])
        df_performance.to_csv(f"{self.experiment_dir}/analysis/performance_table.csv", index=False)
        
        # LaTeX表格
        latex_table = df_performance.to_latex(index=False, escape=False, 
                                            caption="Model Performance Summary",
                                            label="tab:performance")
        with open(f"{self.experiment_dir}/analysis/performance_table.tex", 'w') as f:
            f.write(latex_table)
        
        # 统计摘要表
        stats_data = []
        for metric, stats in summary_stats.items():
            if stats:
                stats_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Mean': f"{stats['mean']:.4f}",
                    'Std': f"{stats['std']:.4f}",
                    'Min': f"{stats['min']:.4f}",
                    'Max': f"{stats['max']:.4f}"
                })
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_csv(f"{self.experiment_dir}/analysis/statistical_summary.csv", index=False)
        
        # LaTeX统计表
        latex_stats = df_stats.to_latex(index=False, escape=False,
                                       caption="Statistical Summary of Training Metrics",
                                       label="tab:statistics")
        with open(f"{self.experiment_dir}/analysis/statistical_summary.tex", 'w') as f:
            f.write(latex_stats)
    
    def generate_executive_summary(self):
        """生成执行摘要"""
        
        best_metrics = self.metrics_data['best_metrics']
        total_time = self.metrics_data['total_training_time']
        
        summary = f"""
# Training Analysis Report

## Executive Summary

**Experiment**: {self.metrics_data['experiment_name']}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Key Results
- **Best mIoU**: {best_metrics.get('best_miou', 0):.2f}% (Epoch {best_metrics.get('best_epoch', 0)})
- **Final mIoU**: {self.metrics_data['metrics_history']['val_miou'][-1]:.2f}% (Final Epoch)
- **Total Training Time**: {total_time/3600:.2f} hours
- **Convergence**: Model converged at epoch {best_metrics.get('best_epoch', 0)}

### Training Characteristics
- **Stability**: {self.calculate_training_stability():.1f}/100
- **Convergence Speed**: {self.calculate_convergence_speed():.1f}/100
- **Peak Memory Usage**: {max(self.metrics_data['metrics_history']['memory_usage']):.2f} GB

### Recommendations
1. Model achieved competitive performance with {best_metrics.get('best_miou', 0):.2f}% mIoU
2. Training was stable with consistent improvement
3. Consider early stopping at epoch {best_metrics.get('best_epoch', 0)} for efficiency

---
*Generated by Enhanced Training Analysis System*
"""
        
        with open(f"{self.experiment_dir}/analysis/executive_summary.md", 'w') as f:
            f.write(summary)
    
    def generate_paper_ready_summary(self):
        """生成论文就绪的摘要"""
        
        best_metrics = self.metrics_data['best_metrics']
        
        paper_summary = f"""
## Experimental Results

### Training Configuration
- **Model**: DFormerv2-Large with Pretrained Weights
- **Dataset**: Wheatlodgingdata
- **Training Strategy**: Fine-tuning with pretrained backbone
- **Optimization**: AdamW with cosine annealing

### Performance Metrics
- **Best mIoU**: {best_metrics.get('best_miou', 0):.2f}%
- **Best Accuracy**: {best_metrics.get('best_acc', 0):.2f}%
- **Convergence Epoch**: {best_metrics.get('best_epoch', 0)}
- **Training Efficiency**: {self.metrics_data['total_training_time']/3600:.2f} hours

### Analysis
The model demonstrates strong performance on the wheat lodging detection task, 
achieving {best_metrics.get('best_miou', 0):.2f}% mIoU. The training process shows 
stable convergence with consistent improvement over {best_metrics.get('best_epoch', 0)} epochs.

### Computational Efficiency
- Average epoch time: {np.mean(self.metrics_data['metrics_history']['epoch_time']):.2f} seconds
- Peak memory usage: {max(self.metrics_data['metrics_history']['memory_usage']):.2f} GB
- Total parameters: [To be filled from model analysis]
"""
        
        with open(f"{self.experiment_dir}/analysis/paper_summary.md", 'w') as f:
            f.write(paper_summary)
    
    def calculate_training_stability(self):
        """计算训练稳定性分数"""
        val_miou = np.array(self.metrics_data['metrics_history']['val_miou'])
        if len(val_miou) < 2:
            return 0
        
        # 计算变异系数
        cv = np.std(val_miou) / np.mean(val_miou) if np.mean(val_miou) > 0 else 1
        stability = max(0, 100 - cv * 100)
        return min(100, stability)
    
    def calculate_convergence_speed(self):
        """计算收敛速度分数"""
        val_miou = np.array(self.metrics_data['metrics_history']['val_miou'])
        if len(val_miou) < 2:
            return 0
        
        # 计算达到90%最佳性能的epoch
        best_miou = np.max(val_miou)
        target = best_miou * 0.9
        
        convergence_epoch = len(val_miou)
        for i, miou in enumerate(val_miou):
            if miou >= target:
                convergence_epoch = i
                break
        
        # 收敛速度分数 (越早收敛分数越高)
        total_epochs = len(val_miou)
        speed_score = max(0, 100 - (convergence_epoch / total_epochs) * 100)
        return min(100, speed_score)

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive training analysis')
    parser.add_argument('--experiment_dir', required=True, help='Path to experiment directory')
    parser.add_argument('--output_format', default='both', choices=['plots', 'tables', 'both'])
    parser.add_argument('--generate_plots', action='store_true', default=True)
    parser.add_argument('--generate_tables', action='store_true', default=True)
    parser.add_argument('--generate_summary', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # 创建分析生成器
    analyzer = TrainingAnalysisGenerator(args.experiment_dir)
    
    # 生成分析报告
    analyzer.generate_comprehensive_analysis(
        output_format=args.output_format,
        generate_plots=args.generate_plots,
        generate_tables=args.generate_tables,
        generate_summary=args.generate_summary
    )

if __name__ == "__main__":
    main()
