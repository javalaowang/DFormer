#!/usr/bin/env python3
"""
简化的训练分析报告生成器
用于处理实际训练后的数据并生成论文级别的分析
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import re

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 设置seaborn样式
sns.set_style("whitegrid")

class SimpleTrainingAnalysis:
    """简化的训练分析器"""
    
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.log_files = self.find_log_files()
        self.metrics_data = self.parse_log_files()
        
    def find_log_files(self):
        """查找训练日志文件"""
        log_patterns = [
            f"{self.experiment_dir}/checkpoints/*/log_*.log",
            f"{self.experiment_dir}/logs/*.log",
            "checkpoints/*/log_*.log"  # 也检查全局checkpoints
        ]
        
        log_files = []
        for pattern in log_patterns:
            log_files.extend(glob.glob(pattern))
        
        return log_files
    
    def parse_log_files(self):
        """解析日志文件提取指标"""
        metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_miou': [],
            'val_miou': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        for log_file in self.log_files:
            print(f"Parsing log file: {log_file}")
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    self.extract_metrics_from_content(content, metrics)
            except Exception as e:
                print(f"Error parsing {log_file}: {e}")
        
        return metrics
    
    def extract_metrics_from_content(self, content, metrics):
        """从日志内容中提取指标"""
        lines = content.split('\n')
        
        for line in lines:
            # 提取epoch信息
            epoch_match = re.search(r'epoch[:\s]+(\d+)', line, re.IGNORECASE)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                if epoch not in metrics['epochs']:
                    metrics['epochs'].append(epoch)
            
            # 提取训练损失
            train_loss_match = re.search(r'train.*loss[:\s]+([\d.]+)', line, re.IGNORECASE)
            if train_loss_match:
                metrics['train_loss'].append(float(train_loss_match.group(1)))
            
            # 提取验证损失
            val_loss_match = re.search(r'val.*loss[:\s]+([\d.]+)', line, re.IGNORECASE)
            if val_loss_match:
                metrics['val_loss'].append(float(val_loss_match.group(1)))
            
            # 提取mIoU
            miou_match = re.search(r'miou[:\s]+([\d.]+)', line, re.IGNORECASE)
            if miou_match:
                miou_val = float(miou_match.group(1))
                if 'val' in line.lower():
                    metrics['val_miou'].append(miou_val)
                elif 'train' in line.lower():
                    metrics['train_miou'].append(miou_val)
            
            # 提取准确率
            acc_match = re.search(r'acc[:\s]+([\d.]+)', line, re.IGNORECASE)
            if acc_match:
                acc_val = float(acc_match.group(1))
                if 'val' in line.lower():
                    metrics['val_acc'].append(acc_val)
                elif 'train' in line.lower():
                    metrics['train_acc'].append(acc_val)
            
            # 提取学习率
            lr_match = re.search(r'lr[:\s]+([\d.e-]+)', line, re.IGNORECASE)
            if lr_match:
                metrics['learning_rate'].append(float(lr_match.group(1)))
    
    def generate_analysis(self):
        """生成分析报告"""
        if not any(self.metrics_data.values()):
            print("No metrics found in log files")
            return
        
        print("Generating analysis from training logs...")
        
        # 创建输出目录
        os.makedirs(f"{self.experiment_dir}/analysis", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/visualizations", exist_ok=True)
        
        # 生成训练曲线图
        self.generate_training_curves()
        
        # 生成性能汇总
        self.generate_performance_summary()
        
        # 生成分析报告
        self.generate_analysis_report()
        
        print(f"Analysis completed! Results saved to {self.experiment_dir}/analysis/")
    
    def generate_training_curves(self):
        """生成训练曲线图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Performance Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # 1. 损失曲线
        if self.metrics_data['train_loss'] and self.metrics_data['val_loss']:
            ax1 = axes[0, 0]
            epochs = range(len(self.metrics_data['train_loss']))
            ax1.plot(epochs, self.metrics_data['train_loss'], label='Training Loss', 
                    linewidth=2.5, color='#1f77b4', marker='o', markersize=4, markevery=max(1, len(epochs)//10))
            ax1.plot(epochs, self.metrics_data['val_loss'], label='Validation Loss', 
                    linewidth=2.5, color='#ff7f0e', marker='s', markersize=4, markevery=max(1, len(epochs)//10))
            ax1.set_title('(a) Loss Curves', fontsize=14, fontweight='bold', pad=15)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
        
        # 2. mIoU曲线
        if self.metrics_data['train_miou'] and self.metrics_data['val_miou']:
            ax2 = axes[0, 1]
            epochs = range(len(self.metrics_data['train_miou']))
            ax2.plot(epochs, self.metrics_data['train_miou'], label='Training mIoU', 
                    linewidth=2.5, color='#2ca02c', marker='o', markersize=4, markevery=max(1, len(epochs)//10))
            ax2.plot(epochs, self.metrics_data['val_miou'], label='Validation mIoU', 
                    linewidth=2.5, color='#d62728', marker='s', markersize=4, markevery=max(1, len(epochs)//10))
            ax2.set_title('(b) mIoU Curves', fontsize=14, fontweight='bold', pad=15)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('mIoU (%)', fontsize=12)
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
        
        # 3. 准确率曲线
        if self.metrics_data['train_acc'] and self.metrics_data['val_acc']:
            ax3 = axes[1, 0]
            epochs = range(len(self.metrics_data['train_acc']))
            ax3.plot(epochs, self.metrics_data['train_acc'], label='Training Accuracy', 
                    linewidth=2.5, color='#9467bd', marker='o', markersize=4, markevery=max(1, len(epochs)//10))
            ax3.plot(epochs, self.metrics_data['val_acc'], label='Validation Accuracy', 
                    linewidth=2.5, color='#8c564b', marker='s', markersize=4, markevery=max(1, len(epochs)//10))
            ax3.set_title('(c) Accuracy Curves', fontsize=14, fontweight='bold', pad=15)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Accuracy (%)', fontsize=12)
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)
        
        # 4. 学习率曲线
        if self.metrics_data['learning_rate']:
            ax4 = axes[1, 1]
            epochs = range(len(self.metrics_data['learning_rate']))
            ax4.plot(epochs, self.metrics_data['learning_rate'], 
                    linewidth=2.5, color='#e377c2', marker='o', markersize=4, markevery=max(1, len(epochs)//10))
            ax4.set_title('(d) Learning Rate Schedule', fontsize=14, fontweight='bold', pad=15)
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Learning Rate', fontsize=12)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/analysis/training_curves.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.experiment_dir}/analysis/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_performance_summary(self):
        """生成性能汇总"""
        summary_data = []
        
        # 最佳mIoU
        if self.metrics_data['val_miou']:
            best_miou = max(self.metrics_data['val_miou'])
            best_epoch = self.metrics_data['val_miou'].index(best_miou) + 1
            summary_data.append(['Best mIoU', f"{best_miou:.2f}%", f"Epoch {best_epoch}"])
        
        # 最终mIoU
        if self.metrics_data['val_miou']:
            final_miou = self.metrics_data['val_miou'][-1]
            summary_data.append(['Final mIoU', f"{final_miou:.2f}%", f"Epoch {len(self.metrics_data['val_miou'])}"])
        
        # 最佳准确率
        if self.metrics_data['val_acc']:
            best_acc = max(self.metrics_data['val_acc'])
            best_epoch = self.metrics_data['val_acc'].index(best_acc) + 1
            summary_data.append(['Best Accuracy', f"{best_acc:.2f}%", f"Epoch {best_epoch}"])
        
        # 训练稳定性
        if self.metrics_data['val_miou']:
            stability = self.calculate_stability(self.metrics_data['val_miou'])
            summary_data.append(['Training Stability', f"{stability:.1f}/100", "CV-based"])
        
        # 保存为CSV
        if summary_data:
            df = pd.DataFrame(summary_data, columns=['Metric', 'Value', 'Note'])
            df.to_csv(f"{self.experiment_dir}/analysis/performance_summary.csv", index=False)
            
            # LaTeX表格
            latex_table = df.to_latex(index=False, escape=False, 
                                    caption="Model Performance Summary",
                                    label="tab:performance")
            with open(f"{self.experiment_dir}/analysis/performance_summary.tex", 'w') as f:
                f.write(latex_table)
    
    def calculate_stability(self, values):
        """计算训练稳定性"""
        if len(values) < 2:
            return 0
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1
        stability = max(0, 100 - cv * 100)
        return min(100, stability)
    
    def generate_analysis_report(self):
        """生成分析报告"""
        report = f"""
# Training Analysis Report

## Experiment Summary
- **Experiment Directory**: {self.experiment_dir}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Log Files Found**: {len(self.log_files)}

## Key Results
"""
        
        if self.metrics_data['val_miou']:
            best_miou = max(self.metrics_data['val_miou'])
            best_epoch = self.metrics_data['val_miou'].index(best_miou) + 1
            report += f"- **Best mIoU**: {best_miou:.2f}% (Epoch {best_epoch})\n"
        
        if self.metrics_data['val_miou']:
            final_miou = self.metrics_data['val_miou'][-1]
            report += f"- **Final mIoU**: {final_miou:.2f}% (Final Epoch)\n"
        
        if self.metrics_data['val_acc']:
            best_acc = max(self.metrics_data['val_acc'])
            report += f"- **Best Accuracy**: {best_acc:.2f}%\n"
        
        if self.metrics_data['val_miou']:
            stability = self.calculate_stability(self.metrics_data['val_miou'])
            report += f"- **Training Stability**: {stability:.1f}/100\n"
        
        report += f"""
## Training Characteristics
- **Total Epochs**: {len(self.metrics_data['val_miou']) if self.metrics_data['val_miou'] else 'Unknown'}
- **Log Files**: {', '.join([os.path.basename(f) for f in self.log_files])}

## Generated Files
- Training curves: `analysis/training_curves.pdf`
- Performance summary: `analysis/performance_summary.csv`
- LaTeX table: `analysis/performance_summary.tex`

---
*Generated by Simple Training Analysis System*
"""
        
        with open(f"{self.experiment_dir}/analysis/analysis_report.md", 'w') as f:
            f.write(report)

def main():
    parser = argparse.ArgumentParser(description='Generate simple training analysis from log files')
    parser.add_argument('--experiment_dir', required=True, help='Path to experiment directory')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = SimpleTrainingAnalysis(args.experiment_dir)
    
    # 生成分析报告
    analyzer.generate_analysis()

if __name__ == "__main__":
    main()


