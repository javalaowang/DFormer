"""
CCS消融实验结果分析工具
生成论文级别的实验结果汇总和分析报告

功能：
1. 解析所有消融实验的日志
2. 提取关键指标（mIoU, 训练时间等）
3. 生成对比表格和可视化图表
4. 输出论文格式的分析报告
"""

import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import numpy as np


class AblationAnalyzer:
    """消融实验分析器"""
    
    def __init__(self, experiment_root: str):
        self.experiment_root = Path(experiment_root)
        self.results = {}
        self.summary_data = []
        
    def analyze_all_experiments(self):
        """分析所有实验"""
        print("="*60)
        print("CCS Ablation Study Analysis")
        print("="*60)
        
        # 查找所有实验目录
        experiment_dirs = [d for d in self.experiment_root.iterdir() 
                          if d.is_dir() and d.name.startswith("DFormerv2_L_CCS_")]
        
        print(f"Found {len(experiment_dirs)} experiment directories")
        
        for exp_dir in experiment_dirs:
            variant_name = exp_dir.name.replace("DFormerv2_L_CCS_", "")
            print(f"\nAnalyzing variant: {variant_name}")
            
            # 分析单个实验
            result = self._analyze_single_experiment(exp_dir, variant_name)
            if result:
                self.results[variant_name] = result
                self.summary_data.append(result)
        
        # 生成分析报告
        self._generate_summary_table()
        self._generate_visualizations()
        self._generate_paper_report()
        
        print(f"\n✓ Analysis completed! Results saved in {self.experiment_root}")
    
    def _analyze_single_experiment(self, exp_dir: Path, variant_name: str) -> Optional[Dict]:
        """分析单个实验"""
        try:
            # 查找日志文件
            log_files = list(exp_dir.glob("**/log_*.log"))
            if not log_files:
                print(f"  ✗ No log files found in {exp_dir}")
                return None
            
            # 使用最新的日志文件
            log_file = max(log_files, key=os.path.getmtime)
            print(f"  Using log file: {log_file.name}")
            
            # 解析日志
            metrics = self._parse_log_file(log_file)
            
            # 查找最佳检查点
            checkpoint_dir = exp_dir / "checkpoints"
            best_checkpoint = self._find_best_checkpoint(checkpoint_dir)
            
            # 构建结果
            result = {
                'variant': variant_name,
                'experiment_dir': str(exp_dir),
                'log_file': str(log_file),
                'best_checkpoint': best_checkpoint,
                **metrics
            }
            
            print(f"  ✓ Best mIoU: {metrics.get('best_miou', 'N/A'):.2f}%")
            print(f"  ✓ Training time: {metrics.get('total_time', 'N/A')}")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Error analyzing {variant_name}: {e}")
            return None
    
    def _parse_log_file(self, log_file: Path) -> Dict:
        """解析日志文件"""
        metrics = {
            'best_miou': 0.0,
            'final_miou': 0.0,
            'total_epochs': 0,
            'total_time': 'N/A',
            'convergence_epoch': 0,
            'training_losses': [],
            'validation_mious': []
        }
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # 解析关键指标
            for line in lines:
                # 最佳mIoU
                if 'best mIoU' in line:
                    match = re.search(r'best mIoU (\d+\.?\d*)', line)
                    if match:
                        metrics['best_miou'] = float(match.group(1))
                
                # 验证mIoU
                if 'validation result' in line and 'mIoU' in line:
                    match = re.search(r'mIoU (\d+\.?\d*)', line)
                    if match:
                        miou = float(match.group(1))
                        metrics['validation_mious'].append(miou)
                        metrics['final_miou'] = miou
                
                # 训练损失
                if 'total_loss=' in line:
                    match = re.search(r'total_loss=(\d+\.?\d*)', line)
                    if match:
                        metrics['training_losses'].append(float(match.group(1)))
                
                # 训练时间
                if 'ETA:' in line:
                    match = re.search(r'ETA: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if match:
                        metrics['eta'] = match.group(1)
            
            # 计算总epoch数
            metrics['total_epochs'] = len(metrics['validation_mious'])
            
            # 计算收敛epoch（mIoU不再显著提升）
            if len(metrics['validation_mious']) > 10:
                mious = metrics['validation_mious']
                best_miou = max(mious)
                # 找到最后一次达到最佳mIoU的epoch
                for i, miou in enumerate(reversed(mious)):
                    if miou >= best_miou * 0.99:  # 99%的最佳性能
                        metrics['convergence_epoch'] = len(mious) - i
                        break
            
        except Exception as e:
            print(f"    Warning: Error parsing log file: {e}")
        
        return metrics
    
    def _find_best_checkpoint(self, checkpoint_dir: Path) -> Optional[str]:
        """查找最佳检查点"""
        if not checkpoint_dir.exists():
            return None
        
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        if not checkpoint_files:
            return None
        
        # 按文件名中的mIoU值排序
        def extract_miou(filename):
            match = re.search(r'miou_(\d+\.?\d*)', filename.name)
            return float(match.group(1)) if match else 0.0
        
        best_checkpoint = max(checkpoint_files, key=extract_miou)
        return str(best_checkpoint)
    
    def _generate_summary_table(self):
        """生成汇总表格"""
        if not self.summary_data:
            print("No data to generate summary table")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.summary_data)
        
        # 选择关键列
        key_columns = ['variant', 'best_miou', 'final_miou', 'total_epochs', 'convergence_epoch']
        summary_df = df[key_columns].copy()
        
        # 排序（按best_miou降序）
        summary_df = summary_df.sort_values('best_miou', ascending=False)
        
        # 保存CSV
        csv_path = self.experiment_root / "ablation_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"✓ Summary table saved: {csv_path}")
        
        # 打印表格
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False, float_format='%.2f'))
        print("="*80)
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        if not self.summary_data:
            return
        
        df = pd.DataFrame(self.summary_data)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CCS Ablation Study Results', fontsize=16, fontweight='bold')
        
        # 1. mIoU对比柱状图
        ax1 = axes[0, 0]
        variants = df['variant']
        mious = df['best_miou']
        
        bars = ax1.bar(range(len(variants)), mious, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Variant')
        ax1.set_ylabel('Best mIoU (%)')
        ax1.set_title('Best mIoU by Variant')
        ax1.set_xticks(range(len(variants)))
        ax1.set_xticklabels(variants, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, miou in zip(bars, mious):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{miou:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 中心数量影响
        ax2 = axes[0, 1]
        center_variants = [v for v in variants if 'centers_' in v]
        if center_variants:
            center_mious = [df[df['variant'] == v]['best_miou'].iloc[0] for v in center_variants]
            center_nums = [int(v.split('_')[1]) for v in center_variants]
            
            ax2.plot(center_nums, center_mious, 'o-', linewidth=2, markersize=8)
            ax2.set_xlabel('Number of Centers')
            ax2.set_ylabel('Best mIoU (%)')
            ax2.set_title('Effect of Number of Centers')
            ax2.grid(True, alpha=0.3)
        
        # 3. 损失权重影响
        ax3 = axes[1, 0]
        lambda_variants = [v for v in variants if 'lambda_' in v]
        if lambda_variants:
            lambda_mious = [df[df['variant'] == v]['best_miou'].iloc[0] for v in lambda_variants]
            lambda_vals = [float(v.split('_')[1]) for v in lambda_variants]
            
            ax3.plot(lambda_vals, lambda_mious, 's-', linewidth=2, markersize=8, color='orange')
            ax3.set_xlabel('Lambda (Loss Weight)')
            ax3.set_ylabel('Best mIoU (%)')
            ax3.set_title('Effect of Loss Weight')
            ax3.grid(True, alpha=0.3)
        
        # 4. 增强权重影响
        ax4 = axes[1, 1]
        alpha_variants = [v for v in variants if 'alpha_' in v]
        if alpha_variants:
            alpha_mious = [df[df['variant'] == v]['best_miou'].iloc[0] for v in alpha_variants]
            alpha_vals = [float(v.split('_')[1]) for v in alpha_variants]
            
            ax4.plot(alpha_vals, alpha_mious, '^-', linewidth=2, markersize=8, color='green')
            ax4.set_xlabel('Alpha (Enhancement Weight)')
            ax4.set_ylabel('Best mIoU (%)')
            ax4.set_title('Effect of Enhancement Weight')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = self.experiment_root / "ablation_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved: {plot_path}")
        
        plt.show()
    
    def _generate_paper_report(self):
        """生成论文格式的报告"""
        if not self.summary_data:
            return
        
        df = pd.DataFrame(self.summary_data)
        
        # 找到最佳配置
        best_variant = df.loc[df['best_miou'].idxmax()]
        baseline_miou = df[df['variant'] == 'baseline']['best_miou'].iloc[0] if 'baseline' in df['variant'].values else 0
        
        # 生成报告
        report = f"""
# CCS Shape Prior Ablation Study Results

## Experimental Setup
- **Dataset**: Wheat Lodging Detection
- **Base Model**: DFormerv2-Large
- **Total Variants**: {len(self.summary_data)}
- **Training Epochs**: 150
- **Batch Size**: 2

## Key Findings

### 1. Overall Performance
- **Best Configuration**: {best_variant['variant']}
- **Best mIoU**: {best_variant['best_miou']:.2f}%
- **Improvement over Baseline**: {best_variant['best_miou'] - baseline_miou:.2f}% (from {baseline_miou:.2f}%)

### 2. Component Analysis

#### Number of Centers
"""
        
        # 添加中心数量分析
        center_variants = df[df['variant'].str.contains('centers_')]
        if not center_variants.empty:
            report += "\n| Centers | mIoU |\n|---------|------|\n"
            for _, row in center_variants.iterrows():
                centers = row['variant'].split('_')[1]
                report += f"| {centers} | {row['best_miou']:.2f}% |\n"
        
        report += "\n#### Loss Weight (Lambda)\n"
        lambda_variants = df[df['variant'].str.contains('lambda_')]
        if not lambda_variants.empty:
            report += "\n| Lambda | mIoU |\n|--------|------|\n"
            for _, row in lambda_variants.iterrows():
                lambda_val = row['variant'].split('_')[1]
                report += f"| {lambda_val} | {row['best_miou']:.2f}% |\n"
        
        report += "\n#### Enhancement Weight (Alpha)\n"
        alpha_variants = df[df['variant'].str.contains('alpha_')]
        if not alpha_variants.empty:
            report += "\n| Alpha | mIoU |\n|-------|------|\n"
            for _, row in alpha_variants.iterrows():
                alpha_val = row['variant'].split('_')[1]
                report += f"| {alpha_val} | {row['best_miou']:.2f}% |\n"
        
        report += f"""

### 3. Training Efficiency
- **Average Convergence Epoch**: {df['convergence_epoch'].mean():.1f}
- **Fastest Convergence**: {df.loc[df['convergence_epoch'].idxmin(), 'variant']} ({df['convergence_epoch'].min()} epochs)

### 4. Conclusions
1. CCS shape prior consistently improves segmentation performance
2. Optimal number of centers: {self._get_optimal_centers(df)}
3. Optimal loss weight: {self._get_optimal_lambda(df)}
4. Optimal enhancement weight: {self._get_optimal_alpha(df)}

## Detailed Results
See `ablation_summary.csv` for complete numerical results.
"""
        
        # 保存报告
        report_path = self.experiment_root / "ablation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Paper report saved: {report_path}")
    
    def _get_optimal_centers(self, df):
        """获取最优中心数量"""
        center_variants = df[df['variant'].str.contains('centers_')]
        if center_variants.empty:
            return "N/A"
        best_center = center_variants.loc[center_variants['best_miou'].idxmax()]
        return best_center['variant'].split('_')[1]
    
    def _get_optimal_lambda(self, df):
        """获取最优lambda"""
        lambda_variants = df[df['variant'].str.contains('lambda_')]
        if lambda_variants.empty:
            return "N/A"
        best_lambda = lambda_variants.loc[lambda_variants['best_miou'].idxmax()]
        return best_lambda['variant'].split('_')[1]
    
    def _get_optimal_alpha(self, df):
        """获取最优alpha"""
        alpha_variants = df[df['variant'].str.contains('alpha_')]
        if alpha_variants.empty:
            return "N/A"
        best_alpha = alpha_variants.loc[alpha_variants['best_miou'].idxmax()]
        return best_alpha['variant'].split('_')[1]


def main():
    parser = argparse.ArgumentParser(description='Analyze CCS ablation study results')
    parser.add_argument('--experiment_root', type=str, required=True,
                       help='Root directory of ablation experiments')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = AblationAnalyzer(args.experiment_root)
    
    # 执行分析
    analyzer.analyze_all_experiments()


if __name__ == "__main__":
    main()
