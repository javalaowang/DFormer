"""
论文结果分析脚本
自动从训练日志中提取指标并生成论文表格

使用方法:
python scripts/analyze_paper_results.py --log_dir checkpoints/
"""

import argparse
import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


class PaperResultsAnalyzer:
    def __init__(self, log_dir="checkpoints"):
        self.log_dir = Path(log_dir)
        self.results = {}
    
    def parse_log_file(self, log_path: Path) -> Dict:
        """解析训练日志文件"""
        results = {
            'best_miou': 0.0,
            'final_miou': 0.0,
            'final_loss': 0.0,
            'consistency_loss': [],
            'epochs': []
        }
        
        with open(log_path, 'r') as f:
            for line in f:
                # 匹配mIoU结果
                miou_match = re.search(r'miou ([\d.]+) best ([\d.]+)', line)
                if miou_match:
                    results['final_miou'] = float(miou_match.group(1))
                    results['best_miou'] = float(miou_match.group(2))
                
                # 匹配一致性损失
                consis_match = re.search(r'avg_consistency_loss=([\d.]+)', line)
                if consis_match:
                    results['consistency_loss'].append(float(consis_match.group(1)))
        
        return results
    
    def scan_experiments(self):
        """扫描所有实验目录"""
        for exp_dir in self.log_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # 查找日志文件
            log_files = list(exp_dir.glob("log_*.log"))
            if not log_files:
                continue
            
            # 解析最新日志
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            results = self.parse_log_file(latest_log)
            
            # 提取实验名称
            exp_name = exp_dir.name
            self.results[exp_name] = results
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """生成对比表格"""
        data = []
        
        for exp_name, results in self.results.items():
            # 判断是baseline还是vCLR
            is_vclr = 'vCLR' in exp_name
            
            data.append({
                'Experiment': exp_name,
                'Type': 'vCLR' if is_vclr else 'Baseline',
                'Best mIoU': results['best_miou'],
                'Final mIoU': results['final_miou'],
                'Avg Consistency Loss': sum(results['consistency_loss']) / len(results['consistency_loss']) if results['consistency_loss'] else 0,
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('Best mIoU', ascending=False)
    
    def generate_ablation_table(self) -> pd.DataFrame:
        """生成消融实验表格"""
        # 这里需要根据实际的消融实验命名规则来解析
        # 示例：
        ablation_data = []
        
        components = {
            'baseline': {'view': False, 'consistency': False, 'alignment': False},
            'vCLR_full': {'view': True, 'consistency': True, 'alignment': True},
            # ... 其他配置
        }
        
        for config_name, components_dict in components.items():
            if config_name in self.results:
                ablation_data.append({
                    'Configuration': config_name,
                    'Multi-View': '✓' if components_dict['view'] else '✗',
                    'Consistency': '✓' if components_dict['consistency'] else '✗',
                    'Alignment': '✓' if components_dict['alignment'] else '✗',
                    'mIoU': self.results[config_name]['best_miou'],
                    'Δ mIoU': 0  # 需要计算
                })
        
        return pd.DataFrame(ablation_data)
    
    def export_to_latex(self, df: pd.DataFrame, output_path: Path):
        """导出为LaTeX表格"""
        latex_str = df.to_latex(
            index=False,
            float_format="%.2f",
            caption="Experiment Results",
            label="tab:results"
        )
        
        with open(output_path, 'w') as f:
            f.write(latex_str)
    
    def generate_summary_report(self):
        """生成汇总报告"""
        # 扫描实验
        self.scan_experiments()
        
        # 生成表格
        comparison_df = self.generate_comparison_table()
        ablation_df = self.generate_ablation_table()
        
        # 保存
        output_dir = Path("paper_output")
        output_dir.mkdir(exist_ok=True)
        
        comparison_df.to_csv(output_dir / "comparison_table.csv", index=False)
        ablation_df.to_csv(output_dir / "ablation_table.csv", index=False)
        
        self.export_to_latex(comparison_df, output_dir / "comparison_table.tex")
        self.export_to_latex(ablation_df, output_dir / "ablation_table.tex")
        
        print(f"✓ Results exported to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze paper results")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="checkpoints",
        help="Directory containing experiment logs"
    )
    
    args = parser.parse_args()
    
    analyzer = PaperResultsAnalyzer(log_dir=args.log_dir)
    analyzer.generate_summary_report()


if __name__ == "__main__":
    main()

