"""
综合性实验运行脚本
用于自动化运行所有论文需要的实验

使用方法:
python scripts/run_comprehensive_experiments.py --experiment_type ablation
python scripts/run_comprehensive_experiments.py --experiment_type multi_dataset
python scripts/run_comprehensive_experiments.py --experiment_type robustness
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

class ExperimentRunner:
    def __init__(self, output_dir="experiments_comprehensive"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def run_experiment(self, config_name, experiment_name, gpu_id=0):
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name}")
        print(f"Config: {config_name}")
        print(f"{'='*60}\n")
        
        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 构建命令
        cmd = [
            "python", "utils/train.py",
            f"--config={config_name}",
            "--gpus=1",
            "--no-sliding",
            "--syncbn",
            "--mst",
            "--val_amp",
        ]
        
        # 运行实验
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            elapsed = time.time() - start_time
            
            # 提取结果（从日志中）
            # 这里需要根据实际日志格式解析
            self.results[experiment_name] = {
                'status': 'success',
                'time': elapsed,
                'config': config_name
            }
            
            print(f"✓ Completed in {elapsed:.1f}s")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {e}")
            self.results[experiment_name] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_ablation_study(self):
        """运行消融实验"""
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)
        
        experiments = [
            # 基础实验
            ("local_configs.Wheatlodgingdata.DFormerv2_Large_pretrained", "baseline"),
            
            # 组件消融
            ("local_configs.Wheatlodgingdata.DFormerv2_Large_vCLR_no_alignment", "vCLR_no_alignment"),
            ("local_configs.Wheatlodgingdata.DFormerv2_Large_vCLR_no_consistency", "vCLR_no_consistency"),
            ("local_configs.Wheatlodgingdata.DFormerv2_Large_vCLR", "vCLR_full"),
            
            # 权重消融
            ("local_configs.Wheatlodgingdata.DFormerv2_Large_vCLR_weight_low", "vCLR_weight_0.05"),
            ("local_configs.Wheatlodgingdata.DFormerv2_Large_vCLR_weight_high", "vCLR_weight_0.2"),
        ]
        
        for config, name in experiments:
            if not os.path.exists(config.replace(".", "/") + ".py"):
                print(f"⚠️ Config {config} not found, skipping...")
                continue
            self.run_experiment(config, name)
    
    def run_multi_dataset(self):
        """多数据集实验"""
        print("\n" + "="*60)
        print("MULTI-DATASET EXPERIMENTS")
        print("="*60)
        
        datasets = [
            ("NYUDepthv2", "NYUDepth v2"),
            ("SUNRGBD", "SUN RGB-D"),
            ("Wheatlodgingdata", "Wheat Lodging"),
        ]
        
        for dataset_name, display_name in datasets:
            # Baseline
            baseline_config = f"local_configs.{dataset_name}.DFormerv2_Large_pretrained"
            self.run_experiment(
                baseline_config,
                f"{display_name}_baseline"
            )
            
            # vCLR
            vclr_config = f"local_configs.{dataset_name}.DFormerv2_Large_vCLR"
            self.run_experiment(
                vclr_config,
                f"{display_name}_vCLR"
            )
    
    def run_robustness_test(self):
        """鲁棒性测试"""
        print("\n" + "="*60)
        print("ROBUSTNESS EXPERIMENTS")
        print("="*60)
        
        # 不同增强强度
        augmentation_levels = ["low", "medium", "high"]
        
        for level in augmentation_levels:
            config = f"local_configs.Wheatlodgingdata.DFormerv2_Large_vCLR_aug_{level}"
            self.run_experiment(config, f"robustness_aug_{level}")
    
    def generate_report(self):
        """生成实验报告"""
        report_path = self.output_dir / "experiment_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Comprehensive Experiment Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Results Summary\n\n")
            f.write("| Experiment | Status | Time (s) | Config |\n")
            f.write("|------------|--------|----------|--------|\n")
            
            for name, result in self.results.items():
                status = result.get('status', 'unknown')
                elapsed = result.get('time', 0)
                config = result.get('config', 'N/A')
                f.write(f"| {name} | {status} | {elapsed:.1f} | {config} |\n")
        
        print(f"\n✓ Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive experiments")
    parser.add_argument(
        "--experiment_type",
        type=str,
        choices=["ablation", "multi_dataset", "robustness", "all"],
        default="all",
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments_comprehensive",
        help="Output directory for results"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use"
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    if args.experiment_type == "ablation" or args.experiment_type == "all":
        runner.run_ablation_study()
    
    if args.experiment_type == "multi_dataset" or args.experiment_type == "all":
        runner.run_multi_dataset()
    
    if args.experiment_type == "robustness" or args.experiment_type == "all":
        runner.run_robustness_test()
    
    runner.generate_report()


if __name__ == "__main__":
    main()

