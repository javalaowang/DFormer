"""
小麦倒伏分割专门实验方案
Wheat Lodging Segmentation Specialized Experiment

实验设计：
1. 基线对比实验
2. 形状先验类型对比
3. 参数敏感性分析
4. 混合策略优化
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import sys


class WheatLodgingExperiment:
    """小麦倒伏分割实验管理器"""
    
    def __init__(self, base_dir: str = "experiments/wheat_lodging"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验配置
        self.experiments = self._define_experiments()
        
        # 结果记录
        self.results = {}
        
    def _define_experiments(self) -> Dict[str, Dict]:
        """定义所有实验配置"""
        return {
            # 阶段1: 基线对比实验
            "stage1_baseline": {
                "name": "基线对比实验",
                "description": "建立性能基准，验证形状先验的必要性",
                "experiments": {
                    "baseline": {
                        "use_shape_prior": False,
                        "description": "原始DFormer，无形状约束",
                        "config_file": "DFormerv2_L_Baseline.py"
                    },
                    "ccs_star": {
                        "use_shape_prior": True,
                        "shape_type": "star",
                        "num_centers": 5,
                        "temperature": 1.0,
                        "variational_weight": 0.1,
                        "shape_lambda": 0.1,
                        "description": "星形形状先验",
                        "config_file": "DFormerv2_L_CCS_Star.py"
                    },
                    "wheat_bar": {
                        "use_shape_prior": True,
                        "shape_type": "bar",
                        "num_centers": 3,
                        "temperature": 1.0,
                        "variational_weight": 0.05,
                        "shape_lambda": 0.05,
                        "description": "条状形状先验（适合正常小麦）",
                        "config_file": "DFormerv2_L_Wheat_Bar.py"
                    },
                    "wheat_diffusion": {
                        "use_shape_prior": True,
                        "shape_type": "diffusion",
                        "num_centers": 3,
                        "temperature": 1.0,
                        "variational_weight": 0.05,
                        "shape_lambda": 0.05,
                        "description": "扩散形状先验（适合倒伏小麦）",
                        "config_file": "DFormerv2_L_Wheat_Diffusion.py"
                    }
                }
            },
            
            # 阶段2: 形状先验类型对比
            "stage2_shape_types": {
                "name": "形状先验类型对比",
                "description": "深入分析不同形状先验的适用性",
                "experiments": {
                    "bar_vertical": {
                        "shape_type": "bar",
                        "orientation": "vertical",
                        "num_centers": 3,
                        "description": "垂直条状（正常小麦）"
                    },
                    "bar_diagonal": {
                        "shape_type": "bar",
                        "orientation": "diagonal", 
                        "num_centers": 3,
                        "description": "对角条状（倾斜小麦）"
                    },
                    "bar_learnable": {
                        "shape_type": "bar",
                        "orientation": "learnable",
                        "num_centers": 3,
                        "description": "学习条状方向"
                    },
                    "diffusion_small": {
                        "shape_type": "diffusion",
                        "radius_range": [10, 30],
                        "num_centers": 3,
                        "description": "小范围扩散"
                    },
                    "diffusion_large": {
                        "shape_type": "diffusion",
                        "radius_range": [20, 60],
                        "num_centers": 3,
                        "description": "大范围扩散"
                    },
                    "diffusion_learnable": {
                        "shape_type": "diffusion",
                        "radius": "learnable",
                        "num_centers": 3,
                        "description": "学习扩散半径"
                    },
                    "mixed_adaptive": {
                        "shape_type": "mixed",
                        "bar_weight": "learnable",
                        "diffusion_weight": "learnable",
                        "num_centers": 3,
                        "description": "自适应混合形状"
                    }
                }
            },
            
            # 阶段3: 参数敏感性分析
            "stage3_parameter_analysis": {
                "name": "参数敏感性分析",
                "description": "找到最优参数组合",
                "experiments": {
                    # 中心数量分析
                    "centers_1": {"num_centers": 1, "description": "单中心"},
                    "centers_2": {"num_centers": 2, "description": "双中心"},
                    "centers_3": {"num_centers": 3, "description": "三中心"},
                    "centers_5": {"num_centers": 5, "description": "五中心"},
                    "centers_7": {"num_centers": 7, "description": "七中心"},
                    
                    # 权重参数分析
                    "weight_light": {"shape_lambda": 0.01, "variational_weight": 0.01},
                    "weight_medium": {"shape_lambda": 0.05, "variational_weight": 0.05},
                    "weight_strong": {"shape_lambda": 0.1, "variational_weight": 0.1},
                    "weight_very_strong": {"shape_lambda": 0.2, "variational_weight": 0.2},
                    
                    # 温度参数分析
                    "temp_sharp": {"temperature": 0.5, "description": "锐利形状"},
                    "temp_normal": {"temperature": 1.0, "description": "标准形状"},
                    "temp_smooth": {"temperature": 2.0, "description": "平滑形状"}
                }
            },
            
            # 阶段4: 混合策略优化
            "stage4_hybrid_optimization": {
                "name": "混合策略优化",
                "description": "结合多种策略，达到最佳性能",
                "experiments": {
                    "progressive_light": {
                        "strategy": "progressive",
                        "stages": [
                            {"epochs": 50, "shape_lambda": 0.0},
                            {"epochs": 50, "shape_lambda": 0.02},
                            {"epochs": 50, "shape_lambda": 0.05}
                        ]
                    },
                    "progressive_medium": {
                        "strategy": "progressive",
                        "stages": [
                            {"epochs": 50, "shape_lambda": 0.0},
                            {"epochs": 50, "shape_lambda": 0.05},
                            {"epochs": 50, "shape_lambda": 0.1}
                        ]
                    },
                    "multiscale": {
                        "strategy": "multiscale",
                        "scales": [
                            {"num_centers": 2, "radius_range": [5, 15]},
                            {"num_centers": 3, "radius_range": [10, 30]},
                            {"num_centers": 2, "radius_range": [20, 50]}
                        ]
                    }
                }
            }
        }
    
    def run_stage(self, stage_name: str, dry_run: bool = False):
        """运行指定阶段的实验"""
        if stage_name not in self.experiments:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        stage_config = self.experiments[stage_name]
        print(f"\n{'='*80}")
        print(f"Running Stage: {stage_config['name']}")
        print(f"Description: {stage_config['description']}")
        print(f"{'='*80}")
        
        stage_dir = self.base_dir / stage_name
        stage_dir.mkdir(exist_ok=True)
        
        results = {}
        
        for exp_name, exp_config in stage_config['experiments'].items():
            print(f"\n--- Running Experiment: {exp_name} ---")
            print(f"Config: {exp_config}")
            
            if dry_run:
                print("DRY RUN: Would run this experiment")
                results[exp_name] = {"status": "dry_run", "config": exp_config}
                continue
            
            # 创建实验目录
            exp_dir = stage_dir / exp_name
            exp_dir.mkdir(exist_ok=True)
            
            # 生成配置文件
            config_file = self._generate_config_file(exp_name, exp_config, exp_dir)
            
            # 运行训练
            try:
                result = self._run_training(exp_name, config_file, exp_dir)
                results[exp_name] = result
                print(f"✓ Experiment {exp_name} completed successfully")
            except Exception as e:
                print(f"✗ Experiment {exp_name} failed: {e}")
                results[exp_name] = {"status": "failed", "error": str(e)}
        
        # 保存阶段结果
        self.results[stage_name] = results
        self._save_results(stage_name, results)
        
        return results
    
    def _generate_config_file(self, exp_name: str, exp_config: Dict, exp_dir: Path) -> str:
        """生成实验配置文件"""
        config_content = self._create_config_content(exp_name, exp_config)
        
        config_file = exp_dir / f"{exp_name}_config.py"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        return str(config_file)
    
    def _create_config_content(self, exp_name: str, exp_config: Dict) -> str:
        """创建配置文件内容"""
        # 基础配置模板
        base_config = '''
from .._base_.datasets.Wheatlodgingdata import *

""" Settings for network """
C.backbone = "DFormerv2_L"
C.pretrained_model = "/root/DFormer/checkpoints/pretrained/DFormerv2_Large_pretrained.pth"
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

""" Wheat Lodging Shape Prior Settings """
C.use_shape_prior = {use_shape_prior}
C.shape_type = "{shape_type}"
C.num_centers = {num_centers}
C.temperature = {temperature}
C.variational_weight = {variational_weight}
C.shape_lambda = {shape_lambda}

"""Train Config"""
C.lr = 2e-5
C.batch_size = 2
C.nepochs = 150
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8
C.train_scale_array = [0.75, 1, 1.25]
C.warm_up_epoch = 10

# 正则化
C.drop_path_rate = 0.1
C.aux_rate = 0.0

# 早停机制
C.early_stopping = True
C.patience = 15
C.min_delta = 0.001

"""Path Config"""
import time
import os

experiment_name = "{experiment_name}"
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

C.log_dir = osp.abspath(osp.join("experiments/wheat_lodging", experiment_name, timestamp))
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoints"))

if not os.path.exists(C.log_dir):
    os.makedirs(C.log_dir, exist_ok=True)

# 日志文件
exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = osp.join(C.log_dir, f"log_{{exp_time}}.log")
C.link_log_file = osp.join(C.log_dir, "log_last.log")
C.val_log_file = osp.join(C.log_dir, f"val_{{exp_time}}.log")
C.link_val_log_file = osp.join(C.log_dir, "val_last.log")
'''
        
        # 填充配置参数
        config_params = {
            'use_shape_prior': exp_config.get('use_shape_prior', False),
            'shape_type': exp_config.get('shape_type', 'star'),
            'num_centers': exp_config.get('num_centers', 5),
            'temperature': exp_config.get('temperature', 1.0),
            'variational_weight': exp_config.get('variational_weight', 0.1),
            'shape_lambda': exp_config.get('shape_lambda', 0.1),
            'experiment_name': exp_name
        }
        
        return base_config.format(**config_params)
    
    def _run_training(self, exp_name: str, config_file: str, exp_dir: Path) -> Dict:
        """运行训练"""
        # 构建训练命令
        cmd = [
            "python", "utils/train.py",
            f"--config=local_configs.Wheatlodgingdata.{exp_name}",
            "--gpus=1",
            "--no-sliding",
            "--no-compile",
            "--syncbn",
            "--mst",
            "--compile_mode=default",
            "--no-amp",
            "--val_amp",
            "--use_seed"
        ]
        
        # 设置环境变量
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'LOCAL_RANK': '0',
            'WORLD_SIZE': '1',
            'RANK': '0'
        })
        
        # 运行训练
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd="/root/DFormer",
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "status": "failed",
                    "duration": duration,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "duration": 3600,
                "error": "Training timeout after 1 hour"
            }
        except Exception as e:
            return {
                "status": "error",
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    def _save_results(self, stage_name: str, results: Dict):
        """保存实验结果"""
        results_file = self.base_dir / f"{stage_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {results_file}")
    
    def generate_analysis_report(self, stage_name: str):
        """生成分析报告"""
        if stage_name not in self.results:
            print(f"No results found for stage: {stage_name}")
            return
        
        results = self.results[stage_name]
        
        # 生成报告
        report = f"""
# Wheat Lodging Segmentation Experiment Report
## Stage: {stage_name}

### Experiment Summary
"""
        
        for exp_name, result in results.items():
            status = result.get('status', 'unknown')
            duration = result.get('duration', 0)
            
            report += f"\n- **{exp_name}**: {status} (Duration: {duration:.1f}s)"
        
        # 保存报告
        report_file = self.base_dir / f"{stage_name}_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✓ Analysis report saved to: {report_file}")
    
    def run_all_stages(self, dry_run: bool = False):
        """运行所有实验阶段"""
        print("🌾 Wheat Lodging Segmentation Experiment Suite")
        print("="*80)
        
        for stage_name in self.experiments.keys():
            try:
                self.run_stage(stage_name, dry_run)
                self.generate_analysis_report(stage_name)
            except Exception as e:
                print(f"✗ Stage {stage_name} failed: {e}")
        
        print("\n🎉 All experiments completed!")
        print(f"Results saved in: {self.base_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wheat Lodging Segmentation Experiment')
    parser.add_argument('--stage', type=str, help='Specific stage to run')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--all', action='store_true', help='Run all stages')
    
    args = parser.parse_args()
    
    # 创建实验管理器
    experiment = WheatLodgingExperiment()
    
    if args.all:
        experiment.run_all_stages(dry_run=args.dry_run)
    elif args.stage:
        experiment.run_stage(args.stage, dry_run=args.dry_run)
        experiment.generate_analysis_report(args.stage)
    else:
        print("Please specify --stage or --all")
        print("Available stages:", list(experiment.experiments.keys()))


if __name__ == "__main__":
    main()
