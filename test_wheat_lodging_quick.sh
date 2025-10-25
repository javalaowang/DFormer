#!/bin/bash
# 小麦倒伏分割快速测试脚本
# Quick test for wheat lodging segmentation experiments

echo "🌾 Quick Test: Wheat Lodging Segmentation"
echo "========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES="0"
export LOCAL_RANK=0
export WORLD_SIZE=1
export RANK=0

# 切换到项目目录
cd /root/DFormer

# 创建实验目录
mkdir -p experiments/wheat_lodging

echo "Running quick test with dry run mode..."
echo "This will show what experiments would be run without actually training."

# 运行快速测试（dry run模式）
python experiments/wheat_lodging_experiment.py --stage stage1_baseline --dry-run

echo ""
echo "✓ Quick test completed!"
echo "To run actual experiments, use: bash run_wheat_lodging_experiments.sh"
echo "========================================="
