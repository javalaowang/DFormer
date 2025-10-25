#!/bin/bash
# 小麦倒伏分割专门实验运行脚本
# Wheat Lodging Segmentation Specialized Experiment Runner

echo "🌾 Wheat Lodging Segmentation Experiment Suite"
echo "=============================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES="0"
export LOCAL_RANK=0
export WORLD_SIZE=1
export RANK=0

# 切换到项目目录
cd /root/DFormer

# 创建实验目录
mkdir -p experiments/wheat_lodging

# 运行实验管理器
python experiments/wheat_lodging_experiment.py --all

echo ""
echo "🎉 All wheat lodging experiments completed!"
echo "Results saved in: experiments/wheat_lodging/"
echo "=============================================="
