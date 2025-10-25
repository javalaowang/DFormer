#!/bin/bash
# CCS Paper Implementation Training Script
# 基于CVPR 2025论文的严谨实现训练脚本

GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1

echo "=========================================="
echo "CCS Paper Implementation Training"
echo "=========================================="
echo "Based on: Zhao et al. CVPR 2025"
echo "Implementation: Paper-based mathematical formulation"
echo "=========================================="

# 设置必要的环境变量
export LOCAL_RANK=0
export WORLD_SIZE=1
export RANK=0

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=local_configs.Wheatlodgingdata.DFormerv2_L_CCS_Paper \
    --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --mst \
    --compile_mode="default" \
    --no-amp \
    --val_amp \
    --use_seed

echo ""
echo "=========================================="
echo "✓ CCS Paper Implementation Training Completed"
echo "=========================================="
echo "Results saved in: experiments/paper_ccs/"
echo "Implementation: Strict mathematical formulation"
echo "Paper: Zhao et al. CVPR 2025"
echo "=========================================="



