#!/bin/bash
# 快速CCS实验脚本

GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
# 使用动态端口避免冲突
PORT=${PORT:-$((29158 + RANDOM % 1000))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1

echo "🌾 Quick CCS Experiment"
echo "========================================="
echo "Config: DFormerv2_L_CCS_Quick"
echo "Epochs: 20"
echo "Batch Size: 1"
echo "CCS Centers: 3"
echo "========================================="

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=local_configs.Wheatlodgingdata.DFormerv2_L_CCS_Quick \
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
echo "========================================="
echo "✅ Quick CCS Experiment Completed!"
echo "========================================="
