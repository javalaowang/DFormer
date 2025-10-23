#!/bin/bash

# Wheatlodging数据集改进训练脚本 - 解决过拟合问题
GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1

echo "🚀 开始改进训练 - 解决过拟合问题"
echo "📊 使用配置: DFormer-Small + 正则化 + 早停"
echo "🎯 目标: 提高泛化能力，减少过拟合"

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=local_configs.Wheatlodgingdata.DFormer_Small_improved --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --mst \
    --compile_mode="default" \
    --no-amp \
    --val_amp \
    --no-use_seed

echo "✅ 训练完成！"
echo "📈 请检查训练日志和验证曲线"


