#!/bin/bash

# Wheatlodging数据集评估脚本
GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/eval.py \
    --config=local_configs.Wheatlodgingdata.DFormer_Large \
    --gpus=$GPUS \
    --sliding \
    --no-compile \
    --syncbn \
    --mst \
    --compile_mode="reduce-overhead" \
    --amp \
    --continue_fpath="checkpoints/Wheatlodgingdata_DFormer-Large_20251021-172616/epoch-40_miou_72.1.pth"


