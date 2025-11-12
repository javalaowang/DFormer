#!/bin/bash

# NYUDepth v2 vCLR 训练脚本 (单GPU)
# 使用GPU 0进行训练，启用vCLR模块

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 创建日志目录
LOG_DIR="logs/nyu_vclr_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "=========================================="
echo "开始训练 NYUDepth v2 with vCLR"
echo "=========================================="
echo "配置: local_configs.NYUDepthv2.DFormerv2_L_vCLR"
echo "GPU: 0"
echo "日志目录: $LOG_DIR"
echo "开始时间: $(date)"
echo "=========================================="

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=29159 \
    utils/train.py \
    --config=local_configs.NYUDepthv2.DFormerv2_L_vCLR \
    --gpus=1 \
    --no-sliding \
    --no-compile \
    --syncbn \
    --mst \
    --compile_mode="default" \
    --amp \
    --val_amp \
    --no-pad_SUNRGBD \
    --use_seed \
    2>&1 | tee $LOG_DIR/training.log

echo "=========================================="
echo "训练完成"
echo "结束时间: $(date)"
echo "=========================================="

