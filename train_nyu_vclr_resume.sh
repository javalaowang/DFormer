#!/bin/bash

# NYUDepth v2 vCLR 恢复训练脚本 (从 Epoch 30 继续)
# 使用GPU 0进行训练，启用vCLR模块

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Checkpoint 路径
CHECKPOINT_PATH="checkpoints/NYUDepthv2_DFormerv2_L_vCLR_20251104-111339/epoch-30_miou_11.13.pth"

# 创建日志目录
LOG_DIR="logs/nyu_vclr_resume_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "=========================================="
echo "恢复训练 NYUDepth v2 with vCLR"
echo "=========================================="
echo "配置: local_configs.NYUDepthv2.DFormerv2_L_vCLR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "GPU: 0"
echo "日志目录: $LOG_DIR"
echo "开始时间: $(date)"
echo "=========================================="

# 检查 checkpoint 是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: Checkpoint 文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "从 checkpoint 恢复训练: $CHECKPOINT_PATH"

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
    --continue_fpath="$CHECKPOINT_PATH" \
    2>&1 | tee $LOG_DIR/training.log

echo "=========================================="
echo "训练完成"
echo "结束时间: $(date)"
echo "=========================================="

