#!/bin/bash

# Enhanced Wheatlodging数据集预训练模型训练脚本 - 论文级别输出
# 包含完整的可视化和分析功能

GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 设置实验标识符
EXPERIMENT_NAME="Wheatlodging_DFormerv2_L_pretrained_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="experiments/${EXPERIMENT_NAME}"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}/{logs,checkpoints,visualizations,metrics,analysis}

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1

# 设置环境变量用于增强日志记录
export EXPERIMENT_NAME=${EXPERIMENT_NAME}
export OUTPUT_DIR=${OUTPUT_DIR}
export ENABLE_DETAILED_LOGGING=1
export ENABLE_VISUALIZATION=1
export ENABLE_METRICS_ANALYSIS=1

echo "=========================================="
echo "Enhanced Training for Paper Publication"
echo "=========================================="
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Start Time: $(date)"
echo "=========================================="

# 记录系统信息
echo "System Information:" > ${OUTPUT_DIR}/logs/system_info.log
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)" >> ${OUTPUT_DIR}/logs/system_info.log
echo "CUDA Version: $(nvcc --version | grep release)" >> ${OUTPUT_DIR}/logs/system_info.log
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')" >> ${OUTPUT_DIR}/logs/system_info.log
echo "Python Version: $(python --version)" >> ${OUTPUT_DIR}/logs/system_info.log

# 启动增强训练 - 使用原始训练脚本但添加增强功能
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
    --config=local_configs.Wheatlodgingdata.DFormerv2_Large_pretrained \
    --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --mst \
    --compile_mode="default" \
    --no-amp \
    --val_amp \
    --use_seed \
    --save_path=${OUTPUT_DIR}/checkpoints \
    --checkpoint_dir=${OUTPUT_DIR}/checkpoints

# 训练完成后生成分析报告
echo "=========================================="
echo "Generating Analysis Report..."
echo "=========================================="

# 生成训练分析报告
python utils/generate_simple_analysis.py \
    --experiment_dir=${OUTPUT_DIR}

echo "=========================================="
echo "Training Completed Successfully!"
echo "=========================================="
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "End Time: $(date)"
echo "=========================================="
echo "Generated Files:"
echo "- Training logs: ${OUTPUT_DIR}/logs/"
echo "- Model checkpoints: ${OUTPUT_DIR}/checkpoints/"
echo "- Visualizations: ${OUTPUT_DIR}/visualizations/"
echo "- Metrics analysis: ${OUTPUT_DIR}/metrics/"
echo "- Paper-ready figures: ${OUTPUT_DIR}/analysis/"
echo "=========================================="
