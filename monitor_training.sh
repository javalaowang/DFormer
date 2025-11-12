#!/bin/bash

# 训练监控脚本
# 用法: bash monitor_training.sh [dataset_name] [experiment_type]

DATASET_NAME=${1:-"NYUDepthv2"}
EXP_TYPE=${2:-"baseline"}

echo "=========================================="
echo "训练监控 - $DATASET_NAME $EXP_TYPE"
echo "=========================================="

# 查找最新的checkpoint目录
if [ "$EXP_TYPE" == "vclr" ]; then
    CHECKPOINT_DIR=$(find checkpoints -name "*${DATASET_NAME}*vCLR*" -type d | sort -r | head -1)
else
    CHECKPOINT_DIR=$(find checkpoints -name "*${DATASET_NAME}*DFormerv2_L*" -type d | grep -v vCLR | sort -r | head -1)
fi

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "❌ 未找到checkpoint目录"
    exit 1
fi

echo "📁 Checkpoint目录: $CHECKPOINT_DIR"
echo ""

# 检查进程
echo "🔄 训练进程状态:"
ps aux | grep -E "train.py.*${DATASET_NAME}" | grep -v grep || echo "  ⚠️  未找到训练进程"
echo ""

# GPU使用情况
echo "🖥️  GPU状态:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "无法获取GPU信息"
echo ""

# 查找日志文件
LOG_FILE=$(find "$CHECKPOINT_DIR" -name "log_*.log" | sort -r | head -1)
VAL_LOG_FILE=$(find "$CHECKPOINT_DIR" -name "val_*.log" | sort -r | head -1)

if [ -n "$LOG_FILE" ]; then
    echo "📊 训练日志 (最后20行):"
    echo "文件: $LOG_FILE"
    echo "----------------------------------------"
    tail -20 "$LOG_FILE" 2>/dev/null || echo "日志文件为空"
    echo ""
fi

if [ -n "$VAL_LOG_FILE" ]; then
    echo "📈 验证日志 (最后10行):"
    echo "文件: $VAL_LOG_FILE"
    echo "----------------------------------------"
    tail -10 "$VAL_LOG_FILE" 2>/dev/null || echo "验证日志为空"
    echo ""
fi

# 检查checkpoint
echo "💾 保存的模型:"
find "$CHECKPOINT_DIR/checkpoint" -name "*.pth" 2>/dev/null | sort -r | head -5 || echo "  暂无保存的模型"
echo ""

# 训练进度估算（如果有）
if [ -n "$LOG_FILE" ]; then
    echo "📉 训练进度:"
    grep -E "Epoch|epoch|iter" "$LOG_FILE" 2>/dev/null | tail -3 || echo "  暂无进度信息"
fi

echo "=========================================="
echo "实时监控命令:"
echo "  tail -f $LOG_FILE"
echo "=========================================="
