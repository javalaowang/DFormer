#!/bin/bash
# 安全的vCLR训练启动脚本
# 使用nohup确保SSH断连后训练继续运行

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="vCLR_training_safe_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Safe v-CLR Training Launch"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo "=========================================="

# 使用nohup启动训练，并在后台运行
nohup bash train_wheatlodging_vclr.sh > "$LOG_FILE" 2>&1 &

# 获取进程ID
TRAIN_PID=$!

echo "Training started with PID: $TRAIN_PID"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To check if running: ps aux | grep $TRAIN_PID"
echo "=========================================="

# 等待3秒，检查进程是否正常启动
sleep 3

if ps -p $TRAIN_PID > /dev/null; then
    echo "✓ Training process is running successfully"
    echo "✓ You can safely disconnect your SSH session"
    echo ""
    echo "Useful commands:"
    echo "  - Monitor: tail -f $LOG_FILE"
    echo "  - Check status: ps aux | grep 'train.py'"
    echo "  - Kill training: kill $TRAIN_PID"
else
    echo "✗ Training process failed to start"
    echo "Check log file: $LOG_FILE"
    exit 1
fi

