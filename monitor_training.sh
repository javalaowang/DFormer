#!/bin/bash
#
# 监控训练进度
#

echo "=========================================="
echo "Training Monitor - v-CLR Training Status"
echo "=========================================="
echo ""

# 检查训练进程
echo "1. Training Process:"
ps aux | grep "train.py" | grep -v grep | awk '{print "   PID:", $2, "- CPU:", $3"%", "- Memory:", $4"%"}'
echo ""

# GPU状态
echo "2. GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
echo ""

# 检查最新的训练日志
LATEST_DIR=$(ls -td checkpoints/Wheatlodgingdata_DFormerv2_L_pretrained_* 2>/dev/null | head -1)

if [ -n "$LATEST_DIR" ]; then
    echo "3. Latest Training Directory:"
    echo "   $LATEST_DIR"
    echo ""
    
    echo "4. Recent Log Output:"
    LATEST_LOG=$(find "$LATEST_DIR" -name "*.log" -type f | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "   Log file: $LATEST_LOG"
        echo "   Last 20 lines:"
        tail -20 "$LATEST_LOG"
    fi
fi

echo ""
echo "=========================================="
echo "To check GPU in real-time: watch -n 1 nvidia-smi"
echo "To follow logs: tail -f $LATEST_LOG"
echo "=========================================="

