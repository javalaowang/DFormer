#!/bin/bash
#
# Multi-View Consistency Learning Experiment Script
# 用于运行v-CLR对比实验
#

set -e

echo "=========================================="
echo "Multi-View Consistency Learning Experiment"
echo "=========================================="

# 配置
GPUS=2
CONFIG_PATH="local_configs.Wheatlodgingdata.DFormerv2_Large_vCLR"
OUTPUT_DIR="experiments/vCLR_$(date +%Y%m%d_%H%M%S)"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo ""
echo "Starting Experiment..."
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo ""

# 运行实验框架
python utils/experiment_framework.py \
    --experiments baseline vclr \
    --output_dir $OUTPUT_DIR

echo ""
echo "Experiment completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated outputs:"
echo "  - comparison_table.tex (LaTeX table)"
echo "  - comparison_table.csv (CSV data)"
echo "  - comparison_plots.png (Visualization)"
echo "  - ablation_study.tex (Ablation results)"
echo "  - experiment_report.md (Full report)"
echo ""

