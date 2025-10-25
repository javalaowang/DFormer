#!/bin/bash
# CCS消融实验训练脚本
# 用于论文实验：系统性地测试CCS各个组件的效果

GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1

# 消融实验配置
ABLATION_VARIANTS=(
    "baseline"           # 基线：不使用CCS
    "centers_3"          # 3个中心
    "centers_5"          # 5个中心
    "centers_7"          # 7个中心
    "lambda_0.05"        # 损失权重0.05
    "lambda_0.1"         # 损失权重0.1
    "lambda_0.2"         # 损失权重0.2
    "alpha_0.05"         # 增强权重0.05
    "alpha_0.1"          # 增强权重0.1
    "alpha_0.2"          # 增强权重0.2
    "fixed_centers"      # 固定中心
    "learnable_centers"  # 学习中心
)

# 实验根目录
EXPERIMENT_ROOT="experiments/ablation_ccs"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

echo "=========================================="
echo "CCS Ablation Study Training"
echo "=========================================="
echo "Experiment Root: $EXPERIMENT_ROOT"
echo "Timestamp: $TIMESTAMP"
echo "Total Variants: ${#ABLATION_VARIANTS[@]}"
echo "=========================================="

# 创建实验根目录
mkdir -p $EXPERIMENT_ROOT

# 记录实验开始时间
echo "Experiment started at: $(date)" > $EXPERIMENT_ROOT/experiment_log.txt

# 遍历所有消融变体
for i in "${!ABLATION_VARIANTS[@]}"; do
    VARIANT="${ABLATION_VARIANTS[$i]}"
    
    echo ""
    echo "=========================================="
    echo "Training Variant $((i+1))/${#ABLATION_VARIANTS[@]}: $VARIANT"
    echo "=========================================="
    
    # 设置变体特定的配置
    case $VARIANT in
        "baseline")
            CCS_USE="False"
            CCS_CENTERS="5"
            CCS_LAMBDA="0.0"
            CCS_ALPHA="0.0"
            CCS_LEARNABLE="True"
            ;;
        "centers_3")
            CCS_USE="True"
            CCS_CENTERS="3"
            CCS_LAMBDA="0.1"
            CCS_ALPHA="0.1"
            CCS_LEARNABLE="True"
            ;;
        "centers_5")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_LAMBDA="0.1"
            CCS_ALPHA="0.1"
            CCS_LEARNABLE="True"
            ;;
        "centers_7")
            CCS_USE="True"
            CCS_CENTERS="7"
            CCS_LAMBDA="0.1"
            CCS_ALPHA="0.1"
            CCS_LEARNABLE="True"
            ;;
        "lambda_0.05")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_LAMBDA="0.05"
            CCS_ALPHA="0.1"
            CCS_LEARNABLE="True"
            ;;
        "lambda_0.1")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_LAMBDA="0.1"
            CCS_ALPHA="0.1"
            CCS_LEARNABLE="True"
            ;;
        "lambda_0.2")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_LAMBDA="0.2"
            CCS_ALPHA="0.1"
            CCS_LEARNABLE="True"
            ;;
        "alpha_0.05")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_LAMBDA="0.1"
            CCS_ALPHA="0.05"
            CCS_LEARNABLE="True"
            ;;
        "alpha_0.1")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_LAMBDA="0.1"
            CCS_ALPHA="0.1"
            CCS_LEARNABLE="True"
            ;;
        "alpha_0.2")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_LAMBDA="0.1"
            CCS_ALPHA="0.2"
            CCS_LEARNABLE="True"
            ;;
        "fixed_centers")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_LAMBDA="0.1"
            CCS_ALPHA="0.1"
            CCS_LEARNABLE="False"
            ;;
        "learnable_centers")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_LAMBDA="0.1"
            CCS_ALPHA="0.1"
            CCS_LEARNABLE="True"
            ;;
    esac
    
    # 创建变体特定的配置
    VARIANT_CONFIG="local_configs/Wheatlodgingdata/DFormerv2_L_CCS_${VARIANT}.py"
    
    # 复制基础配置
    cp local_configs/Wheatlodgingdata/DFormerv2_L_CCS_Ablation.py $VARIANT_CONFIG
    
    # 修改配置参数
    sed -i "s/C.ablation_variant = \"default\"/C.ablation_variant = \"$VARIANT\"/" $VARIANT_CONFIG
    sed -i "s/C.use_ccs = True/C.use_ccs = $CCS_USE/" $VARIANT_CONFIG
    sed -i "s/C.ccs_num_centers = 5/C.ccs_num_centers = $CCS_CENTERS/" $VARIANT_CONFIG
    sed -i "s/C.ccs_lambda = 0.1/C.ccs_lambda = $CCS_LAMBDA/" $VARIANT_CONFIG
    sed -i "s/C.ccs_alpha = 0.1/C.ccs_alpha = $CCS_ALPHA/" $VARIANT_CONFIG
    sed -i "s/C.ccs_learnable_centers = True/C.ccs_learnable_centers = $CCS_LEARNABLE/" $VARIANT_CONFIG
    
    echo "Configuration:"
    echo "  CCS Use: $CCS_USE"
    echo "  Centers: $CCS_CENTERS"
    echo "  Lambda: $CCS_LAMBDA"
    echo "  Alpha: $CCS_ALPHA"
    echo "  Learnable Centers: $CCS_LEARNABLE"
    
    # 开始训练
    echo "Starting training for variant: $VARIANT"
    
    PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
        torchrun \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        utils/train.py \
        --config=local_configs.Wheatlodgingdata.DFormerv2_L_CCS_${VARIANT} \
        --gpus=$GPUS \
        --no-sliding \
        --no-compile \
        --syncbn \
        --mst \
        --compile_mode="default" \
        --no-amp \
        --val_amp \
        --use_seed
    
    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo "✓ Training completed successfully for variant: $VARIANT"
        echo "Variant $VARIANT completed at: $(date)" >> $EXPERIMENT_ROOT/experiment_log.txt
    else
        echo "✗ Training failed for variant: $VARIANT"
        echo "Variant $VARIANT failed at: $(date)" >> $EXPERIMENT_ROOT/experiment_log.txt
    fi
    
    # 清理临时配置文件
    rm -f $VARIANT_CONFIG
    
    echo "Completed variant $((i+1))/${#ABLATION_VARIANTS[@]}: $VARIANT"
done

# 实验完成
echo ""
echo "=========================================="
echo "All Ablation Experiments Completed!"
echo "=========================================="
echo "Experiment finished at: $(date)" >> $EXPERIMENT_ROOT/experiment_log.txt

# 生成实验结果汇总
echo "Generating experiment summary..."
python utils/generate_ablation_summary.py --experiment_root=$EXPERIMENT_ROOT

echo "✓ Ablation study completed!"
echo "Results saved in: $EXPERIMENT_ROOT"
