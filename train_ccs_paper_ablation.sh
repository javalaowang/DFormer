#!/bin/bash
# CCS Paper Implementation Ablation Study
# 基于CVPR 2025论文的消融实验脚本

GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
# 使用动态端口避免冲突
PORT=${PORT:-$((29158 + RANDOM % 1000))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1

# 消融实验配置 - 基于论文的严谨设计
ABLATION_VARIANTS=(
    "baseline"           # 基线：不使用CCS
    "centers_3"          # 3个星形中心
    "centers_5"          # 5个星形中心
    "centers_7"          # 7个星形中心
    "temp_0.5"           # 温度参数0.5
    "temp_1.0"           # 温度参数1.0
    "temp_2.0"           # 温度参数2.0
    "var_0.05"           # 变分权重0.05
    "var_0.1"            # 变分权重0.1
    "var_0.2"            # 变分权重0.2
    "shape_0.05"         # 形状损失权重0.05
    "shape_0.1"          # 形状损失权重0.1
    "shape_0.2"          # 形状损失权重0.2
    "fixed_centers"      # 固定中心位置
    "learnable_centers"  # 学习中心位置
    "fixed_radius"       # 固定半径函数
    "learnable_radius"   # 学习半径函数
)

# 实验根目录
EXPERIMENT_ROOT="experiments/paper_ablation"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

echo "=========================================="
echo "CCS Paper Implementation Ablation Study"
echo "=========================================="
echo "Based on: Zhao et al. CVPR 2025"
echo "Implementation: Paper-based mathematical formulation"
echo "Experiment Root: $EXPERIMENT_ROOT"
echo "Timestamp: $TIMESTAMP"
echo "Total Variants: ${#ABLATION_VARIANTS[@]}"
echo "=========================================="

# 创建实验根目录
mkdir -p $EXPERIMENT_ROOT

# 记录实验开始时间
echo "Experiment started at: $(date)" > $EXPERIMENT_ROOT/experiment_log.txt
echo "Paper: Zhao et al. CVPR 2025" >> $EXPERIMENT_ROOT/experiment_log.txt
echo "Implementation: Paper-based mathematical formulation" >> $EXPERIMENT_ROOT/experiment_log.txt

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
            CCS_TEMP="1.0"
            CCS_VAR="0.0"
            CCS_SHAPE="0.0"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "centers_3")
            CCS_USE="True"
            CCS_CENTERS="3"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "centers_5")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "centers_7")
            CCS_USE="True"
            CCS_CENTERS="7"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "temp_0.5")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="0.5"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "temp_1.0")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "temp_2.0")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="2.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "var_0.05")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.05"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "var_0.1")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "var_0.2")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.2"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "shape_0.05")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.05"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "shape_0.1")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "shape_0.2")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.2"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "fixed_centers")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="False"
            CCS_LEARN_RADIUS="True"
            ;;
        "learnable_centers")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
        "fixed_radius")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="False"
            ;;
        "learnable_radius")
            CCS_USE="True"
            CCS_CENTERS="5"
            CCS_TEMP="1.0"
            CCS_VAR="0.1"
            CCS_SHAPE="0.1"
            CCS_LEARN_CENTERS="True"
            CCS_LEARN_RADIUS="True"
            ;;
    esac
    
    # 创建变体特定的配置
    VARIANT_CONFIG="local_configs/Wheatlodgingdata/DFormerv2_L_CCS_Paper_${VARIANT}.py"
    
    # 复制基础配置
    cp local_configs/Wheatlodgingdata/DFormerv2_L_CCS_Paper_Ablation.py $VARIANT_CONFIG
    
    # 修改配置参数
    sed -i "s/C.ablation_variant = \"default\"/C.ablation_variant = \"$VARIANT\"/" $VARIANT_CONFIG
    sed -i "s/C.use_ccs = True/C.use_ccs = $CCS_USE/" $VARIANT_CONFIG
    sed -i "s/C.ccs_num_centers = 5/C.ccs_num_centers = $CCS_CENTERS/" $VARIANT_CONFIG
    sed -i "s/C.ccs_temperature = 1.0/C.ccs_temperature = $CCS_TEMP/" $VARIANT_CONFIG
    sed -i "s/C.ccs_variational_weight = 0.1/C.ccs_variational_weight = $CCS_VAR/" $VARIANT_CONFIG
    sed -i "s/C.ccs_shape_lambda = 0.1/C.ccs_shape_lambda = $CCS_SHAPE/" $VARIANT_CONFIG
    sed -i "s/C.ccs_learnable_centers = True/C.ccs_learnable_centers = $CCS_LEARN_CENTERS/" $VARIANT_CONFIG
    sed -i "s/C.ccs_learnable_radius = True/C.ccs_learnable_radius = $CCS_LEARN_RADIUS/" $VARIANT_CONFIG
    
    echo "Configuration:"
    echo "  CCS Use: $CCS_USE"
    echo "  Centers: $CCS_CENTERS"
    echo "  Temperature: $CCS_TEMP"
    echo "  Variational Weight: $CCS_VAR"
    echo "  Shape Lambda: $CCS_SHAPE"
    echo "  Learnable Centers: $CCS_LEARN_CENTERS"
    echo "  Learnable Radius: $CCS_LEARN_RADIUS"
    
    # 开始训练
    echo "Starting training for variant: $VARIANT"
    
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
        --config=local_configs.Wheatlodgingdata.DFormerv2_L_CCS_Paper_${VARIANT} \
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
echo "All Paper-based Ablation Experiments Completed!"
echo "=========================================="
echo "Experiment finished at: $(date)" >> $EXPERIMENT_ROOT/experiment_log.txt

# 生成实验结果汇总
echo "Generating paper-based experiment summary..."
python utils/generate_paper_ablation_summary.py --experiment_root=$EXPERIMENT_ROOT

echo "✓ Paper-based ablation study completed!"
echo "Results saved in: $EXPERIMENT_ROOT"
echo "Implementation: Strict mathematical formulation"
echo "Paper: Zhao et al. CVPR 2025"



