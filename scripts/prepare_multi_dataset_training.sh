#!/bin/bash
# 多数据集训练准备脚本

set -e

echo "=========================================="
echo "多数据集训练准备脚本"
echo "=========================================="
echo ""

# 检查数据集是否存在
check_dataset() {
    local dataset_name=$1
    local dataset_path="datasets/${dataset_name}"
    
    if [ -d "$dataset_path" ]; then
        echo "✅ ${dataset_name}: 已存在"
        echo "   路径: $dataset_path"
        
        # 检查关键文件
        if [ -d "${dataset_path}/RGB" ] && [ -f "${dataset_path}/train.txt" ]; then
            train_count=$(wc -l < "${dataset_path}/train.txt")
            echo "   训练集: ${train_count} 张图像"
        fi
        
        if [ -f "${dataset_path}/test.txt" ]; then
            test_count=$(wc -l < "${dataset_path}/test.txt")
            echo "   测试集: ${test_count} 张图像"
        fi
        
        return 0
    else
        echo "❌ ${dataset_name}: 不存在"
        echo "   请下载数据集到: $dataset_path"
        return 1
    fi
}

echo "检查数据集..."
echo ""

# 检查各个数据集
WHEAT_EXISTS=0
NYU_EXISTS=0
SUN_EXISTS=0

if check_dataset "Wheatlodgingdata"; then
    WHEAT_EXISTS=1
fi
echo ""

if check_dataset "NYUDepthv2"; then
    NYU_EXISTS=1
fi
echo ""

if check_dataset "SUNRGBD"; then
    SUN_EXISTS=1
fi
echo ""

# 总结
echo "=========================================="
echo "数据集状态总结"
echo "=========================================="
echo ""
echo "✅ Wheatlodgingdata: $([ $WHEAT_EXISTS -eq 1 ] && echo '已准备' || echo '未准备')"
echo "⏳ NYUDepth v2:     $([ $NYU_EXISTS -eq 1 ] && echo '已准备' || echo '未准备')"
echo "⏳ SUN RGB-D:        $([ $SUN_EXISTS -eq 1 ] && echo '已准备' || echo '未准备')"
echo ""

# 提供下一步建议
if [ $NYU_EXISTS -eq 0 ] || [ $SUN_EXISTS -eq 0 ]; then
    echo "⚠️  需要下载数据集，请参考: DATASET_PREPARATION_GUIDE.md"
    echo ""
    echo "下载链接:"
    echo "  NYUDepth v2:"
    echo "    - Google Drive: https://drive.google.com/drive/folders/1P5HwnAvifEI6xiTAx6id24FUCt_i7GH8"
    echo "    - 百度网盘: https://pan.baidu.com/s/1AkvlsAvJPv21bz2sXlrADQ?pwd=6vuu"
    echo ""
    echo "  SUN RGB-D:"
    echo "    - Google Drive: https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8"
    echo "    - 百度网盘: https://pan.baidu.com/s/1D6UMiBv6fApV5lafo9J04w?pwd=7ewv"
    echo ""
fi

# 检查配置文件
echo "检查vCLR配置文件..."
echo ""

check_config() {
    local config_path=$1
    if [ -f "$config_path" ]; then
        echo "✅ $(basename $config_path): 已存在"
        return 0
    else
        echo "❌ $(basename $config_path): 不存在"
        return 1
    fi
}

CONFIGS_OK=1

if ! check_config "local_configs/Wheatlodgingdata/DFormerv2_Large_vCLR.py"; then
    CONFIGS_OK=0
fi
if ! check_config "local_configs/NYUDepthv2/DFormerv2_L_vCLR.py"; then
    CONFIGS_OK=0
fi
if ! check_config "local_configs/SUNRGBD/DFormerv2_L_vCLR.py"; then
    CONFIGS_OK=0
fi

echo ""

if [ $CONFIGS_OK -eq 1 ]; then
    echo "✅ 所有vCLR配置文件已准备"
else
    echo "❌ 部分配置文件缺失"
fi

echo ""
echo "=========================================="
echo "准备完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 如果数据集未准备，请先下载数据集"
echo "2. 运行训练命令:"
echo ""
echo "   # NYUDepth v2"
echo "   bash train.sh --config local_configs.NYUDepthv2.DFormerv2_L_vCLR"
echo ""
echo "   # SUN RGB-D"
echo "   bash train.sh --config local_configs.SUNRGBD.DFormerv2_L_vCLR"
echo ""

