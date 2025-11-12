#!/bin/bash

# ========================================
# 开题报告LaTeX文档打包脚本
# ========================================
#
# 用途: 将所有LaTeX文件打包为zip，方便上传到Overleaf
#
# 使用方法:
#   bash 打包上传.sh
#
# 输出:
#   thesis_proposal.zip (可直接上传Overleaf)
#
# ========================================

echo "========================================="
echo "开题报告LaTeX文档打包工具"
echo "========================================="

# 进入thesis_proposal目录的父目录
cd "$(dirname "$0")/.."

# 检查文件是否存在
if [ ! -f "thesis_proposal/main.tex" ]; then
    echo "❌ 错误: 找不到main.tex文件"
    exit 1
fi

# 创建临时目录
TEMP_DIR="thesis_proposal_temp"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo ""
echo "📂 复制文件..."

# 复制所有需要的文件
cp thesis_proposal/main.tex "$TEMP_DIR/"
cp -r thesis_proposal/sections "$TEMP_DIR/"
cp thesis_proposal/README.md "$TEMP_DIR/"

echo "✓ 文件复制完成"

# 创建zip压缩包
echo ""
echo "📦 打包中..."
cd "$TEMP_DIR"
zip -r ../thesis_proposal.zip . -q

cd ..

# 清理临时文件
rm -rf "$TEMP_DIR"

echo "✓ 打包完成"

echo ""
echo "========================================="
echo "✅ 成功!"
echo "========================================="
echo ""
echo "输出文件: thesis_proposal.zip"
echo "文件大小: $(du -h thesis_proposal.zip | cut -f1)"
echo ""
echo "下一步:"
echo "  1. 访问 https://www.overleaf.com/"
echo "  2. 点击 'New Project' → 'Upload Project'"
echo "  3. 上传 thesis_proposal.zip"
echo "  4. 点击 'Recompile' 编译PDF"
echo ""
echo "========================================="

