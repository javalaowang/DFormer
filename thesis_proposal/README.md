# 硕士学位论文开题报告 LaTeX 文档

## 📁 文件结构

```
thesis_proposal/
├── main.tex                    # 主文档
├── sections/                   # 章节目录
│   ├── section1_background.tex # 第一部分：研究背景与进展
│   ├── section2_content.tex    # 第二部分：研究内容
│   ├── section3_method.tex     # 第三部分：研究方法
│   ├── section4_schedule.tex   # 第四部分：进度安排
│   ├── section5_resources.tex  # 第五部分：资源与经费
│   └── references.tex          # 参考文献
└── README.md                   # 本文档
```

## 🚀 快速使用

### 方法1: Overleaf在线编译（推荐）

1. 访问 [Overleaf](https://www.overleaf.com/)
2. 新建项目 → Upload Project
3. 将整个 `thesis_proposal` 文件夹打包上传
4. 点击 "Recompile" 编译
5. 下载PDF

### 方法2: 本地编译

#### 前置要求

安装完整的TeX发行版：
- **Windows**: TeX Live 或 MiKTeX
- **macOS**: MacTeX
- **Linux**: TeX Live

#### 编译命令

```bash
cd thesis_proposal
xelatex main.tex
xelatex main.tex  # 编译两次以生成目录和引用
```

或使用 latexmk:
```bash
latexmk -xelatex main.tex
```

## 📝 自定义修改

### 修改个人信息

编辑 `main.tex` 第23-32行：

```latex
研究生姓名： & \underline{\hspace{6cm}张三} \\
学号： & \underline{\hspace{6cm}2022XXXXXX} \\
...
```

### 修改研究内容

各章节独立编辑：
- 研究背景：`sections/section1_background.tex`
- 研究内容：`sections/section2_content.tex`
- 研究方法：`sections/section3_method.tex`
- 进度安排：`sections/section4_schedule.tex`
- 资源经费：`sections/section5_resources.tex`

### 修改参考文献

编辑 `sections/references.tex`，按 BibTeX 格式添加：

```latex
\bibitem{author2024title}
作者. 标题[J/C]. 期刊/会议, 年份, 卷(期): 页码.
```

## 🎨 格式说明

### 文档格式
- 字体：宋体（中文）、Times New Roman（英文）
- 字号：小四（正文）、五号（表格）
- 行距：1.5倍
- 页边距：上下左右2.5cm

### 章节编号
- 一级标题：\section{} （如：一、）
- 二级标题：\subsection{} （如：1.1）
- 三级标题：\subsubsection{} （如：1.1.1）

### 公式编号
自动编号，使用 `\begin{equation}...\end{equation}`

### 表格与图片
- 表格：使用 `\begin{table}...\end{table}`
- 图片：使用 `\begin{figure}...\end{figure}`
- 标题：`\caption{}`

## 📊 内容概要

### 核心创新点

1. **SACL框架**: 空间自适应跨模态一致性学习
2. **双层优化**: 几何引导+特征一致性
3. **MDA-CMC**: 多域对齐的跨模态学习
4. **梯度一致性**: 边界对齐增强

### 研究内容

1. DFormerv2基线模型
2. SACL方法研究（核心）
3. 跨地块泛化评估
4. 半监督学习扩展
5. GIS监测系统

### 预期成果

- 同域性能：87.5% mIoU
- 跨域性能：平均76.8% mIoU（+8.9%）
- 数据集：≥5000张RGB-D标注图像
- 论文：1篇期刊 + 1篇会议
- 系统：可演示的GIS原型

## 🔧 常见问题

### Q1: 编译报错"Missing \$ inserted"

**原因**: 数学符号未在数学环境中

**解决**: 检查下划线 `_`，应为 `\_` 或在 `$...$` 中

### Q2: 中文显示乱码

**原因**: 编译器不对

**解决**: 使用 `xelatex` 而非 `pdflatex`

### Q3: 参考文献编号不连续

**原因**: 需要编译两次

**解决**: 运行两次 `xelatex main.tex`

### Q4: 图片无法显示

**原因**: 图片路径错误或图片不存在

**解决**: 
1. 创建 `figures/` 目录
2. 将图片放入该目录
3. 使用 `\includegraphics{figures/xxx.png}`

## 📚 参考文献数量

当前已包含：**57篇参考文献**

- 作物倒伏监测：6篇
- RGB-D语义分割：8篇  
- 外观不变性与域泛化：6篇
- 深度学习基础：10篇
- Transformer与注意力：4篇
- 农业遥感：5篇
- 深度估计：2篇
- 其他相关：16篇

符合要求（≥20篇），且引用规范。

## 🎯 使用建议

1. **首次使用**: 推荐Overleaf，无需配置环境
2. **本地编辑**: 使用VS Code + LaTeX Workshop插件
3. **版本管理**: 建议用Git管理LaTeX源文件
4. **协作编辑**: Overleaf支持多人协作

## 📞 技术支持

遇到LaTeX编译问题，可查阅：
- Overleaf文档: https://www.overleaf.com/learn
- LaTeX Stack Exchange: https://tex.stackexchange.com/

## ✅ 检查清单

提交前确认：
- [ ] 个人信息已填写
- [ ] 时间节点已更新为实际时间
- [ ] 导师信息已填写
- [ ] 所有公式编号正确
- [ ] 表格和图片可正常显示
- [ ] 参考文献格式统一
- [ ] 编译无错误和警告
- [ ] PDF输出正常

## 🎉 完成

现在您可以：
1. 上传到Overleaf编译
2. 或在本地用 `xelatex` 编译
3. 生成PDF后根据需要微调格式
4. 填写个人信息后提交

祝开题顺利！🎓

