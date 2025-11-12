# NYUDepth v2 和 SUN RGB-D 数据集准备指南

## 📍 数据集位置

### 当前状态
- ❌ **NYUDepth v2**: 未下载（需要下载）
- ❌ **SUN RGB-D**: 未下载（需要下载）
- ✅ **Wheatlodgingdata**: 已存在 (`datasets/Wheatlodgingdata/`)

### 目标目录结构
```
datasets/
├── Wheatlodgingdata/      ✅ 已有
│   ├── RGB/
│   ├── Label/
│   ├── HHA/
│   ├── train.txt
│   └── test.txt
│
├── NYUDepthv2/            ⏳ 需要下载
│   ├── RGB/               # RGB图像
│   ├── Label/             # 标注图像
│   ├── Depth/             # 深度图
│   ├── train.txt          # 训练集列表
│   └── test.txt           # 测试集列表
│
└── SUNRGBD/               ⏳ 需要下载
    ├── RGB/               # RGB图像
    ├── labels/            # 标注图像（注意是labels，不是Label）
    ├── Depth/             # 深度图
    ├── train.txt          # 训练集列表
    └── test.txt           # 测试集列表
```

---

## 🔗 数据集下载链接

### ⚠️ 重要更正：之前提供的链接是预训练模型！

### NYUDepth v2 数据集（正确链接）

**统一的数据集下载链接**（包含所有数据集）：

1. **百度网盘**（推荐）:
   - 链接: https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q
   - 密码: `ij7q`

2. **Google Drive**:
   - https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing

3. **OneDrive**:
   - https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob

**注意**: 这个链接包含所有数据集（NYUDepth v2、SUN RGB-D等），下载后解压即可。

**数据集信息**:
- 训练集: 795张图像
- 测试集: 654张图像
- 类别数: 40类
- 图像尺寸: 640×480

---

### SUN RGB-D 数据集（正确链接）

**与NYUDepth v2使用同一个下载链接**（统一的数据集包）：

1. **百度网盘**（推荐）:
   - 链接: https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q
   - 密码: `ij7q`

2. **Google Drive**:
   - https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing

3. **OneDrive**:
   - https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob

**注意**: 这个链接包含所有数据集，下载后应该同时包含NYUDepth v2和SUN RGB-D。

**数据集信息**:
- 训练集: 5285张图像
- 测试集: 5050张图像
- 类别数: 37类
- 图像尺寸: 480×480

---

## 📥 下载和准备步骤

### 步骤1：下载数据集

选择任一方式下载：

#### 方式A：使用gdown（Google Drive）
```bash
cd /root/DFormer/datasets

# 下载NYUDepth v2（需要根据实际分享链接调整）
pip install gdown
gdown --folder <GoogleDriveFolderID>  # 需要从链接中提取ID

# 下载SUN RGB-D
gdown --folder <GoogleDriveFolderID>
```

#### 方式B：手动下载（推荐）
1. 从上面的链接下载压缩包
2. 解压到 `datasets/` 目录

#### 方式C：使用wget（如果有直接下载链接）
```bash
cd /root/DFormer/datasets
# 下载NYUDepth v2
wget <下载链接> -O NYUDepthv2.zip
unzip NYUDepthv2.zip -d NYUDepthv2/

# 下载SUN RGB-D
wget <下载链接> -O SUNRGBD.zip
unzip SUNRGBD.zip -d SUNRGBD/
```

---

### 步骤2：验证数据集结构

下载后，验证目录结构是否正确：

```bash
# 检查NYUDepth v2
cd /root/DFormer/datasets/NYUDepthv2
ls -la
# 应该看到：RGB/, Label/, Depth/, train.txt, test.txt

# 检查SUN RGB-D
cd /root/DFormer/datasets/SUNRGBD
ls -la
# 应该看到：RGB/, labels/, Depth/, train.txt, test.txt
```

**注意**: SUN RGB-D的标注文件夹是 `labels`（小写），不是 `Label`！

---

### 步骤3：检查数据文件

```bash
# 检查NYUDepth v2
echo "NYUDepth v2:"
echo "Train images: $(wc -l < datasets/NYUDepthv2/train.txt)"
echo "Test images: $(wc -l < datasets/NYUDepthv2/test.txt)"
echo "RGB images: $(ls datasets/NYUDepthv2/RGB | wc -l)"
echo "Depth images: $(ls datasets/NYUDepthv2/Depth | wc -l)"

# 检查SUN RGB-D
echo "SUN RGB-D:"
echo "Train images: $(wc -l < datasets/SUNRGBD/train.txt)"
echo "Test images: $(wc -l < datasets/SUNRGBD/test.txt)"
echo "RGB images: $(ls datasets/SUNRGBD/RGB | wc -l)"
echo "Depth images: $(ls datasets/SUNRGBD/Depth | wc -l)"
```

---

## 🚀 快速开始：创建vCLR配置文件

### 创建NYUDepth v2的vCLR配置

我将为你创建配置文件。

---

## 📝 数据集信息总结

| 数据集 | 训练集 | 测试集 | 类别数 | 尺寸 | 状态 |
|--------|--------|--------|--------|------|------|
| **NYUDepth v2** | 795 | 654 | 40 | 640×480 | ⏳ 需要下载 |
| **SUN RGB-D** | 5285 | 5050 | 37 | 480×480 | ⏳ 需要下载 |
| **Wheatlodgingdata** | 357 | 153 | 3 | 500×500 | ✅ 已有 |

---

## ⚠️ 重要提示

### 数据集格式要求

根据README说明，这些数据集已经过预处理：
- 深度图已从 `.npy` 转换为 `.png`
- 文件路径已重新组织为清晰格式
- 已包含分割文件（train.txt, test.txt）

**不需要自己预处理**，直接下载使用即可！

---

## 🎯 下载后的下一步

1. ✅ 下载数据集（选择任一链接）
2. ✅ 解压到 `datasets/` 目录
3. ✅ 验证目录结构
4. ⏳ 创建vCLR配置文件（我将帮你创建）
5. ⏳ 开始训练实验

---

## 💡 如果下载遇到问题

### 常见问题：

1. **Google Drive下载慢**
   - 使用百度网盘或OneDrive
   - 或使用gdown工具

2. **百度网盘需要登录**
   - 使用百度网盘客户端下载
   - 或使用第三方下载工具

3. **数据集很大，下载时间长**
   - NYUDepth v2: 约几个GB
   - SUN RGB-D: 约十几个GB
   - 建议使用稳定的网络环境

---

**下一步**: 下载数据集后，告诉我，我将帮你创建vCLR配置文件！

