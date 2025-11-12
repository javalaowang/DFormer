# NYUDepth v2 和 SUN RGB-D 数据集下载（更正版）

## ⚠️ 重要更正

之前提供的链接是**预训练模型**，不是数据集！

正确的数据集下载链接如下：

---

## 📥 数据集下载链接（DFormer项目提供）

根据DFormer项目的README，**数据集下载链接**在第125行：

### 统一的数据集下载链接

| 数据集类型 | GoogleDrive | OneDrive | 百度网盘 |
|-----------|------------|----------|---------|
| **所有数据集** | [链接](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [链接](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [链接](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) |

**注意**：
- 这个链接包含**所有数据集**（NYUDepth v2、SUN RGB-D等）
- 密码（百度网盘）：`ij7q`

---

## 🔗 官方原始数据集下载（如果需要原始数据）

### NYUDepth v2 官方下载

- **官方网站**: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
- **说明**: 需要注册并填写申请表格
- **数据格式**: 原始格式，可能需要转换

### SUN RGB-D 官方下载

- **官方网站**: http://rgbd.cs.princeton.edu/
- **说明**: 需要注册并填写申请表格
- **数据格式**: 原始格式，可能需要转换

---

## ✅ 推荐方案：使用DFormer预处理好的数据集

**强烈推荐**：使用DFormer项目提供的预处理好的数据集，因为：

1. ✅ **已经预处理**
   - 深度图已从`.npy`转换为`.png`
   - 文件路径已重新组织
   - 已包含分割文件（train.txt, test.txt）

2. ✅ **直接可用**
   - 无需自己预处理
   - 格式与项目完全匹配

3. ✅ **节省时间**
   - 避免预处理步骤
   - 直接开始训练

---

## 📋 下载步骤

### 方式1：使用统一的数据集链接（推荐）

#### 步骤1：下载

选择一个平台下载：
- **百度网盘**（推荐，国内速度快）:
  - 链接: https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q
  - 密码: `ij7q`

- **Google Drive**:
  - https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing

- **OneDrive**:
  - https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob

#### 步骤2：解压和放置

```bash
cd /root/DFormer/datasets

# 解压下载的压缩包
unzip <下载的文件>.zip -d .

# 或如果是tar.gz
tar -xzf <下载的文件>.tar.gz -C .
```

#### 步骤3：验证结构

下载后应该看到：
```
datasets/
├── NYUDepthv2/
│   ├── RGB/
│   ├── Label/
│   ├── Depth/
│   ├── train.txt
│   └── test.txt
│
└── SUNRGBD/
    ├── RGB/
    ├── labels/      # 注意是小写labels
    ├── Depth/
    ├── train.txt
    └── test.txt
```

---

## 📊 数据集信息

### NYUDepth v2
- **训练集**: 795张图像
- **测试集**: 654张图像
- **类别数**: 40类
- **图像尺寸**: 640×480

### SUN RGB-D
- **训练集**: 5285张图像
- **测试集**: 5050张图像
- **类别数**: 37类
- **图像尺寸**: 480×480

---

## ⚠️ 之前错误信息更正

### ❌ 错误的链接（预训练模型）
以下链接是**预训练模型**，不是数据集：

- NYUDepth v2 预训练模型: 
  - GoogleDrive: https://drive.google.com/drive/folders/1P5HwnAvifEI6xiTAx6id24FUCt_i7GH8
  - 百度网盘: https://pan.baidu.com/s/1AkvlsAvJPv21bz2sXlrADQ?pwd=6vuu

- SUN RGB-D 预训练模型:
  - GoogleDrive: https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8
  - 百度网盘: https://pan.baidu.com/s/1D6UMiBv6fApV5lafo9J04w?pwd=7ewv

### ✅ 正确的链接（数据集）

**统一的数据集下载链接**：
- **百度网盘**: https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q
- **Google Drive**: https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing
- **OneDrive**: https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob

---

## 🔍 如何在DFormer README中区分数据集和预训练模型

### 数据集链接
在README中的位置：第121-126行
```markdown
- **Datasets:** 
  | Datasets | [GoogleDrive](...) | [OneDrive](...) | [BaiduNetdisk](...) |
```

### 预训练模型链接
在README中的位置：第132行之后
```markdown
- **Checkpoints:**
  | NYUDepth v2 | [GoogleDrive](...) | ... |
  | SUNRGBD | [GoogleDrive](...) | ... |
```

**关键区别**：
- **"Datasets"** = 数据集
- **"Checkpoints"** / **"Weights"** / **"trained"** = 预训练模型

---

## 📝 v-CLR项目说明

v-CLR项目主要针对：
- COCO数据集
- Open-world instance segmentation

**不涉及NYUDepth v2和SUN RGB-D**，所以v-CLR项目的README中没有这两个数据集的下载信息。

---

## ✅ 总结

### 正确的数据集下载方式：

1. ✅ **使用DFormer提供的统一数据集链接**
   - 百度网盘: https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q
   - 包含所有数据集（NYUDepth v2、SUN RGB-D等）

2. ✅ **下载后解压到 `datasets/` 目录**

3. ✅ **验证目录结构是否正确**

### 如果遇到问题：

- 如果统一链接中找不到特定数据集，可以尝试官方下载链接
- 官方链接可能需要注册和申请
- 官方数据可能需要预处理（DFormer的README第128行说明了预处理方法）

---

**下一步**: 使用正确的数据集链接下载后，告诉我，我可以帮你验证和配置！

