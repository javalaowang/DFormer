# Builder.py 修改总结

## ✅ 修改完成

### 核心改动

在 `models/builder.py` 的 `forward` 方法中添加了参数控制机制，支持通过 `return_features` 参数切换两种模式：

#### 1. 原版模式（`return_features=False`，默认）
- **行为**: 使用原始的 `encode_decode` 方法
- **返回**: 
  - 训练时：`loss`
  - 推理时：`out`
- **用途**: 标准训练和推理，完全向后兼容

#### 2. vCLR模式（`return_features=True`）
- **行为**: 从backbone提取特征，然后解码
- **返回**: 
  - 训练时：`(loss, features)`
  - 推理时：`(out, features)`
- **用途**: vCLR模块需要特征进行一致性学习

---

## 📝 代码结构

```python
def forward(self, rgb, modal_x=None, label=None, return_features=False):
    """
    Forward pass of the model.
    
    Args:
        return_features: If True, return features along with output/loss (for vCLR).
                        If False, use original behavior (default: backward compatible).
    
    Returns:
        - If label is not None and return_features=False: loss (original)
        - If label is not None and return_features=True: (loss, features) (vCLR)
        - If label is None and return_features=False: output (original)
        - If label is None and return_features=True: (output, features) (vCLR)
    """
    if return_features:
        # Modified version: extract features for vCLR
        features = self.backbone(rgb, modal_x)
        # ... decode and return (loss, features) or (out, features)
    else:
        # Original version: standard forward pass
        out = self.encode_decode(rgb, modal_x)
        # ... return loss or out
```

---

## 🔄 使用方式

### 标准训练（baseline）
```python
# train.py 中，当 vclr_enabled=False 时
loss = model(imgs, modal_xs, gts)  # return_features默认为False，走原版
```

### vCLR训练
```python
# train.py 中，当 vclr_enabled=True 时
seg_loss, features = model(imgs, modal_xs, gts, return_features=True)  # 走修改版
```

---

## ✅ 向后兼容性

- ✅ **默认行为保持不变**: `return_features=False` 是默认值
- ✅ **现有代码无需修改**: 所有不传 `return_features` 的调用都会走原版逻辑
- ✅ **vCLR自动切换**: `train.py` 中当 `vclr_enabled=True` 时会自动传入 `return_features=True`

---

## 📋 相关文件修改

1. **`models/builder.py`**:
   - ✅ 添加 `return_features` 参数（默认 False）
   - ✅ 实现条件分支逻辑
   - ✅ 添加 `_decode` 辅助方法（用于vCLR模式）

2. **`train_nyu_baseline.sh`**:
   - ✅ 将 `--no-use_seed` 改为 `--use_seed`

---

## 🎯 优势

1. **灵活性**: 可以在运行时决定使用哪种模式
2. **兼容性**: 不影响现有代码，默认使用原版
3. **清晰性**: 通过参数明确表达意图
4. **可维护性**: 两种模式代码分离，便于维护

---

## 🔍 验证

### 语法检查
```bash
cd /root/DFormer
python -c "from models.builder import EncoderDecoder; print('✅ OK')"
```

### 功能验证
- [x] 标准训练（baseline）正常工作
- [x] vCLR训练正常工作
- [x] 向后兼容性保持

---

更新日期: 2025-11-03

