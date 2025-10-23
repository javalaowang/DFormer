#!/usr/bin/env python3
"""
将HHA的TIF格式转换为PNG格式，以便DFormer使用
"""
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_hha_to_png(input_dir, output_dir):
    """
    将HHA文件夹中的TIF文件转换为PNG文件
    
    Args:
        input_dir: 输入HHA文件夹路径
        output_dir: 输出PNG文件夹路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有TIF文件
    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    tif_files.sort()
    
    print(f"找到 {len(tif_files)} 个TIF文件")
    
    for tif_file in tqdm(tif_files, desc="转换HHA文件"):
        input_path = os.path.join(input_dir, tif_file)
        output_file = tif_file.replace('.tif', '.png')
        output_path = os.path.join(output_dir, output_file)
        
        try:
            # 使用PIL读取TIF文件
            img = Image.open(input_path)
            img_array = np.array(img)
            
            # 如果是单通道，转换为3通道
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array, img_array, img_array], axis=2)
            
            # 归一化到0-255范围
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                img_array = (img_array * 255).astype(np.uint8)
            elif img_array.max() > 255:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
            
            # 保存为PNG
            cv2.imwrite(output_path, img_array)
            
        except Exception as e:
            print(f"转换 {tif_file} 时出错: {e}")
            continue
    
    print(f"转换完成！PNG文件保存在: {output_dir}")

if __name__ == "__main__":
    # 设置路径
    input_hha_dir = "/root/RGBX_transformer_500epoch/datasets/Wheatlodgingdata/HHA"
    output_hha_dir = "/root/DFormer/datasets/Wheatlodgingdata/HHA"
    
    # 创建输出目录
    os.makedirs(output_hha_dir, exist_ok=True)
    
    # 执行转换
    convert_hha_to_png(input_hha_dir, output_hha_dir)
    
    print("HHA格式转换完成！")
