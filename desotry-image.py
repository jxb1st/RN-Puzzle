import numpy as np
import os
import cv2
import glob

def put_mask(img_folder, mask_folder, output_folder):
    # 1. 获取所有图片和mask的路径
    img_paths = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))  # 假设图片是jpg格式
    mask_paths = sorted(glob.glob(os.path.join(mask_folder, '*.png')))  # 假设mask是png格式

    # 检查图片和mask数量是否匹配
    if len(img_paths) != len(mask_paths):
        print("Error: The number of images and masks do not match!")
        return

    # 2. 遍历图片和mask列表
    for img_path, mask_path in zip(img_paths, mask_paths):
        # 读取原图
        image = cv2.imread(img_path)
        if image is None:
            print("Failed to read image:", img_path)
            continue

        # 读取 mask 图，并缩放到与原图相同的尺寸
        mask = cv2.imread(mask_path)
        if mask is None:
            print("Failed to read mask:", mask_path)
            continue
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # 3. 将 mask 图与原图融合
        alpha = 1
        beta = 1
        gamma = 0
        mask_img = cv2.addWeighted(image, alpha, mask_resized, beta, gamma)

        # 4. 获取输出文件名和路径
        basename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, basename)

        # 保存融合后的图像到指定路径
        cv2.imwrite(output_path, mask_img)
        print("Saved blended image to:", output_path)

# 输入原图文件夹、mask文件夹和输出文件夹路径
img_folder = r'C:\Users\11504\Desktop\RN-Puzzle\temp-image'
mask_folder = r'C:\Users\11504\Desktop\RN-Puzzle\temp-mask'
output_folder = r'C:\Users\11504\Desktop\RN-Puzzle\destoryed-image'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 处理图片和mask
put_mask(img_folder, mask_folder, output_folder)
