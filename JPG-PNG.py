import os
from PIL import Image


def convert_jpg_to_png(source_directory, target_directory):
    # 确保目标文件夹存在，如果不存在，则创建它
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 遍历源目录下的所有文件
    for filename in os.listdir(source_directory):
        # 检查文件扩展名是否为.jpg
        if filename.lower().endswith(".jpg"):
            # 构建完整的源文件路径
            file_path = os.path.join(source_directory, filename)

            # 打开并读取图像文件
            img = Image.open(file_path)

            # 构建目标文件路径，将后缀名改为.png
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(target_directory, png_filename)

            # 保存图像为PNG格式
            img.save(png_path, "PNG")

            # 关闭图像文件
            img.close()
            print(f"Converted and saved: {file_path} to {png_path}")


# 使用示例：替换以下路径为你的源文件夹和目标文件夹路径
convert_jpg_to_png(r'C:\Users\11504\Desktop\RN-Puzzle\dataset\coco-train-256-mask', r'C:\Users\11504\Desktop\RN-Puzzle\dataset\coco-train-256-mask-png')
