# the code is used to generate a regular mask with 256 * 256 pixels
from PIL import Image
import random

# 创建一个512x512的黑色图像
img_size = 256
block_size = img_size // 3  # 计算每个小块的大小
img = Image.new('RGB', (img_size, img_size), 'black')

# 生成一个白色的小块
white_block = Image.new('RGB', (block_size, block_size), 'white')

# 随机选择一个小块的位置
x = random.randint(0, 2) * block_size
y = random.randint(0, 2) * block_size

# 将白色小块粘贴到图像上
img.paste(white_block, (x, y))

# 保存图像
img.save(r'C:\Users\11504\Desktop\RN-Puzzle\temp-mask/masked_image.png')

# 显示图像（如果在支持显示的环境中运行）
img.show()
