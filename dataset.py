import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
# from scipy.misc import imread
import imageio
import cv2
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from torch.utils.data import DataLoader

# 将图像调整为256x256大小并转换为Tensor格式。
def my_transforms():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform

# 初始化数据集对象。接收文件列表、掩码文件列表、是否进行数据增强、是否处于训练模式以及输入图像的大小等参数，并加载文件列表。
class Dataset(torch.utils.data.Dataset):
    def __init__(self, flist, mask_flist, augment, training, input_size):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        # print("This is the data from flist: ", self.data)
        self.mask_data = self.load_flist(mask_flist)
        self.input_size = input_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        # try:
        #     item = self.load_item(index)
        # except:
        #     print('loading error: ' + self.data[index])
        #     item = self.load_item(0)

        return item

    # 加载指定索引处图像的文件名。
    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    # 加载和预处理指定索引处的图像及其掩码。包括读取图像文件、灰度图像转为RGB、调整大小、数据增强等。
    def load_item(self, index):

        size = self.input_size
        # print("This is self.data[index]: ", self.data[index])

        # imgfiles = os.listdir(self.data[index])

        # load image
        img = imageio.imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if self.training:
            if size != 0:
                img = self.resize(img, size, size)

        # load mask
        mask = self.load_mask(img, index)

        # augment data 如果启用了数据增强，并且随机生成的二项分布值大于0，则对图像和掩码进行左右翻转。
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask = mask[:, ::-1, ...]
        # 返回图像的Tensor，掩码的Tensor，以及索引index
        return self.to_tensor(img), self.to_tensor(mask), index


    def load_mask(self, img, index):
        # 获取输入图像的高度（imgh）和宽度（imgw）。
        imgh, imgw = img.shape[0:2]

        # external
        # 如果处于训练模式，随机选择一个掩码索引。
        if self.training:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imageio.imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
        else:   # in test mode, there's a one-to-one relationship between mask and image; masks are loaded non random 如果处于测试模式，根据输入图像的索引加载对应的掩码。
            # mask = 255 - imread(self.mask_data[index])[:,:,0]    # ICME original (H,W,3) mask: 0 for hole
            mask = imageio.imread(self.mask_data[index])   # mask must be 255 for hole in this InpaintingModel
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            if len(mask.shape) == 3:
                mask = rgb2gray(mask)
        mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation 返回经过预处理的掩码图像。
        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        # print(type(img))    # imageio.core.util.Array
        # img = scipy.misc.imresize(img, [height, width])
        img = cv2.resize(img, (height, width))

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                print("hahaha")
                return flist

            if os.path.isfile(flist):
                try:
                    with open(flist, 'r') as f:
                        file_list = [line.strip() for line in f if line.strip()]
                    print("Loaded", len(file_list), "files from file", flist)
                    return file_list
                except Exception as e:
                    print("Error loading file list from", flist, ":", e)
                    return [flist]

        print("Invalid flist:", flist)
        return []


def build_dataloader(flist, mask_flist, augment, training, input_size, batch_size, \
num_workers, shuffle):

    dataset = Dataset(
        flist=flist,
        mask_flist=mask_flist,
        augment=augment,
        training=training,
        input_size=input_size
        )

    print('Total instance number:', dataset.__len__())

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle
    )

    return dataloader