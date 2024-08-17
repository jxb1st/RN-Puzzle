import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *
import torch.nn.functional as F

from rn import RN_B, RN_L


class G_Net(nn.Module):
    # 用于 RN_L 层的阈值
    def __init__(self, input_channels, residual_blocks, threshold):
        super(G_Net, self).__init__()

        # Encoder 由3个卷积层、归一化层（RN_B）和激活层（ReLU）组成。
        """
        输入：带有掩码的图像。
        输出：填补缺失区域后的图像。
        架构：编码器 -> 中间残差块 -> 解码器。
        """
        # 对输入图像进行边界填充，以避免卷积操作时边缘信息丢失
        self.encoder_prePad = nn.ReflectionPad2d(3)
        # 3个卷积层逐步增加通道数，并减少空间分辨率。
        self.encoder_conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=0) # [1, 3, 256, 256] -> [1, 64, 250, 250] assume batch == 1
        # 归一化层（具体实现需参考 RN_B 类定义）
        self.encoder_in1 = RN_B(feature_channels=64)
        self.encoder_relu1 = nn.ReLU(True)
        self.encoder_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1) # [1, 64, 250, 250] -> [1, 128, 125, 125]
        self.encoder_in2 = RN_B(feature_channels=128)
        self.encoder_relu2 = nn.ReLU(True)
        self.encoder_conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1) # [1, 128, 125, 125] -> [1, 256, 62, 62]
        self.encoder_in3 = RN_B(feature_channels=256)
        self.encoder_relu3 = nn.ReLU(True)


        # Middle 中间部分: 包含多个自定义的残差块（saRN_ResnetBlock）。
        blocks = []
        for _ in range(residual_blocks):
            # block = ResnetBlock(256, 2, use_spectral_norm=False)
            block = saRN_ResnetBlock(256, dilation=2, threshold=threshold, use_spectral_norm=False)
            blocks.append(block)
        # Sequential: 将多个残差块组合成一个序列。
        self.middle = nn.Sequential(*blocks)


        # Decoder 包含反卷积层（通过像素重排实现），归一化层（RN_L）和激活层（ReLU），负责将编码器和中间层处理过的特征图恢复为原始输入图像的大小和通道数
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), # 像素重排操作，将特征图的空间分辨率扩大2倍，同时减少通道数。这一步将输出通道数从512减少到128，同时将空间分辨率扩大2倍。
            RN_L(128),# 自定义的归一化层
            nn.ReLU(True),

            nn.Conv2d(128, 64*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), # 像素重排操作，将特征图的空间分辨率再次扩大2倍，同时减少通道数。这一步将输出通道数从256减少到64，同时将空间分辨率扩大2倍。
            RN_L(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=input_channels, kernel_size=7, padding=0) # 输入通道数为64，输出通道数为原始输入的通道数（input_channels），卷积核大小为7x7，填充为0。这一步将特征图恢复为原始输入图像的通道数。

        )


    def encoder(self, x, mask):
        print("Original x.shape: ", x.shape, "Original mask.shape: ", mask.shape)
        x = self.encoder_prePad(x)
        print("After prepad x.shape: ", x.shape, "After prepad mask.shape: ", mask.shape)
        x = self.encoder_conv1(x)
        print("After encoder_conv1 x.shape: ", x.shape, "After encoder_conv1 mask.shape: ", mask.shape)
        x = self.encoder_in1(x, mask)
        print("After encoder_in1 x.shape: ", x.shape, "After encoder_in1 mask.shape: ", mask.shape)
        x = self.encoder_relu2(x)

        x = self.encoder_conv2(x)
        x = self.encoder_in2(x, mask)
        x = self.encoder_relu2(x)

        x = self.encoder_conv3(x)
        x = self.encoder_in3(x, mask)
        x = self.encoder_relu3(x)
        print("This is x.shape:", x.shape)
        return x

    # 输入图像 x 和掩码 mask。将图像与掩码进行组合并传入编码器。经过中间的残差块处理后，传入解码器。输出通过 tanh 激活函数处理并缩放至 [0, 1]。
    def forward(self, x, mask):
        gt = x
        x = (x * (1 - mask).float()) + mask # 将gt和mask合并
        # input mask: 1 for hole, 0 for valid
        x = self.encoder(x, mask) # 输入图像 x 和掩码 mask 通过编码器处理。编码器的作用是提取图像的特征，将图像从高分辨率映射到低分辨率的特征空间。
        print("this is x size after encoder: ", x.shape)
        x = self.middle(x) #  经过多个自定义的残差块（saRN_ResnetBlock）处理。残差块的作用是通过跳跃连接（skip connections）保留输入信息，同时进行更深层次的特征提取和处理。
        print("this is x size after middle: ", x.shape)
        x = self.decoder(x) # 经过中间处理的特征图 x 通过解码器处理。解码器的作用是将特征图从低分辨率恢复到高分辨率，输出与输入图像大小一致的图像。
        print("this is x size after decoder: ", x.shape)
        x = (torch.tanh(x) + 1) / 2 # 将范围从 [-1, 1] 缩放到 [0, 1]，使其与图像像素值范围一致。
        # x = x*mask+gt*(1-mask)
        return x


# original D 判别输入图像是真实图像还是生成器生成的伪造图像
class D_Net(nn.Module):
    # 是否使用 sigmoid 激活函数。是否使用谱归一化
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(D_Net, self).__init__()
        self.use_sigmoid = use_sigmoid #  是否在网络的最后一层使用 Sigmoid 激活函数。Sigmoid 激活函数将输出值限制在 [0, 1] 之间，适用于二分类任务。


        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 最后一层，将特征图的通道数降到1。输出一个单通道的特征图，用于最终的判别结果。
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    # 经过5个卷积层，每个卷积层之后是 LeakyReLU 激活函数。 最后输出判别结果和各层的特征图
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]





class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            # 反射填充，用于在卷积前增加边界像素，避免边界信息丢失。
            nn.ReflectionPad2d(dilation),
            # spectral_norm: 谱归一化，用于稳定训练
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # 实例归一化，标准化操作，减少不同图像间的风格差异。
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )
    # 跳跃连接（Residual Connection）: 将输入 x 直接加到卷积块的输出上，实现了跳跃连接。
    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


class saRN_ResnetBlock(nn.Module):
    # dim: 输入和输出的通道数。
    # dilation: 卷积核的扩张率。
    # use_spectral_norm: 是否使用谱归一化。
    def __init__(self, dim, dilation, threshold, use_spectral_norm=True):
        super(saRN_ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            RN_L(feature_channels=256, threshold=threshold),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
            RN_L(feature_channels=dim, threshold=threshold),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        # skimage.io.imsave('block.png', out[0].detach().permute(1,2,0).cpu().numpy()[:,:,0])

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out
#  用于条件地应用谱归一化。
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    mask = torch.randn(1, 1, 256, 256)
    model = G_Net(input_channels=3, residual_blocks=3, threshold=0.5)
    output = model(x, mask)
    print(output.shape)
    print("No Abnormal!")
