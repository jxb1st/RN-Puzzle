import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import torchvision.models as models
# from networks import G_Net, D_Net
from networks_vit import G_Net, D_Net
from module_util import print_network
# 对抗性损失类
class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        # ：用于注册常量，不作为模型参数更新
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        # 标准GAN，使用二元交叉熵损失（BCELoss）
        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        # lsgan：最小二乘GAN，使用均方误差损失（MSELoss）
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        # hinge：Hinge损失，使用ReLU。
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        # 对于 hinge 损失，根据是否是判别器的损失来处理。
        # 对于 nsgan 和 lsgan，根据标签（真实或假）计算损失。
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class InpaintingModel(nn.Module):
    def __init__(self, g_lr, d_lr, l1_weight, gan_weight, iter=0, threshold=None):
        super(InpaintingModel, self).__init__()

        # 实例化生成器和判别器，以及相应的优化器和损失函数。
        self.generator = G_Net(input_channels=3, residual_blocks=8, threshold=threshold)
        self.discriminator = D_Net(in_channels=3, use_sigmoid=True)

        # L1Loss：像素级的L1损失，用于衡量生成图像与真实图像的差异。
        self.l1_loss = nn.L1Loss()
        # AdversarialLoss：对抗性损失，用于衡量生成器和判别器之间的对抗性表现。
        self.adversarial_loss = AdversarialLoss('nsgan')
        # 生成器和判别器的loss
        self.g_lr, self.d_lr = g_lr, d_lr

        self.l1_weight, self.gan_weight = l1_weight, gan_weight

        # 全局迭代次数，用于记录训练的进度
        self.global_iter = iter
        # 使用Adam优化器来更新生成器的参数
        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=float(self.g_lr),
            betas=(0., 0.9)
        )
        # # 使用Adam优化器来更新判别器的参数
        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(self.d_lr),
            betas=(0., 0.9)
        )


if __name__ == '__main__':
    model = InpaintingModel(g_lr=0.0001, d_lr=(0.1 * 0.0001), l1_weight=1.0, gan_weight=0.1,
                            iter=0, threshold=0.8)
    print('---------- Networks architecture -------------')
    print("Generator:")
    print_network(model.generator)
    print("Discriminator:")
    print_network(model.discriminator)
    print('----------------------------------------------')
