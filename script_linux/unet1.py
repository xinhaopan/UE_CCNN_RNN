# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 23:18:04 2021

@author: Xhpan
"""

import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

from osgeo import gdal

# ## 读入数据

# In[2]:


class DeeplabDataset(Dataset):
    def __init__(self, train_lines, image_size, num_classes, dataset_path):
        super(DeeplabDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.num_classes = num_classes
        self.dataset_path = dataset_path

    def __len__(self):
        return self.train_batches

    def __getitem__(self, index):
        # 打乱顺序
        annotation_line = self.train_lines[index]
        name = annotation_line.split()[0]
        # 从文件中读取图像
        jpg = self.readTif(os.path.join(os.path.join(self.dataset_path, "images"), name + ".tif"))
        png = self.readTif(os.path.join(os.path.join(self.dataset_path, "labels"), name + ".tif"))

        # # 从文件中读取图像
        png = np.array(png, dtype=np.uint8)

        # 得到one-hot编码
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.image_size[1]), int(self.image_size[0]), self.num_classes + 1))
        return jpg, png, seg_labels, name

    def readTif(self, fileName):
        dataset = gdal.Open(fileName)
        if dataset == None:
            print(fileName + "文件无法打开")
            return None
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
        return im_data


# DataLoader中collate_fn使用
def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    names = []
    for img, png, labels, name in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
        names.append(name)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels, names


# ## unet模型

# In[3]:


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features


def make_layers(cfg, batch_norm=True, in_channels=3):
    layers = []  # (卷积 + 标准化 + elu) *3 + dropout + maxpool
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                # 在vgg加标准化,卷积，激活函数
                # layers += [conv2d, nn.Dropout(p=0.2), nn.BatchNorm2d(v), nn.ELU(inplace=True)]
                # layers += [conv2d, nn.BatchNorm2d(v), nn.ELU(inplace=True)]
                layers += [conv2d, nn.ELU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [32, 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
}  # 数字表示通道数，M表示进行最大池化


def VGG16(in_channels, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=True, in_channels=in_channels), **kwargs)
    # del model.avgpool
    # del model.classifier
    return model


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.elu2 = nn.ELU(inplace=True)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.elu3 = nn.ELU(inplace=True)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        

    def forward(self, inputs1, inputs2):  # 合并 + （卷积 + 标准化 + elu） * 3 + dropout + 上采样

        outputs = torch.cat([inputs1, self.up(inputs2)], 1)

        outputs = self.conv1(outputs)
        outputs = self.elu1(outputs)

        outputs = self.conv2(outputs)
        outputs = self.elu2(outputs)

        outputs = self.conv3(outputs)
        outputs = self.elu3(outputs)

        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3):
        super(Unet, self).__init__()
        self.vgg = VGG16(in_channels=in_channels)  # 建立主干提取网络
        in_filters = [96, 192, 384, 768]
        out_filters = [32, 64, 128, 256]
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        # 拿出主干特征提取的结果
        feat1 = self.vgg.features[:6](inputs)
        feat2 = self.vgg.features[6:13](feat1)
        feat3 = self.vgg.features[13:20](feat2)
        feat4 = self.vgg.features[20:27](feat3)
        feat5 = self.vgg.features[27:-1](feat4)

        # 上采样后合并
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)
        final = F.dropout(final, p=0.2)

        return final

    def _initialize_weights(self, *stages):  # 模型参数随机化
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()


# ## 模型训练相关函数

# In[4]:




# ## 精度评价函数

# In[5]:


def CE_Loss(inputs, target, num_classes=21):
    n, c, h, w = inputs.size()
    nt,ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim = -1), temp_target)
    return CE_loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs,threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


# ## 画图

# In[6]:


class LossHistory():
    def __init__(self, save_path,time_str):
        self.time_str   = time_str
        self.save_path  = save_path
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_train_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

