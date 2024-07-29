#!/usr/bin/env python
# coding: utf-8

# 三次，无标准化
# # 函数定义

# In[1]:


import os
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import random
from osgeo import gdal
import datetime

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
                layers += [conv2d,nn.Dropout(p=0.2),nn.ELU(inplace=True)]
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
        outputs = F.dropout(outputs,p=0.2)
        
        outputs = self.conv2(outputs)
        outputs = self.elu2(outputs)
        outputs = F.dropout(outputs,p=0.2)

        outputs = self.conv3(outputs)
        outputs = self.elu3(outputs)
        outputs = F.dropout(outputs,p=0.2)

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
        feat1 = self.vgg.features[:9](inputs)
        feat2 = self.vgg.features[9:19](feat1)
        feat3 = self.vgg.features[19:29](feat2)
        feat4 = self.vgg.features[29:39](feat3)
        feat5 = self.vgg.features[39:-1](feat4)

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


# 训练一个epoch
def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda, save_path):
    net = net.train()
    total_loss = 0
    total_f_score = 0

    val_toal_loss = 0
    val_total_f_score = 0
    start_time = time.time()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs, pngs, labels, _ = batch
            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).type(torch.FloatTensor).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
            # set_trace()
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                's/step': waste_time,
                                'lr': get_lr(optimizer)})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels, _ = batch
            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).type(torch.FloatTensor).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

                outputs = net(imgs)
                val_loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    val_loss = val_loss + main_dice
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'f_score': val_total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    loss_history.append_loss(total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    pthpath = save_path  + "/pth"
    if not os.path.exists(pthpath):
        os.makedirs(pthpath)
    
    torch.save(model.state_dict(), pthpath + '/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))
            
# 获取当前的学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ## 精度评价函数

# In[5]:


def CE_Loss(inputs, target, num_classes=21):
    n, c, h, w = inputs.size()
    nt,ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    # CE_loss  = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim = -1), temp_target)
    CE_loss = F.binary_cross_entropy_with_logits(temp_inputs[:,0], temp_target.float())
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
    def __init__(self, save_path):
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


# # 代码执行

# ## 参数设置

# In[7]:

if __name__ == '__main__':
    startyear = 1985

    dataset_path = os.path.abspath(os.path.dirname(os.getcwd())) + "/traindata{}_0.75".format(str(startyear))
    inputs_size = [256,256,13]
    NUM_CLASSES = 1
    Batch_size = 30 # 256 6 , 32 20

    lr = 1e-4
    Init_Epoch      = 0
    Interval_Epoch  = 50
    dice_loss = True


    # ## 导入训练数据

    # In[8]:


    with open(os.path.join(dataset_path, "Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    train_dataset   = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, dataset_path)
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=deeplab_dataset_collate)

    with open(os.path.join(dataset_path, "Segmentation/trainval.txt"),"r") as f:
        val_lines = f.readlines()
    val_dataset     = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, dataset_path)
    gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True,
                                    drop_last=True, collate_fn=deeplab_dataset_collate)

    epoch_size      = len(train_lines) // Batch_size
    epoch_size_val  = len(val_lines) // Batch_size


    # ## 初始化模型

    # In[9]:


    model = Unet(num_classes=NUM_CLASSES, in_channels=inputs_size[-1]).train()
    Cuda = True
    if Cuda:
        # net = torch.nn.DataParallel(model,device_ids=[1,2,3])
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
    

    # ## 开始训练

    # In[10]:


    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%Y%m%d_%H.%M.%S')
    time_str   = time_str
    save_path  = dataset_path + "/logs/loss_" + str(time_str)
    loss_history = LossHistory(save_path)
    optimizer = optim.Adam(model.parameters(),lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)
    epoch_size = len(train_lines) // Batch_size
    if epoch_size == 0 or epoch_size_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
    for param in model.vgg.parameters():
        param.requires_grad = False

    for epoch in range(Init_Epoch,Interval_Epoch):
        fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Interval_Epoch,Cuda,save_path)
        lr_scheduler.step()

    if True:
        lr              = 1e-4
        Interval_Epoch  = 50
        Epoch           = 100

        optimizer       = optim.Adam(model.parameters(),lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)

        train_dataset   = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES,  dataset_path)
        val_dataset     = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, dataset_path)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=deeplab_dataset_collate)

        epoch_size      = len(train_lines) // Batch_size
        epoch_size_val  = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for param in model.vgg.parameters():
            param.requires_grad = True

        for epoch in range(Interval_Epoch,Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Interval_Epoch,Cuda,save_path)
            lr_scheduler.step()
    torch.cuda.empty_cache()






