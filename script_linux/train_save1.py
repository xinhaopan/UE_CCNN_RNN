# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 23:19:47 2021

@author: Xhpan
"""
import os
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import datetime
import time
from tqdm import tqdm
from unet1 import Unet,CE_Loss, Dice_loss, LossHistory,f_score
import numpy as np
import torch.nn.functional as F
import random
from osgeo import gdal
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# 训练一个epoch

def getTiffFileName(filepath, suffix):
    L1 = []
    L2 = []
    for root, dirs, files in os.walk(filepath):  # 遍历该文件夹
        for file in files:  # 遍历刚获得的文件名files
            (filename, extension) = os.path.splitext(file)  # 将文件名拆分为文件名与后缀
            if (extension == suffix):  # 判断该后缀是否为.c文件
                L1.append(filepath + "/" + file)
                L2.append(filename)
    return L1, L2


class DeeplabDataset(Dataset):
    def __init__(self, train_lines, image_size, num_classes,  dataset_path):
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


        return jpg, name

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
    names = []
    for img, name in batch:
        images.append(img)
        names.append(name)
    images = np.array(images)
    return images, names



def fit_one_epoch(net, epoch, epoch_size, gen, Epoch, cuda,dataset_path):
    net = net.eval()
    with torch.no_grad():
        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_size:
                    break
                imgs, names = batch
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                if cuda:
                    imgs = imgs.cuda()
                # optimizer.zero_grad()
                outputs = net(imgs)
                save_numpy = saveNumpy(outputs, num_classes=NUM_CLASSES)
                if not os.path.exists(dataset_path + '/np_save/'):
                    os.makedirs(dataset_path + '/np_save/')

                np.save(dataset_path + '/np_save/' + names[0], save_numpy)

                pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    
def saveNumpy(inputs, num_classes=21):
    n, c, h, w = inputs.size()
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

    save_numpy = F.softmax(temp_inputs, dim=-1).cpu()
    save_numpy = save_numpy.transpose(1, 0)
    save_numpy = np.resize(save_numpy, (num_classes, h, w))
    return save_numpy
            



os.environ['CUDA_VISIBLE_DEVICES']='0'
if __name__ == "__main__":
    inputs_size = [256, 256, 17]
    years = [2015]

    for startyear in years:
        log_path = "../traindata1_{}_0.75".format(str(startyear))
        
        path = log_path + "/logs"
        
        loss_names = os.listdir(path)
        loss_names.sort()
        loss_name = loss_names[-1]
        
        pthpath = path + "/" + loss_name + "/pth"
        pthnames = os.listdir(pthpath)
        pthname = [x for x in pthnames if 'Epoch100-Total' in x][0]
        model_path = pthpath + "/" + pthname
        
        NUM_CLASSES = 2
        dice_loss = False
        Cuda = True
        
        lr = 1e-4
        Init_Epoch = 0
        Interval_Epoch = 50
        Batch_size = 1
        
        # ## 导入训练数据
        
        # In[6]:
        
        dataset_path = "../simdata1_{}_0".format(str(startyear))
        _, train_lines = getTiffFileName(dataset_path + "/" + "images", ".tif")
        
        random.shuffle(train_lines)
        train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, dataset_path)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        
        # ## 初始化模型
        
        # In[7]:
        
        
        model = Unet(num_classes=NUM_CLASSES, in_channels=inputs_size[-1]).train()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        if Cuda:
            net = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            net = net.cuda()
        
        # ## 开始预测
        
        # In[ ]:
        
        
        optimizer = optim.Adam(model.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        
        epoch_size = len(train_lines) // Batch_size
        
        if epoch_size == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        for param in model.vgg.parameters():
            param.requires_grad = False
        print(startyear)
        fit_one_epoch(model, 1, epoch_size, gen, Interval_Epoch, Cuda,dataset_path)
            