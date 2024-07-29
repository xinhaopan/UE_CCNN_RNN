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
from unet import Unet,CE_Loss, Dice_loss, LossHistory,DeeplabDataset, deeplab_dataset_collate,f_score

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



os.environ['CUDA_VISIBLE_DEVICES']='4'
if __name__ == "__main__":
    years = [2000,2001,2002]
    for startyear in years:
    
        dataset_path = "../traindata{}_0.75".format(str(startyear))
        inputs_size = [256,256,13]
        NUM_CLASSES = 2
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
        loss_history = LossHistory(save_path,time_str)
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