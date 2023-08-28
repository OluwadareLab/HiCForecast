import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from model.arch_1d import *
from loss.loss import *

device = torch.device("cuda")
    
class Model:
    def __init__(self, local_rank=-1, resume_path=None, resume_epoch=0, load_path=None, training=True, rgb=False, lr=1e-3, loss='single_chanel_no_vgg'):
        self.dmvfn = DMVFN(rgb=rgb)
        self.optimG = AdamW(self.dmvfn.parameters(), lr=lr, weight_decay=1e-3)
        self.rgb = rgb
        self.loss = loss
        if rgb == True:
            input_chan = 3
        else:
            input_chan = 1
        self.input_chan = input_chan
        self.lap = LapLoss(channels=input_chan)
        self.vggloss = VGGPerceptualLoss()
        self.MSELoss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.device()

        if training:
            if local_rank != -1:
                self.dmvfn = DDP(self.dmvfn, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            if resume_path is not None:
                assert resume_epoch>=1
                print(local_rank,": loading...... ", '{}'.format(resume_path))
                self.dmvfn.load_state_dict(torch.load('{}'.format(resume_path)), strict=True)
            else:
                if load_path is not None:
                    self.dmvfn.load_state_dict(torch.load(load_path), strict=True)
        else:
            state_dict = torch.load(load_path)
            model_state_dict = self.dmvfn.state_dict()
            for k in model_state_dict.keys():
                model_state_dict[k] = state_dict['module.'+k]
            self.dmvfn.load_state_dict(model_state_dict)

    def train(self, imgs, learning_rate=0):
        self.dmvfn.train()
        #for param_group in self.optimG.param_groups:
        #    param_group['lr'] = learning_rate
        b, n, c, h, w = imgs.shape
        loss_avg = 0
        for i in range(n-2):
            img0, img1, gt = imgs[:, i], imgs[:, i+1], imgs[:, i+2]
            
            merged =  self.dmvfn(torch.cat((img0, img1, gt), 1), scale=[4,4,4,2,2,2,1,1,1])
            loss_G = 0.0

            loss_l1, loss_vgg = 0, 0
            loss_mse = 0
            l1_loss = 0
            for i in range(9):
                if self.loss == 'single_channel_no_vgg' or self.loss == 'single_channel_default_VGG':
                    loss_l1 +=  (self.lap(merged[i], gt)).mean()*(0.8**(8-i))
                if self.loss == 'single_channel_MSE_no_vgg':
                    loss_mse += (self.MSELoss(merged[i], gt)).mean()*(0.8**(8-i)) 
                if self.loss == 'single_channel_L1_no_vgg':
                    loss_l1 += (self.l1_loss(merged[i], gt)).mean()*(0.8**(8-i)) 

            if self.loss == 'single_channel_default_VGG':
                merged_sq = torch.squeeze(merged[-1])
                merged_rgb = torch.stack((merged_sq, merged_sq, merged_sq))
                merged_rgb = torch.permute(merged_rgb, (1,0,2,3))
                gt_sq = torch.squeeze(gt)
                gt_rgb = torch.stack((gt_sq, gt_sq, gt_sq))
                gt_rgb = torch.permute(gt_rgb, (1,0,2,3))
                loss_vgg = (self.vggloss(merged_rgb, gt_rgb)).mean()
                #loss_vgg = (self.vggloss(merged[-1], gt)).mean()
                loss_G = loss_mse + loss_vgg * 0.5

            self.optimG.zero_grad()
            if self.loss == 'single_channel_no_vgg' or self.loss == 'single_channel_L1_no_vgg':
                loss_G =  loss_l1 
            if self.loss == 'single_channel_MSE_no_vgg':
                loss_G = loss_mse
            loss_avg += loss_G
            loss_G.backward()
            self.optimG.step()

        return loss_avg/(n-2)


    def get_lr(self):
        lr = self.optimG.param_groups[-1]['lr']
        return lr

    def eval(self, imgs, name='city', scale_list = [4,4,4,2,2,2,1,1,1]):
        self.dmvfn.eval()
        b, n, c, h, w = imgs.shape 
        preds = []
        if name == 'CityValDataset':
            assert n == 14
            img0, img1 = imgs[:, 2], imgs[:, 3]
            for i in range(5):
                merged= self.dmvfn(torch.cat((img0, img1), 1), scale=scale_list, training=False)
                length = len(merged)
                if length == 0:
                    pred = img0
                else:
                    pred = merged[-1]

                preds.append(pred)
                img0 = img1
                img1 = pred
            assert len(preds) == 5
        elif name == 'KittiValDataset' or name == 'DavisValDataset':
            assert n == 9
            img0, img1 = imgs[:, 2], imgs[:, 3]
            for i in range(5):
                merged = self.dmvfn(torch.cat((img0, img1), 1), scale=scale_list, training=False)
                length = len(merged)
                if length == 0:
                    pred = img0
                else:
                    pred = merged[-1]
                preds.append(pred)
                img0 = img1
                img1 = pred
            assert len(preds) == 5
        elif name == 'VimeoValDataset':
            assert n == 3
            merged = self.dmvfn(torch.cat((imgs[:, 0], imgs[:, 1]), 1), scale=scale_list, training=False)
            length = len(merged)
            if length == 0:
                pred = imgs[:, 0]
            else:
                pred = merged[-1]
            preds.append(pred)
            assert len(preds) == 1
        elif name == 'single_test': # 1, C, H, W
            merged = self.dmvfn(imgs[0], scale=scale_list, training=False) # 1, 3, H, W
            length = len(merged)
            if length == 0:
                pred = imgs[:, 0]
            else:
                pred = merged[-1]
            return pred
        elif name == 'hic':
            img0, img1 = imgs[:, 0], imgs[:, 1]
            for i in range(3):
                merged = self.dmvfn(torch.cat((img0, img1), 1), scale=scale_list, training=False)
                length = len(merged)
                if length == 0:
                    pred = img0
                else:
                    pred = merged[-1]
                preds.append(pred)
                img0 = img1
                img1 = pred
        return torch.stack(preds, 1)

    def device(self):
        self.dmvfn.to(device)
        self.lap.to(device)
        self.vggloss.to(device)
        self.MSELoss.to(device)
        self.l1_loss.to(device)

    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            torch.save(self.dmvfn.state_dict(),'{}/dmvfn_{}.pkl'.format(path, str(epoch)))
