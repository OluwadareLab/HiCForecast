import os
import cv2
import sys
import math
import time
import torch
import lpips
import random
import logging
import argparse
import importlib
import traceback
import numpy as np
import torch.distributed as dist
from pytorch_msssim import ssim, ms_ssim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from numpy.lib import format as npy_format

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from hicrep import *
from predict import *
from assemble import *
from utils.util import *
from model.model_1d import Model

model_path = ""


def evaluate(model, data_val_path, name, epoch, step):
    with torch.no_grad():
        #lpips_score_mine, psnr_score_mine, msssim_score_mine, ssim_score_mine = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
        hicrep = np.zeros(3)
        time_stamp = time.time()
        #num = val_data.__len__()
        val_save_path = "./../data/data_{}/train_val/{}/epoch_{}/".format(patch_size,current_time, epoch)  
        load_path = save_model_path_cache + "/dmvfn_{}.pkl".format( epoch) 
        if not os.path.exists(val_save_path):
            os.makedirs(val_save_path)
        print("Calling predict.py")
        '''
        os.system("python3 ./predict.py \
        --data_path {} \
        --load_path {} \
        --output_dir {} \
        --single_channel \
        --max_HiC {}".format(data_val_path, load_path, val_save_path + "pred_chr19.npy", max_HiC))
        '''
        predict(model, data_val_path, val_save_path + "pred_chr19.npy", cut_off) 
        print("Calling assemble.py")
        get_predictions(val_save_path, 1534, patch_size) #assemble into one big matrix
        val_hicrep = get_hicrep(val_save_path + "pred_chr19_final.npy", patch_size, 400000)
        val_hicrep_40k = get_hicrep(val_save_path + "pred_chr19_final.npy", patch_size, 40000)
        logger.info("Validation scores: {}".format(val_hicrep))
        for i in range(3):
            writer_val.add_scalar(name+' hicrep_%d'%(i),  val_hicrep[i], epoch)
            writer_val.add_scalar(name+' hicrep_40k_%d'%(i),  val_hicrep_40k[i], epoch)
        return val_hicrep

def predict(model, data_path, output_dir, cut_off):
    with torch.no_grad():
        dat_test = np.load(data_path).astype(np.float32)
        if cut_off == True:
            dat_test[dat_test > max_HiC] = max_HiC
            print("Performed cut off for prediction")
        dat_test = dat_test / max_HiC
        #dat_test = dat_test / 225.
        #print("dat_test.shape: ", dat_test.shape)
        #print("dat_test[:10,1:3].shape: ", dat_test[:10, 1:3].shape)

        test_loader = torch.utils.data.DataLoader(dat_test[:,1:3], batch_size=1, shuffle=False)
        print("dat_test.shape: ", dat_test.shape)
        
        predictions = []
        for i, X in enumerate(tqdm(test_loader)):
            # X = X.unsqueeze(0).to(device, non_blocking=True)
            if rgb == True:
                #untested
                X = torch.stack((X, X, X), dim=0)
                X = torch.permute(X, (1, 2, 0, 3, 4))
            else:
            #print("X.shape: ", X.shape)
                X = torch.unsqueeze(X, 2)

            X = X.to(device, dtype=torch.float32, non_blocking=True)

            pred = model.eval(X, 'hic') # BNCHW
            #print("pred.shape: ", pred.shape)
            #pred = torch.cat(pred)
            #print("pred.shape after concat: ", pred.shape)
            if rgb == True:
                pred = pred[:,:,0,:,:]
            #print("pred.shape: ", pred.shape)
            pred = torch.squeeze(pred, dim=2)
            pred = np.array(pred.cpu() * max_HiC)
            predictions.append(pred)

        predictions = np.concatenate(predictions, axis=0)
        print("predictions.shape: ", predictions.shape)
        np.save(output_dir, predictions)
