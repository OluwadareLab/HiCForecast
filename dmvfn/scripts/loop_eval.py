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

start_epoch = 100
last_epoch = 149
num_predictions = 4
#num_predictions = 3
patch_size = 96
max_HiC = 250
cutoff = True
rgb = False
#model_id = "20230920-213115"
model_id = "20230930-190101"

data_val_path = "./../data/data_{}/val/data_val_chr19_{}.npy".format(patch_size, patch_size)
model_cache_path = "./../models/hic_train_log_cache/"
model_path = model_cache_path + model_id + "/"
log_path = "./../logs/validation/" + model_id
if num_predictions == 3:
    writer_val = SummaryWriter(log_path + "_p" + '/validate')
if num_predictions == 4:
    writer_val = SummaryWriter(log_path + "_4_p" + '/validate')


def evaluate(model, data_val_path, epoch):
    with torch.no_grad():
        #lpips_score_mine, psnr_score_mine, msssim_score_mine, ssim_score_mine = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
        hicrep = np.zeros(3)
        time_stamp = time.time()
        #num = val_data.__len__()
        val_save_path = "./../data/data_{}/posthoc_val/{}/epoch_{}/".format(patch_size, model_id, epoch)  
        load_path = model_path + "/dmvfn_{}.pkl".format( epoch) 
        if not os.path.exists(val_save_path):
            os.makedirs(val_save_path)
        print("Calling predict.py")
        predict(model, data_val_path, val_save_path + "pred_chr19.npy", cutoff) 
        print("Calling assemble.py")
        get_predictions(val_save_path, 1534, patch_size, num_predictions=num_predictions) #assemble into one big matrix
        val_hicrep = get_hicrep(val_save_path + "pred_chr19_final.npy", patch_size, 400000, num_predictions=num_predictions)
        val_hicrep_40k = get_hicrep(val_save_path + "pred_chr19_final.npy", patch_size, 40000, num_predictions=num_predictions)
        val_hicrep_0k = get_hicrep(val_save_path + "pred_chr19_final.npy", patch_size, 0, num_predictions=num_predictions)
        #logger.info("Validation scores: {}".format(val_hicrep))
        for i in range(num_predictions):
            writer_val.add_scalar('hic hicrep_%d'%(i-1),  val_hicrep[i], epoch)
            writer_val.add_scalar('hic hicrep_%d'%(i-1),  val_hicrep[i], epoch)
            writer_val.add_scalar('hic hicrep_40k_%d'%(i-1),  val_hicrep_40k[i], epoch)
            writer_val.add_scalar('hic hicrep_0k_%d'%(i-1),  val_hicrep_0k[i], epoch)
        return val_hicrep

def predict(model, data_path, output_dir, cut_off):
    with torch.no_grad():
        dat_test = np.load(data_path).astype(np.float32)
        if cut_off == True:
            dat_test[dat_test > max_HiC] = max_HiC
            print("Performed cut off for prediction")
        dat_test = dat_test / max_HiC
        #dat_test = dat_test / 225.
        print("dat_test.shape: ", dat_test.shape)
        #print("dat_test[:10,1:3].shape: ", dat_test[:10, 1:3].shape)

        if num_predictions == 3:
            test_loader = torch.utils.data.DataLoader(dat_test[:,1:3], batch_size=1, shuffle=False)
        elif num_predictions == 4:
            test_loader = torch.utils.data.DataLoader(dat_test[:,0:2], batch_size=1, shuffle=False)
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

            pred = model.eval(X, 'hic', num_predictions=num_predictions) # BNCHW
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

dir_list = os.listdir(model_path)
hic_rep = []
for epoch in range(start_epoch, last_epoch + 1):
    load_path = model_path + "dmvfn_{}.pkl".format(epoch)
    print("load_path: ", load_path)
    model = Model(load_path=load_path, training=False, rgb=False)
    hic_rep_current = evaluate(model, data_val_path, epoch)
    hic_rep.append(hic_rep_current)
print("hicrep: ", hic_rep)



