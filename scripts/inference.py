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
import argparse
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from numpy.lib import format as npy_format

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from assemble import *
from predict import *
from utils.util import *
from model.model_1d import Model


    


def predict(model, data_path, cut_off, max_HiC, rgb=False, batch_max=False):
    with torch.no_grad():
        dat_test = np.load(data_path).astype(np.float32)
        if cut_off == True:
            dat_test[dat_test > max_HiC] = max_HiC
            print("Performed cut off for prediction")
            dat_test = dat_test / max_HiC 
        test_loader = torch.utils.data.DataLoader(dat_test[:,1:3], batch_size=1, shuffle=False)
        
        predictions = []
        for i, X in enumerate(tqdm(test_loader)):
            if rgb == True:
                #untested
                X = torch.stack((X, X, X), dim=0)
                X = torch.permute(X, (1, 2, 0, 3, 4))
            else:
                X = torch.unsqueeze(X, 2)

            if batch_max == True:
                batch_max_val= torch.max(X)
                if batch_max_val > 0:
                    X = X / batch_max_val
            X = X.to(device, dtype=torch.float32, non_blocking=True)

            pred = model.eval(X, 'hic') # BNCHW
            if rgb == True:
                pred = pred[:,:,0,:,:]
            pred = torch.squeeze(pred, dim=2)
            if batch_max == True:
                if batch_max_val > 0:
                    pred = np.array(pred.cpu() * batch_max_val)
                else:
                    pred = np.array(pred.cpu())
            else:
                pred = np.array(pred.cpu() * max_HiC)
            predictions.append(pred)

        predictions = np.concatenate(predictions, axis=0)
        #print("predictions.shape: ", predictions.shape)
        return predictions

def my_assemble(dat_predict, output_path, file_index,  num_bins, sub_mat_n, num_predictions=3, hic4d=False):

    dim = sub_mat_n
    dat_index = np.load(file_index)
    
    predictions = []
    for i in range(-num_predictions,0):
        tid = "t"+str(i+7)
        mat_chr = np.zeros((num_bins, num_bins))
        mat_n = np.zeros((num_bins, num_bins))
        for j in range(dat_predict.shape[0]):
            i1, i2 = dat_index[j, i]
            mat_chr[i1:(i1+sub_mat_n), i2:(i2+sub_mat_n)] += dat_predict[j, i]
            mat_n[i1:(i1+sub_mat_n), i2:(i2+sub_mat_n)] += 1
     
        mat_chr2 = np.divide(mat_chr, mat_n, out=np.zeros_like(mat_chr), where=mat_n!=0)
        predictions.append(mat_chr2)
    
    #print(np.array(predictions).shape)
    np.save(output_path +".npy", np.array(predictions))
    return predictions

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_HiC', type=int, default=300)
    parser.add_argument('--batch_max', action='store_true')
    parser.add_argument('--no_batch_max', dest='batch_max', action='store_false')
    parser.add_argument('--patch_size', required=True, type=int, default=64)
    parser.add_argument('--cut_off', action='store_true')
    parser.add_argument('--no_cut_off', dest='cut_off', action='store_false')
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--file_index_path', type=str)
    parser.add_argument('--gt_path', type=str)

    args = parser.parse_args()
    
    #arguments:
    max_HiC = args.max_HiC
    batch_max = args.batch_max
    sub_mat_n = args.patch_size
    cut_off = args.cut_off
    model_path = args.model_path 
    model = Model(load_path=model_path, training=False, rgb=False)
    data_path = args.data_path 
    output_path = args.output_path
    file_index = args.file_index_path
    gt_path =  args.gt_path

    #get the dimensions of final ouput
    gt_mx = np.load(gt_path)
    num_bins = gt_mx.shape[1]

    #predict patches and assemble into one final prediction
    dat_predict = predict(model, data_path, cut_off, max_HiC, batch_max=batch_max)
    my_assemble(dat_predict, output_path, file_index, num_bins, sub_mat_n) #assembles predicted outputs into one final matrix






