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
            dat_test = dat_test / max_HiC # put this line iside the if statement after adding batch_max
        #dat_test = dat_test / 225.
        #print("dat_test.shape: ", dat_test.shape)
        #print("dat_test[:10,1:3].shape: ", dat_test[:10, 1:3].shape)

        #test_loader = torch.utils.data.DataLoader(dat_test[:,1:3], batch_size=1, shuffle=False)
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

            if batch_max == True:
                batch_max_val= torch.max(X)
                if batch_max_val > 0:
                    X = X / batch_max_val
            X = X.to(device, dtype=torch.float32, non_blocking=True)

            pred = model.eval(X, 'hic') # BNCHW
            #print("pred.shape: ", pred.shape)
            #pred = torch.cat(pred)
            #print("pred.shape after concat: ", pred.shape)
            if rgb == True:
                pred = pred[:,:,0,:,:]
            #print("pred.shape: ", pred.shape)
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
        print("predictions.shape: ", predictions.shape)
        #np.save(output_dir, predictions)
        return predictions

def my_assemble(dat_predict, output_path, file_index,  num_bins, sub_mat_n, num_predictions=3, hic4d=False):
    #if not os.path.exists(file_predict):
    #    os.makedirs(file_predict)

    dim = sub_mat_n
    '''
    if hic4d == False:
        dat_predict = np.load(file_predict + "pred_chr19.npy")
    if hic4d == True:
        dat_predict = np.load(file_predict + "chr19_predicted.npy")
    '''
    dat_index = np.load(file_index)
    #print("dat_index.shape: ", dat_index.shape)
    
    predictions = []
    for i in range(-num_predictions,0):
        tid = "t"+str(i+7)
        mat_chr = np.zeros((num_bins, num_bins))
        mat_n = np.zeros((num_bins, num_bins))
        for j in range(dat_predict.shape[0]):
            i1, i2 = dat_index[j, i]
            #print("j: ", j) 
            mat_chr[i1:(i1+sub_mat_n), i2:(i2+sub_mat_n)] += dat_predict[j, i]
            mat_n[i1:(i1+sub_mat_n), i2:(i2+sub_mat_n)] += 1
     
        mat_chr2 = np.divide(mat_chr, mat_n, out=np.zeros_like(mat_chr), where=mat_n!=0)
        predictions.append(mat_chr2)
    
    print(np.array(predictions).shape)
    np.save(output_path +".npy", np.array(predictions))
    return predictions

if __name__ == "__main__":    
    max_HiC = 300
    batch_max = False
    sub_mat_n = 64
    #chr_num = 2
    #dataset_num = 8
    cut_off = True
    model_path = "./../final_model/dmvfn_99.pkl"
    model = Model(load_path=model_path, training=False, rgb=False)
    for i in [2,3,4]:
        dataset_num = i
        print("dataset_num: ", dataset_num)
        for chr_num in [2,6]:
            print("chr_num: ", chr_num)
            #model_path = "./../models/hic_train_log/20240414-233329/dmvfn_149.pkl" 
            #data_path = "/scratch/dpinchuk_scratch/HiCForecast/dmvfn/data/data_64/val/data_val_chr19_64.npy"
            data_path = "/scratch/dpinchuk_scratch/HiCForecast/dmvfn/data/dataset_{}/data_64/test/data_test_chr{}_64.npy".format(dataset_num, chr_num)
            if batch_max == True and cut_off == False:
                output_path = "./../final_prediction/HiCForecast/batch_max_trained/dataset_{}/HiCForecast_d{}_pred_chr{}_final".format(dataset_num,dataset_num, chr_num)
            elif batch_max == False and cut_off == True:
                output_path = "./../final_prediction/HiCForecast/dataset_{}/HiCForecast_d{}_pred_chr{}_final".format(dataset_num,dataset_num, chr_num)
            else:
                print("File structue not created for these inputs.")
                quit()
            file_index = "./../data/dataset_{}/data_{}/test/data_test_index_chr{}_{}.npy".format(dataset_num, sub_mat_n, chr_num, sub_mat_n)
            gt_path =  "./../data/dataset_{}/data_64/data_gt_chr{}_64.npy".format(dataset_num, chr_num)
            gt_mx = np.load(gt_path)
            num_bins = gt_mx.shape[1]

            dat_predict = predict(model, data_path, cut_off, max_HiC, batch_max=batch_max)
            my_assemble(dat_predict, output_path, file_index, num_bins, sub_mat_n) #assembles predicted outputs into one final matrix






