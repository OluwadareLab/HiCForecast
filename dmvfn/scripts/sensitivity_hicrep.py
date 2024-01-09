import os
import cv2
import csv
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

#start_epoch = 100
#last_epoch = 149
num_predictions = 3
#num_predictions = 3
patch_size = 50
max_HiC = 100
cutoff = True
rgb = False
#model_id = "20230920-213115"
model_id = "20230930-190101"
hic4d = True

data_val_path = "./../data/data_{}/val/data_val_chr19_{}.npy".format(patch_size, patch_size)
model_log_path = "./../models/hic_train_log/"
model_path = model_log_path + model_id + "/"

#dir_list = os.listdir(model_path)
#hic_rep = []
#csv_file_path = "./../results/190101/hicrep_190101.csv"
csv_file_path = "./../results/hic4d/hicrep_hic4d_45_46.csv"
epochs = [99, 149]
hicrep_range = []
if hic4d == True:
    for u in range(46, 44, -1):
        print("u: ", u)
        val_save_path = "./../data/data_50/predictions/norm_100/"
        get_predictions(val_save_path, 1534, patch_size, num_predictions=num_predictions, hic4d=hic4d) #assemble into one big matrix
        hicrep =  get_hicrep(val_save_path + "pred_chr19_final.npy", patch_size, 0, ubr=u*40000, num_predictions=num_predictions).tolist()
        hicrep_range.append(hicrep)
    hicrep_list = np.array(hicrep_range)
else:
    for u in range(47, 7, -3):
        hicrep_range = []
        for epoch in epochs: 
            load_path = model_path + "dmvfn_{}.pkl".format(epoch)
            print("load_path: ", load_path)
            val_save_path = "./../data/data_96/190101_predictions/{}/".format(epoch)  
            get_predictions(val_save_path, 1534, patch_size, num_predictions=num_predictions) #assemble into one big matrix
            hicrep =  get_hicrep(val_save_path + "pred_chr19_final.npy", patch_size, 0, ubr=u*40000, num_predictions=num_predictions).tolist()
            hicrep_range.append(hicrep)
        hicrep_list.append(hicrep_range)

    hicrep_list = np.array(hicrep_list)
print(hicrep_list)

if hic4d == True:
    with open (csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ubr", "t4","t5","t6"])
        
        for i in range(hicrep_list.shape[0]):
            row = hicrep_list[i]
            row = row.tolist()
            row.insert(0, 46 - i)
            writer.writerow(row)
else: 
    with open (csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", 99, "","", 149, "", "",])
        writer.writerow(["ubr", "t4","t5","t6","t4", "t5", "t6"])
        
        for i in range(hicrep_list.shape[0]):
            row = np.hstack(hicrep_list[i])
            row = row.tolist()
            row.insert(0, 47 - 3*i)
            writer.writerow(row)


