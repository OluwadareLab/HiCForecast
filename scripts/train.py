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
from model.model import Model
from disco_eval import *
from pearson_eval import *
from ssim_eval import * 
from psnr_eval import *


def base_build_dataset(name):
    return getattr(importlib.import_module('dataset.dataset', package=None), name)()


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--max_HiC', type=int)
parser.add_argument('--num_gpu', default=1, type=int) # or 8
parser.add_argument('--device_id', type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--batch_size', default=8, type=int, help='minibatch size')
parser.add_argument('--block_num', default=9, type=int)
parser.add_argument('--learning_rate',required=False, type=float)
parser.add_argument('--lr_scale', required=True, type=float)
parser.add_argument('--lr_max', required=False, default=1e-4, type=float)
parser.add_argument('--lr_min', required=False, default=1e-5, type=float)
parser.add_argument('--local_rank', default=0, type=int, help='local rank')
parser.add_argument('--resume_path', default=None, type=str, help='continue to train, model path')
parser.add_argument('--resume_epoch', default=0, type=int, help='continue to train, epoch')
parser.add_argument('--code_test', action='store_true')
parser.add_argument('--no_code_test', dest='code_test', action='store_false')
parser.add_argument('--rgb', action='store_true')
parser.add_argument('--no_rgb', dest='rgb', action='store_false')
parser.add_argument('--data_train_path', required=True, type=str)
parser.add_argument('--data_val_path', required=True, type=str)
parser.add_argument('--loss', required=True, type=str, help=['single_channel_no_vgg'])
parser.add_argument('--early_stoppage_epochs', type=int)
parser.add_argument('--early_stoppage_start', type=int)
parser.add_argument('--patch_size', required=True, type=int)
parser.add_argument('--val_gt_path', required=True, type=str)
parser.add_argument('--val_file_index_path', required=True, type=str)
parser.add_argument('--cut_off', action='store_true')
parser.add_argument('--no_cut_off', dest='cut_off', action='store_false')
parser.add_argument('--dynamics', action='store_true')
parser.add_argument('--no_dynamics', dest='dynamics', action='store_false')
parser.add_argument('--max_cut_off', action='store_true')
parser.add_argument('--no_max_cut_off', dest='max_cut_off', action='store_false')
parser.add_argument('--batch_max', action='store_true')
parser.add_argument('--no_batch_max', dest='batch_max', action='store_false')
parser.set_defaults(code_test=False)
args = parser.parse_args()
print("args parsed.")

torch.distributed.init_process_group(backend="nccl", world_size=args.num_gpu)
local_rank = torch.distributed.get_rank()
device_number = args.device_id
torch.cuda.set_device(device_number)
device = torch.device("cuda", device_number)
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

loss_fn_alex = lpips.LPIPS(net='alex').to(device)

max_HiC = args.max_HiC
max_cut_off = args.max_cut_off
rgb = args.rgb
patch_size = args.patch_size
batch_size = args.batch_size
loss = args.loss
val_dataset = "hic"
data_val_path = args.data_val_path
lr = args.learning_rate
es_start = args.early_stoppage_start
cut_off = args.cut_off
block_num = args.block_num
dynamics = args.dynamics
lr_scale = args.lr_scale
lr_max = args.lr_max
lr_min = args.lr_min
batch_max = args.batch_max
if lr_max == None:
    lr_max = 1e-4
if lr_min == None:
    lr_min = 1e-5

current_time = get_timestamp()
code_test = args.code_test
if code_test == True:
    output_folder = 'HiCForecast_train_{}_test'.format(current_time)
    log_path = '{}/log'.format(output_folder)
    save_model_path = output_folder
    save_model_path_cache = '{}/cache'.format(output_folder)
else:
    output_folder = 'HiCForecast_train_{}'.format(current_time)
    log_path = '{}/log'.format(output_folder)
    save_model_path = output_folder
    save_model_path_cache = '{}/cache'.format(output_folder)



if local_rank == 0:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    if not os.path.exists(save_model_path_cache):
        os.makedirs(save_model_path_cache)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    setup_logger('base', log_path, 'train', level=logging.INFO, screen=True, to_file=True)
    writer = SummaryWriter(log_path + '/train')
    writer_val = SummaryWriter(log_path + '/validate')


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
    return (lr_max - lr_min) * mul + lr_min

logger = logging.getLogger('base')

for arg, value in sorted(vars(args).items()):
    logger.info("{} Argument {}: {}".format(get_formatted_timestamp(), arg, value))

if lr == None:
    logger.info("Defualt learning rate schedule")

def train(model, args):
    step = 0
    nr_eval = args.resume_epoch
   

    old_val = []
    for i in range(args.early_stoppage_epochs):
        old_val.append(np.array([0., 0., 0.]))

    data_val_path = args.data_val_path
    train_path = args.data_train_path
    train_list = os.listdir(train_path)
    dataset_length = 0
    for file_name in train_list:
       with open(train_path + file_name, 'rb') as f:
            version = npy_format.read_magic(f)
            shape, _, _ = npy_format._read_array_header(f, version)
            dataset_length = dataset_length + shape[0]

    args.step_per_epoch = dataset_length // args.batch_size

    step = 0 + args.step_per_epoch * args.resume_epoch
    if local_rank == 0:
        print('training...')
    val_hicrep = np.array([-1, -1, -1])
    time_stamp = time.time()
    for epoch in range(args.resume_epoch, args.epoch):
        print("Epoch: ", epoch)
        set_number = 0 
        for chr_file in train_list:
            dataset = np.load(train_path + chr_file)
            if max_cut_off == True:
                max_HiC = np.max(dataset[:,0:3, :, :])
                dataset = dataset / max_HiC
            if args.cut_off == True:
                dataset[dataset > max_HiC] = max_HiC
                dataset = dataset / max_HiC
            sampler = DistributedSampler(dataset)
            train_data = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=sampler)
            sampler.set_epoch(epoch)
            set_number = set_number + 1
            step_per_dataset = train_data.__len__()
            for i, data in enumerate(train_data):
                data_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                if rgb == True:
                    data = np.stack((data, data, data), axis=0)
                    data = np.transpose(data, (1,2,0,3,4))
                else:
                    data = torch.unsqueeze(data, 2)

                if batch_max == True:
                    batch_max_val = torch.max(data[:, 1:3, :, :, :])
                    if batch_max_val > 0:
                        data[:, 1:3] = data[:, 1:3] / batch_max_val
                data_gpu = data.to(device, dtype=torch.float32, non_blocking=True)  #B,3,C,H,W
                
                if lr == None:
                    learning_rate = get_learning_rate(step)
                else:
                    learning_rate = lr

                learning_rate = learning_rate * lr_scale

                loss_avg = model.train(data_gpu,learning_rate)
                
                train_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                if step % 200 == 1 and local_rank == 0:
                    writer.add_scalar('learning_rate', learning_rate, step)
                    writer.add_scalar('loss/loss_l1', loss_avg, step)
                    writer.flush()
                if local_rank == 0:
                    logger.info('{} epoch:{} dataset: {}/{} step: {}/{} time:{:.2f}+{:.2f} loss_avg:{:.4e}'.format( \
                        get_formatted_timestamp(), epoch, set_number % 17, 17, i, step_per_dataset, data_time_interval, train_time_interval, loss_avg))
                step += 1
                if code_test == True and i == 1:
                    break
            if code_test == True and set_number == 2:
                break
            logger.info(f'Training on {chr_file} complete.')
        nr_eval += 1
        if local_rank <= 0:    
            model.save_model(save_model_path_cache, epoch, local_rank)   
            print("Model saved to cache")
            if (epoch == (args.epoch - 1)) or (((epoch + 1) % 5 == 0) and (epoch > 100)):
                model.save_model(save_model_path, epoch, local_rank)   
        if epoch % 1 == 0:
            val_disco_old = old_val.pop(0)
            val_hicrep = evaluate(model, data_val_path, val_dataset, epoch, step, args)
            old_val.append(val_hicrep)
            if epoch >= es_start:
                if val_hicrep[0] - val_hicrep_old[0] < 0.0001 or val_hicrep[1] - val_hicrep_old[1] < 0.0001 or val_hicrep[2] - val_hicrep_old[2]<0.0001:
                    logger.info("Training module complete due to early stoppage")
                    logger.info("es_start: {}".format(es_start))
                    logger.info("epoch: {}".format(epoch))
                    for i in range(3):
                        logger.info("val_hicrep[{}]: {}".format(i, val_hicrep[i]))
                        logger.info("val_hicrep_old[{}]: {}".format(i, val_hicrep_old[i]))
                    quit()
        dist.barrier()
    logger.info('{} Training module completed.'.format(get_formatted_timestamp()))

def evaluate(model, data_val_path, name, epoch, step, args):
    with torch.no_grad():
        time_stamp = time.time()
        val_save_path = "{}/train_val_data_ps_{}_{}/".format(output_folder, patch_size,current_time)  
        load_path = save_model_path_cache + "{}.pkl".format( epoch) 
        if not os.path.exists(val_save_path):
            os.makedirs(val_save_path)
        predict(model, data_val_path, val_save_path + "epoch_{}.npy".format(epoch), cut_off, args) 
        A = np.load(args.val_gt_path)
        num_bins = A.shape[1]
        get_predictions(val_save_path +"epoch_{}".format(epoch) , num_bins, patch_size, args.val_file_index_path) #assemble into one big matrix
        pred_mx = np.load(val_save_path + "epoch_{}_final.npy".format(epoch)) 
        gt_mx = np.load(args.val_gt_path)
        ps = args.patch_size - 2 #evaluation region
        m = np.max(gt_mx)
        disco = compute_disco_avg(pred_mx, gt_mx, True, ps)
        pearson = compute_pearson_avg(pred_mx, gt_mx, ps)
        ssim = compute_ssim_avg(pred_mx, gt_mx, ps, m)
        psnr = compute_psnr_avg(pred_mx, gt_mx, ps, m)

        logger.info("Validation disco scores: {}".format(disco))
        for i in range(3):
            writer_val.add_scalar(name+' disco_%d'%(i),  disco[i], epoch)
            writer_val.add_scalar(name+' pearson_%d'%(i),  pearson[i], epoch)
            writer_val.add_scalar(name+' ssim_%d'%(i),  ssim[i], epoch)
            writer_val.add_scalar(name+' psnr_%d'%(i),  psnr[i], epoch)
        return disco


def predict(model, data_path, output_dir, cut_off, args):
    with torch.no_grad():
        dat_test = np.load(data_path).astype(np.float32)
        if max_cut_off == True:
            max_HiC = np.max(dat_test[:, 0:3, :, :])
        else:
            max_HiC = args.max_HiC
        if cut_off == True:
            dat_test[dat_test > max_HiC] = max_HiC
            dat_test = dat_test / max_HiC

        test_loader = torch.utils.data.DataLoader(dat_test[:,1:3], batch_size=args.batch_size, shuffle=False)
        
        predictions = []
        for i, X in enumerate(tqdm(test_loader)):
            if rgb == True:
                #untested
                X = torch.stack((X, X, X), dim=0)
                X = torch.permute(X, (1, 2, 0, 3, 4))
            else:
                X = torch.unsqueeze(X, 2)
            if args.batch_max == True:
                batch_max_val = torch.max(X)
                if batch_max_val > 0:
                    X = X / batch_max_val

            X = X.to(device, dtype=torch.float32, non_blocking=True)

            pred = model.eval(X, 'hic') # BNCHW
            if rgb == True:
                pred = pred[:,:,0,:,:]
            pred = torch.squeeze(pred, dim=2)
            if args.batch_max == True:
                if batch_max_val > 0:
                    pred = np.array(pred.cpu() * batch_max_val.item())
                else:
                    pred = np.array(pred.cpu())
            else:
                pred = np.array(pred.cpu() * max_HiC)
            predictions.append(pred)

        predictions = np.concatenate(predictions, axis=0)
        np.save(output_dir, predictions)

if __name__ == "__main__":    
    try:
        model = Model(local_rank=device_number, resume_path=args.resume_path, resume_epoch=args.resume_epoch, 
                loss=loss, block_num=block_num, dynamics = dynamics)
        train(model, args)
    except Exception:
        print("Exception caught.")
        logger.info("Traceback:")
        logger.info(traceback.format_exc())
        exit()
        
