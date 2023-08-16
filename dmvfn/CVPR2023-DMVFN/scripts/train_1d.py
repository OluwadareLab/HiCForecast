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



#This comment was made while in the single_channel branch
#Second comment in single_channel branch

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from utils.util import *
from model.model_1d import Model

def base_build_dataset(name):
    return getattr(importlib.import_module('dataset.dataset', package=None), name)()


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--max_HiC', type=int)
parser.add_argument('--num_gpu', default=1, type=int) # or 8
parser.add_argument('--device_number', type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--batch_size', default=8, type=int, help='minibatch size')
parser.add_argument('--local_rank', default=0, type=int, help='local rank')
parser.add_argument('--train_dataset', required=True, type=str, help='CityTrainDataset, KittiTrainDataset, VimeoTrainDataset')
parser.add_argument('--val_datasets', type=str, nargs='+', default=['hic'], help='[CityValDataset,KittiValDataset,VimeoValDataset,DavisValDataset]')
parser.add_argument('--resume_path', default=None, type=str, help='continue to train, model path')
parser.add_argument('--resume_epoch', default=0, type=int, help='continue to train, epoch')
#parser.add_argument('--code_test_mode', action="store_false", help='code_test_mode=True for testing training module')
parser.add_argument('--code_test', action='store_true')
parser.add_argument('--no_code_test', dest='code_test', action='store_false')
parser.add_argument('--rgb', action='store_true')
parser.add_argument('--no_rgb', dest='rgb', action='store_false')
parser.add_argument('--data_train_path', required=True, type=str)
parser.set_defaults(code_test=False)
args = parser.parse_args()
print("args parsed.")

torch.distributed.init_process_group(backend="nccl", world_size=args.num_gpu)
print("distributed.")
local_rank = torch.distributed.get_rank()
device_number = args.device_number
print("got rank")
torch.cuda.set_device(device_number)
device = torch.device("cuda", device_number)
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

#exp = os.path.abspath('.').split('/')[-1]
loss_fn_alex = lpips.LPIPS(net='alex').to(device)

max_HiC = args.max_HiC
rgb = args.rgb

current_time = get_timestamp()
code_test = args.code_test
if code_test == True:
    print("True")
if code_test == True:
    log_path = './../logs/test/{}_train_log/{}'.format(args.train_dataset, current_time)
    save_model_path = './../models/test/{}_train_log/{}'.format(args.train_dataset, current_time)
    save_model_path_cache = './../models/test/{}_train_log_cache/{}'.format(args.train_dataset, current_time)
else:
    log_path = './../logs/{}_train_log/{}'.format(args.train_dataset, current_time)
    save_model_path = './../models/{}_train_log/{}'.format(args.train_dataset, current_time)
    save_model_path_cache = './../models/{}_train_log_cache/{}'.format(args.train_dataset, current_time)


if local_rank == 0:
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
    return (1e-4 - 1e-5) * mul + 1e-5

logger = logging.getLogger('base')
#logger.info("Argument rgb: {}".format(rgb))
logger.info("Default loss without VGG")
logger.info("No cutoff")
#logger.info("Discounted L1 loss with VGG")

for arg, value in sorted(vars(args).items()):
    logger.info("{} Argument {}: {}".format(get_formatted_timestamp(), arg, value))

def train(model, args):
    step = 0
    nr_eval = args.resume_epoch
   
    train_path = args.data_train_path
    train_list = os.listdir(train_path)
    dataset_length = 0
    for file_name in train_list:
       with open(train_path + file_name, 'rb') as f:
            version = npy_format.read_magic(f)
            shape, _, _ = npy_format._read_array_header(f, version)
            dataset_length = dataset_length + shape[0]
    print("dataset_length: ", dataset_length)

    args.step_per_epoch = dataset_length // args.batch_size

    step = 0 + args.step_per_epoch * args.resume_epoch
    if local_rank == 0:
        print('training...')
    time_stamp = time.time()
    for epoch in range(args.resume_epoch, args.epoch):
        print("Epoch: ", epoch)
        set_number = 0 
        for chr_file in train_list:
            dataset = np.load(train_path + chr_file)
            #dataset[dataset > max_HiC] = max_HiC
            dataset = dataset / max_HiC
            #print("dataset[200][1][40]: ", dataset[200][1][40])
            sampler = DistributedSampler(dataset)
            train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=sampler)
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

                #data_gpu = torch.from_numpy(data)
                data_gpu = data.to(device, dtype=torch.float32, non_blocking=True)  #B,3,C,H,W
                
                learning_rate = get_learning_rate(step)

                loss_avg = model.train(data_gpu, learning_rate)
                
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
        '''
        if nr_eval % 1 == 0:
            for dataset_name in args.val_datasets:
                val_dataset = base_build_dataset(dataset_name)
                val_data = DataLoader(val_dataset, batch_size=1, pin_memory=True, num_workers=1)
                evaluate(model, val_data, dataset_name, nr_eval, step)
        '''
        if local_rank <= 0:    
            if (epoch == (args.epoch - 1)) or ((epoch + 1) % 50 == 0):
                model.save_model(save_model_path, epoch, local_rank)   
            else:
                model.save_model(save_model_path_cache, epoch, local_rank)   

        dist.barrier()
    logger.info('{} Training module completed.'.format(get_formatted_timestamp()))

def evaluate(model, val_data, name, nr_eval, step):
    if name == "CityValDataset" or name == "KittiValDataset" or name == "DavisValDataset":
        with torch.no_grad():
            lpips_score_mine, psnr_score_mine, msssim_score_mine, ssim_score_mine = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
            time_stamp = time.time()
            num = val_data.__len__()
            for i, data in enumerate(val_data):
                data_gpu, _ = data
                data_gpu = data_gpu.to(device, non_blocking=True) / 255.
                preds = model.eval(data_gpu, name)

                b,n,c,h,w = preds.shape
                assert b==1 and n==5

                gt, pred = data_gpu[0], preds[0]
                for j in range(5):
                    psnr = -10 * math.log10(torch.mean((gt[j+4] - pred[j]) * (gt[j+4] - pred[j])).cpu().data)
                    ssim_val = ssim( gt[j+4:j+5], pred[j:j+1], data_range=1.0, size_average=False) # return (N,)
                    ms_ssim_val = ms_ssim( gt[j+4:j+5], pred[j:j+1], data_range=1.0, size_average=False ) #(N,)
                    x, y = ((gt[j+4:j+5]-0.5)*2.0).clone(), ((pred[j:j+1]-0.5)*2.0).clone()
                    lpips_val = loss_fn_alex(x, y)

                    lpips_score_mine[j] += lpips_val
                    ssim_score_mine[j] += ssim_val
                    msssim_score_mine[j] += ms_ssim_val
                    psnr_score_mine[j] += psnr
                    
                    gt_1 = (gt[j+4:j+5].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                    pred_1 = (pred[j:j+1].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                    if i == 50 and local_rank == 0:
                            imgs = np.concatenate((gt_1[0], pred_1[0]), 1)[:, :, ::-1]
                            writer_val.add_image(name+str(j) + '/img', imgs.copy(), step, dataformats='HWC')
            eval_time_interval = time.time() - time_stamp
            if local_rank != 0:
                return
            psnr_score_mine, ssim_score_mine, msssim_score_mine, lpips_score_mine = psnr_score_mine/num, ssim_score_mine/num, msssim_score_mine/num, lpips_score_mine/num
            for i in range(5):
                logger.info('%d             '%(nr_eval)+name+'  psnr_%d     '%(i)+'%.4f'%(sum(psnr_score_mine[:(i+1)])/(i+1))+'  ssim_%d     '%(i)+'%.4f'%(sum(ssim_score_mine[:(i+1)])/(i+1))+'  ms_ssim_%d     '%(i)+
                '%.4f'%(sum(msssim_score_mine[:(i+1)])/(i+1))+'  lpips_%d     '%(i)+'%.4f'%(sum(lpips_score_mine[:(i+1)])/(i+1)))

                writer_val.add_scalar(name+' psnr_%d'%(i),  psnr_score_mine[i], step)
                writer_val.add_scalar(name+' ssim_%d'%(i),  ssim_score_mine[i], step)
                writer_val.add_scalar(name+' ms_ssim_%d'%(i),  msssim_score_mine[i], step)
                writer_val.add_scalar(name+' lpips_%d'%(i),  lpips_score_mine[i], step)
    elif name=="VimeoValDataset":
        with torch.no_grad():
            lpips_score_mine, ssim_score_mine, msssim_score_mine, psnr_score_mine   = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
            time_stamp = time.time()
            num = val_data.__len__()
            for i, data in enumerate(val_data):
                data_gpu, _ = data
                data_gpu = data_gpu.to(device, non_blocking=True) / 255.
                preds = model.eval(data_gpu, name)

                b,n,c,h,w = preds.shape
                assert b==1 and n==1

                gt, pred = data_gpu[0], preds[0]
                psnr = -10 * math.log10(torch.mean((gt[2] - pred[0]) * (gt[2] - pred[0])).cpu().data)
                ssim_val = ssim( gt[2:3], pred[0:1], data_range=1.0, size_average=False) # return (N,)
                ms_ssim_val = ms_ssim( gt[2:3], pred[0:1], data_range=1.0, size_average=False ) #(N,)
                x, y = ((gt[2:3]-0.5)*2.0).clone(), ((pred[0:1]-0.5)*2.0).clone()
                lpips_val = loss_fn_alex(x, y)

                lpips_score_mine[0] += lpips_val
                ssim_score_mine[0] += ssim_val
                msssim_score_mine[0] += ms_ssim_val
                psnr_score_mine[0] += psnr
                
                gt_1 = (gt[2:3].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                pred_1 = (pred[0:1].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                if i == 50 and local_rank == 0:
                        imgs = np.concatenate((gt_1[0], pred_1[0]), 1)[:, :, ::-1]
                        writer_val.add_image(name+str(0) + '/img', imgs.copy(), step, dataformats='HWC')
            eval_time_interval = time.time() - time_stamp

            if local_rank != 0:
                return
            psnr_score_mine, ssim_score_mine, msssim_score_mine, lpips_score_mine = psnr_score_mine/num, ssim_score_mine/num, msssim_score_mine/num, lpips_score_mine/num
            for i in range(1):
                logger.info('%d             '%(nr_eval)+name+'  psnr_%d     '%(i)+'%.4f'%(sum(psnr_score_mine[:(i+1)])/(i+1))+'  ssim_%d     '%(i)+'%.4f'%(sum(ssim_score_mine[:(i+1)])/(i+1))+'  ms_ssim_%d     '%(i)+
                '%.4f'%(sum(msssim_score_mine[:(i+1)])/(i+1))+'  lpips_%d     '%(i)+'%.4f'%(sum(lpips_score_mine[:(i+1)])/(i+1)))

                writer_val.add_scalar(name+' psnr_%d'%(i),  psnr_score_mine[i], step)
                writer_val.add_scalar(name+' ssim_%d'%(i),  ssim_score_mine[i], step)
                writer_val.add_scalar(name+' ms_ssim_%d'%(i),  msssim_score_mine[i], step)
                writer_val.add_scalar(name+' lpips_%d'%(i),  lpips_score_mine[i], step)
        
if __name__ == "__main__":    
    try:
        model = Model(local_rank=device_number, resume_path=args.resume_path, resume_epoch=args.resume_epoch)
        #model = nn.parallel.DistributedDataParallel
        train(model, args)
    except Exception:
        print("Exception caught.")
        logger.info("Traceback:")
        logger.info(traceback.format_exc())
        logger.info("Sys.exc_info()[2]:")
        logger.info(sys.exc_info()[2])
        exit()
