import numpy as np
import torch

from scipy.stats import pearsonr, spearmanr

from corr import*
from GenomeDISCO import*
from ssim import*


pred = np.load("/home/dmitryp/dpinchuk/dmvfn/CVPR2023-DMVFN/data/data_64/predictions/single_channel_no_vgg_64/batch_256/epoch_149/pred_chr19.npy")
#ground_truth = np.load("/home/ubuntu/dpinchuk/dmvfn/CVPR2023-DMVFN/data/data_96/data_gt_chr19_96.npy")
ground_truth_batches = np.load("./../data/data_64/val/data_val_chr19_64.npy")
pearson_save_path = "./../results/data_96/single_channel_no_vgg_96/batch_256/epoch_149/pearson_chr19.npy"
disco_save_path = "./../results/data_96/single_channel_no_vgg_96/batch_256/epoch_149/disco_chr19_shift_35.npy"


'''
print("pred.shape: ", pred.shape)
print("ground_truth.shape: ", ground_truth.shape)
pearson_list = []
pearson_mean_list = []
for i in range(3):
    r = diagcorr(pred[i], ground_truth[3+i])
    r = np.expand_dims(r, axis=0)
    pearson_list.append(r)
    pearson_mean_list.append(np.mean(r))

print("pearson_list: ", pearson_list)
pearson_list = np.concatenate(pearson_list, axis=0)
print("pearson_list.shape: ", pearson_list.shape)
print("pearson_mean_list: ", pearson_mean_list)
'''
'''
disco_list = []
for i in range(3):
    disco = compute_reproducibility(ground_truth[3 + i], pred[i], True)
    disco_list.append(disco)
print("disco_list: ", disco_list)
disco_list = np.array(disco_list)
print("disco_list.shape: ", disco_list.shape)
'''
print("pred.shape: ", pred.shape)
print("pred.shape[0]: ", pred.shape[0])
disco_mean_list = [[], [], []]
for i in range(3):
    disco_mean_list[i] = 0
    for j in range(len(pred[0])):
        disco = compute_reproducibility(pred[i][j], ground_truth_batches[i][j], True)
        disco_mean_list[i] += disco
    disco_mean_list[i] = disco_mean_list[i] / pred.shape[0]

print("disco_mean_list: ", disco_mean_list)
        
        

#np.save(pearson_save_path, pearson_list)
#print("Saved pearson_list.")
#np.save(disco_save_path, disco_list)
#print("Saved disco_list.")

#ssim1 = ssim(torch.from_numpy(pred[0]).unsqueeze(0).unsqueeze(0), torch.from_numpy(ground_truth[3]).unsqueeze(0).unsqueeze(0), window_size=5)
#print("ssim1: ", ssim1)
#print("r1: ", r1)
#print("r2: ", r2)
#print("r3: ", r3)
