import numpy as np
import torch

from scipy.stats import pearsonr, spearmanr

from corr import*
from GenomeDISCO import*
from ssim import*


pred = np.load("/home/ubuntu/dpinchuk/dmvfn/CVPR2023-DMVFN/data/data_64/predictions/single_channel_no_vgg_64/epoch_20/data_pred_chr6_final.npy")
ground_truth = np.load("/home/ubuntu/dpinchuk/dmvfn/CVPR2023-DMVFN/data/data_64/data_gt_chr6_64.npy")
pearson_save_path = "./../results/data_64/single_channel_no_vgg_64/epoch_20/pearson_chr6.npy"
disco_save_path = "./../results/data_64/single_channel_no_vgg_64/epoch_20/disco_chr6_shift_35.npy"
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

disco_list = []
for i in range(3):
    disco = compute_reproducibility(ground_truth[3 + i], pred[i], True)
    disco_list.append(disco)
print("disco_list: ", disco_list)
disco_list = np.array(disco_list)
print("disco_list.shape: ", disco_list.shape)


np.save(pearson_save_path, pearson_list)
print("Saved pearson_list.")
np.save(disco_save_path, disco_list)
print("Saved disco_list.")

#ssim1 = ssim(torch.from_numpy(pred[0]).unsqueeze(0).unsqueeze(0), torch.from_numpy(ground_truth[3]).unsqueeze(0).unsqueeze(0), window_size=5)
#print("ssim1: ", ssim1)
#print("r1: ", r1)
#print("r2: ", r2)
#print("r3: ", r3)
