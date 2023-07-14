import numpy as np
import torch

from scipy.stats import pearsonr, spearmanr

from corr import*
from GenomeDISCO import*
from ssim import*


pred = np.load("/home/ubuntu/dpinchuk/dmvfn/CVPR2023-DMVFN/data/data_64/predictions/single_channel_no_vgg_64/epoch_20/data_pred_chr6_final.npy")
ground_truth = np.load("/home/ubuntu/dpinchuk/dmvfn/CVPR2023-DMVFN/data/data_64/data_gt_chr6_64.npy")
print("pred.shape: ", pred.shape)
print("ground_truth.shape: ", ground_truth.shape)
pearson_list = []
for i in range(3):
    r = diagcorr(pred[i], ground_truth[3+i])
    pearson_list.append(r)


print("pearson_list: ", pearson_list)
#disco1 = compute_reproducibility(pred[0], ground_truth[3], True)
#print("disco1: ", disco1)

#ssim1 = ssim(torch.from_numpy(pred[0]).unsqueeze(0).unsqueeze(0), torch.from_numpy(ground_truth[3]).unsqueeze(0).unsqueeze(0), window_size=5)
#print("ssim1: ", ssim1)
#print("r1: ", r1)
#print("r2: ", r2)
#print("r3: ", r3)
