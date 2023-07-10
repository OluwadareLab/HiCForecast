import numpy as np
import torch

from scipy.stats import pearsonr, spearmanr

from corr import*
from GenomeDISCO import*
from ssim import*


pred = np.load("./../data/chr6_predicted_final_untrained_224.npy")
ground_truth = np.load("./../data/ground_truth_chr6.npy")
print("pred.shape: ", pred.shape)
print("ground_truth.shape: ", ground_truth.shape)

r1 = diagcorr(pred[0], ground_truth[3])
r2 = diagcorr(pred[1], ground_truth[4])
r3 = diagcorr(pred[2], ground_truth[5])

disco1 = compute_reproducibility(pred[0], ground_truth[3], True)
print("disco1: ", disco1)

#ssim1 = ssim(torch.from_numpy(pred[0]).unsqueeze(0).unsqueeze(0), torch.from_numpy(ground_truth[3]).unsqueeze(0).unsqueeze(0), window_size=5)
#print("ssim1: ", ssim1)
#print("r1: ", r1)
#print("r2: ", r2)
#print("r3: ", r3)
