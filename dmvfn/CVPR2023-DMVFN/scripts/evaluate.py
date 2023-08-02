import numpy as np
import torch

from scipy.stats import pearsonr, spearmanr

from corr import*
from GenomeDISCO import*
from ssim import*

dim = 96
epoch = 54
batch = 8
max_HiC = 255
#loss = "single_channel_MSE_VGG"
#loss = "single_channel_default_VGG"
loss = "single_channel_no_vgg"
#loss = "single_channel_L1_no_vgg"
#loss = "single_channel_MSE_no_vgg"
#loss = "single_channel_L1_VGG"
#loss = "HiC4D"
#default file structure:
pred = np.load("./../data/data_{}/predictions/{}_{}/norm_{}/batch_{}/epoch_{}/pred_chr19_final.npy".format(dim, loss, dim,max_HiC, batch, epoch))
ground_truth = np.load("./../data/data_{}/data_gt_chr19_{}.npy".format(dim, dim))

#ground_truth_batches = np.load("./../data/data_96/val/data_val_chr19_96.npy")
#pearson_save_path = "./../results/data_{}/{}_{}/batch_{}/epoch_{}/pearson_chr19.npy".format(dim, loss, dim, batch, epoch)
#disco_save_path = "./../results/data_{}/{}_{}/batch_{}/epoch_{}/disco_chr19_shift_35.npy".format(dim, loss, dim, batch, epoch)

#Different file structure for HiC4D
'''
pred = np.load("./../data/data_{}/predictions/{}_{}/norm_{}/chr19_predicted_final.npy".format(dim, loss, dim,max_HiC))
ground_truth = np.load("./../data/data_{}/data_gt_chr19_{}.npy".format(dim, dim))
'''

#ground_truth[ground_truth > max_HiC] = max_HiC
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

#GenomeDISCO:
'''
disco_list = []
for i in range(3):
    disco = compute_reproducibility(ground_truth[3 + i], pred[i], True)
    disco_list.append(disco)
print("disco_list: ", disco_list)
disco_list = np.array(disco_list)
print("disco_list.shape: ", disco_list.shape)
'''

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
'''      
        

#np.save(pearson_save_path, pearson_list)
#print("Saved pearson_list.")
#np.save(disco_save_path, disco_list)
#print("Saved disco_list.")

#ssim1 = ssim(torch.from_numpy(pred[0]).unsqueeze(0).unsqueeze(0), torch.from_numpy(ground_truth[3]).unsqueeze(0).unsqueeze(0), window_size=5)
#print("ssim1: ", ssim1)
#print("r1: ", r1)
#print("r2: ", r2)
#print("r3: ", r3)
