import csv
import numpy as np
from corr import *
from skimage.metrics import structural_similarity
#https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

#pred_path = "./../data/data_96/190101_predictions/149/pred_chr19_final.npy"
pred_path = "./../data/data_50/predictions/norm_100/pred_chr19_final.npy"
gt_path = "./../data/data_96/data_gt_chr19_96.npy"
#csv_file_path = "./../results/190101/190101_ssim_7_48_3.csv"
csv_file_path = "./../results/hic4d/hic4d_ssim_8_48_3.csv"

pred_mx = np.load(pred_path)
gt_mx = np.load(gt_path)
ubd = 48 #inclusive
lbd = 7 #not inclusive
step = 3

print("pred_mx.shape: ", pred_mx.shape)
print("gt_mx.shape: ", gt_mx.shape)
m = np.max(gt_mx)


'''
mx1 = []
for s in range(40, -40, -1):
    d = np.diag(pred_mx[0], k=s)
    mx1 = mx1 + d.tolist()
mx2 = []
for s in range(40, -40, -1):
    d = np.diag(gt_mx[0], k=s)
    mx2 = mx2 + d.tolist()
pc = pearsonr(mx1, mx2)
'''

def compute_ssim_avg(pred_mx, gt_mx, ps):
    ssim_list = [[],[],[]]
    for j in range(3):
        for i in range(0, 1534 - ps, 1):
            pred_patch = pred_mx[j][i:ps+i, i:ps+i]
            gt_patch = gt_mx[j+3][i:ps+i, i:ps+i]
            ssim = structural_similarity(pred_patch, gt_patch, data_range=m)
            #print("pearson: ", pearson)
            #print("perason[0]: ", pearson[0])
            #print("pearson[1]: ", pearson[1])
            ssim_list[j].append(ssim)
            #print("j: {} i: {} disco: {}".format(j, i, disco))
    ssim_avg = [[], [], []]
    for j in range(3):
        ssim_avg[j] = sum(ssim_list[j]) / len(ssim_list[j]) 
    return ssim_avg

def compute_spearman_avg(pred_mx, gt_mx, ps):
    spearman_list = [[],[],[]]
    for j in range(3):
        for i in range(0, 1534 - ps, 1):
            pred_patch = pred_mx[j][i:ps+i, i:ps+i]
            gt_patch = gt_mx[j+3][i:ps+i, i:ps+i]
            if np.sum(gt_patch) == 0:
                continue
            spearman = spearmanr(pred_patch.flatten(), gt_patch.flatten())
            #print("pearson: ", pearson)
            #print("perason[0]: ", pearson[0])
            #print("pearson[1]: ", pearson[1])
            spearman_list[j].append(spearman[0])
            #print("j: {} i: {} disco: {}".format(j, i, disco))
    spearman_avg = [[], [], []]
    for j in range(3):
        spearman_avg[j] = sum(spearman_list[j]) / len(spearman_list[j]) 
    return spearman_avg

val_avg_list = []
for ps in range(ubd, lbd, -1*step):
    val_avg = compute_ssim_avg(pred_mx, gt_mx, ps)
    val_avg_list.append(val_avg)
    print("patch size: {} val_avg: {}".format(ps,val_avg))
print("val_avg_list: ", val_avg_list)

with open (csv_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["patch size", "t4","t5","t6"])
    for i in range(len(val_avg_list)):
        row = val_avg_list[i]
        row.insert(0, ubd - step * i)
        writer.writerow(row)
