import csv
import numpy as np
from corr import *
from scipy.stats import pearsonr

pred_path = "./../data/data_96/190101_predictions/149/pred_chr19_final.npy"
#pred_path = "./../data/data_50/predictions/norm_100/pred_chr19_final.npy"
gt_path = "./../data/data_96/data_gt_chr19_96.npy"
csv_file_path = "./../results/190101/190101_pearson_9_16.csv"
#csv_file_path = "./../results/hic4d/hic4d_pearson_9_16.csv"

pred_mx = np.load(pred_path)
gt_mx = np.load(gt_path)

print("pred_mx.shape: ", pred_mx.shape)
print("gt_mx.shape: ", gt_mx.shape)
mx1 = []
for s in range(40, -40, -1):
    d = np.diag(pred_mx[0], k=s)
    mx1 = mx1 + d.tolist()
mx2 = []
for s in range(40, -40, -1):
    d = np.diag(gt_mx[0], k=s)
    mx2 = mx2 + d.tolist()
pc = pearsonr(mx1, mx2)
print("pc: ", pc)
quit()

def compute_pearson_avg(pred_mx, gt_mx, ps):
    pearson_list = [[],[],[]]
    for j in range(3):
        for i in range(0, 1534 - ps, 1):
            pred_patch = pred_mx[j][i:ps+i, i:ps+i]
            gt_patch = gt_mx[j+3][i:ps+i, i:ps+i]
            if np.sum(gt_patch) == 0:
                continue
            pearson = pearsonr(pred_patch.flatten(), gt_patch.flatten())
            print("pearson: ", pearson)
            print("perason[0]: ", pearson[0])
            print("pearson[1]: ", pearson[1])
            pearson_list[j].append(pearson[0])
            #print("j: {} i: {} disco: {}".format(j, i, disco))
    pearson_avg = [[], [], []]
    for j in range(3):
        pearson_avg[j] = sum(pearson_list[j]) / len(pearson_list[j]) 
    return pearson_avg

pc_avg_list = []
for ps in range(40, 35, -1):
    pc_avg = compute_pearson_avg(pred_mx, gt_mx, ps)
    pc_list = []
    pc_avg_list.append(pc_avg)
    print("pc_avg: ", pc_avg)
    quit()
print("pc_avg_list: ", pc_avg_list)

with open (csv_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["max shift", "t4","t5","t6"])
    for i in range(len(pc_avg_list)):
        row = pc_avg_list[i]
        row.insert(0, 16 - i)
        writer.writerow(row)
'''
for ms in range(16, 8, -1):
    pc_list = []
    for i in range(3):
        pc = diagcorr(pred_mx[i], gt_mx[i+3], max_shift=ms)
        pc_avg = sum(pc) / len(pc)
        pc_list.append(pc_avg)
    pc_avg_list.append(pc_list)
'''
