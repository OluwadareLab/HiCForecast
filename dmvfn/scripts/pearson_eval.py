import csv
import numpy as np
from corr import *

#pred_path = "./../data/data_96/190101_predictions/149/pred_chr19_final.npy"
pred_path = "./../data/data_50/predictions/norm_100/pred_chr19_final.npy"
gt_path = "./../data/data_96/data_gt_chr19_96.npy"
#csv_file_path = "./../results/190101/190101_pearson_9_16.csv"
csv_file_path = "./../results/hic4d/hic4d_pearson_9_16.csv"

pred_mx = np.load(pred_path)
gt_mx = np.load(gt_path)

print("pred_mx.shape: ", pred_mx.shape)
print("gt_mx.shape: ", gt_mx.shape)

pc_avg_list = []
for ms in range(16, 8, -1):
    pc_list = []
    for i in range(3):
        pc = diagcorr(pred_mx[i], gt_mx[i+3], max_shift=ms)
        pc_avg = sum(pc) / len(pc)
        pc_list.append(pc_avg)
    pc_avg_list.append(pc_list)
print("pc_avg_list: ", pc_avg_list)

with open (csv_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["max shift", "t4","t5","t6"])
    for i in range(len(pc_avg_list)):
        row = pc_avg_list[i]
        row.insert(0, 16 - i)
        writer.writerow(row)
