import numpy as np
from GenomeDISCO import *

pred_data = np.load("/scratch/dpinchuk_scratch/HiCForecast/dmvfn/data/data_64/data_gt_chr19_64.npy")
out_name = "chr19_correlation_35"
ps=35
transition=True

def disco_avg_one(pred_mx, gt_mx, transition, ps):
    for i in range(1, 1534 - ps, 1):
        pred_patch = pred_mx[i:ps+i, i:ps+i]
        gt_patch = gt_mx[i:ps+i, i:ps+i]
        if np.sum(gt_patch) == 0:
            continue
        disco = compute_reproducibility(pred_patch, gt_patch, transition, tmax=3, tmin=3)
    return disco

out_data = np.zeros((6,6))
for i in range(6):
    for j in range(i,6):
        pred_mx=pred_data[i,:,:]
        gt_mx = pred_data[j,:,:]
        disco = disco_avg_one(pred_mx, gt_mx, transition, ps)
        out_data[i][j] = disco
        out_data[j][i] = disco
        print("({}, {}): {}".format(i, j, disco))


np.save("/scratch/dpinchuk_scratch/HiCForecast/dmvfn/results/plot_data/"+out_name, out_data)
