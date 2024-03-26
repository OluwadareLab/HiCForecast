import numpy as np
from GenomeDISCO import *

chr19 = np.load("/scratch/dpinchuk_scratch/HiCForecast/dmvfn/data/data_64/data_gt_chr19_64.npy")
ps=60
transition=True

def disco_avg_one(pred_mx, gt_mx, transition, ps):
    for i in range(1, 1534 - ps, 1):
        pred_patch = pred_mx[i:ps+i, i:ps+i]
        gt_patch = gt_mx[i:ps+i, i:ps+i]
        if np.sum(gt_patch) == 0:
            continue
        disco = compute_reproducibility(pred_patch, gt_patch, transition, tmax=3, tmin=3)
    return disco


for i in range(6):
    for j in range(i,6):
        pred_mx=chr19[i,:,:]
        gt_mx = chr19[j,:,:]
        disco = disco_avg_one(pred_mx, gt_mx, transition, ps)
        print("({}, {}): {}".format(i, j, disco))
