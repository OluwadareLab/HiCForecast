import numpy as np
from GenomeDISCO import *


def compute_disco_avg(pred_mx, gt_mx, transition, ps, num_pred=3):
    disco_list = []
    diag_len = gt_mx.shape[0]
    zero_counter = 0
    for i in range(1, diag_len - ps, 1):
        pred_patch = pred_mx[i:ps+i, i:ps+i]
        gt_patch = gt_mx[i:ps+i, i:ps+i]
        if np.sum(gt_patch) == 0:
            zero_counter = zero_counter + 1
            continue
        disco_score = compute_reproducibility(pred_patch, gt_patch, transition, tmax=3, tmin=3)
        disco_list.append(disco_score)
        #print("j: {} i: {} disco: {}".format(j, i, disco))
    disco_avg = sum(disco_list) / len(disco_list) 
    print("Zero_counter: ", zero_counter)
    return disco_avg

predictions = "./../final_matrices/dataset_1/HiCForecast_pred_d1_chr6.npy"
gt = "./../final_matrices/dataset_1/data_gt_chr6.npy"


pred_mx = np.load(predictions)
gt_mx = np.load(gt)

print("pred_mx.shape: ", pred_mx.shape)
print("gt_mx.shape: ", gt_mx.shape)
ps=60

result = np.zeros((3,3))
for p in range(3):
    for g in range(3,6):
        disco = compute_disco_avg(pred_mx[p], gt_mx[g], True, ps)
        print(f"Prediction {p} ground truth {g} score: {disco}")
        result[p][g-3] = disco
        
np.save("./../results/comparePredictions_d1_chr6_ps60", result)
##
