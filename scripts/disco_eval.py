import csv
import numpy as np
from GenomeDISCO import *

def compute_disco_avg(pred_mx, gt_mx, transition, ps, num_pred=3):
    disco_list = [[] for i in range(num_pred)]
    for j in range(num_pred):
        for i in range(1, 1534 - ps, 1):
            pred_patch = pred_mx[j][i:ps+i, i:ps+i]
            gt_patch = gt_mx[j+3][i:ps+i, i:ps+i]
            if np.sum(gt_patch) == 0:
                continue
            disco = compute_reproducibility(pred_patch, gt_patch, transition, tmax=3, tmin=3)
            disco_list[j].append(disco)
            #print("j: {} i: {} disco: {}".format(j, i, disco))
    disco_avg = [[] for i in range(num_pred)]
    for j in range(num_pred):
        disco_avg[j] = sum(disco_list[j]) / len(disco_list[j]) 
    return disco_avg


if __name__ == "__main__":    
    pred_path = "./../data/data_96/190101_predictions/149/pred_chr19_final.npy"
    #pred_path = "./../data/data_50/predictions/norm_100/pred_chr19_final.npy"
    gt_path = "./../data/data_96/data_gt_chr19_96.npy"
    csv_file_path = "./../results/190101/190101_disco_49_95.csv"
    #csv_file_path = "./../results/hic4d/hic4d_disco_6_48.csv"
    ubd = 95 #inclusive
    lbd = 48 #not inclusive
    pred_mx = np.load(pred_path)
    gt_mx = np.load(gt_path)

    print("pred_mx.shape: ", pred_mx.shape)
    print("gt_mx.shape: ", gt_mx.shape)

    
    ps_list = []
    for ps in range(ubd, lbd, -1):
        davg = compute_disco_avg(pred_mx, gt_mx, True, ps)
        ps_list.append(davg)
        print("ps: {} disco_avg: {}".format(ps, davg))

    print("ps_list: ", ps_list)
    with open (csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ps", "t4","t5","t6"])
        
        for i in range(len(ps_list)):
            row = ps_list[i]
            row.insert(0, ubd - i)
            writer.writerow(row)
