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

dim = 64
#chr_num = 2
#dataset_num = 6
'''
GT = np.load("./../data/dataset_{}/data_{}/data_gt_chr{}_{}.npy".format(dataset_num, dim, chr_num, dim))
gts = GT.shape
print("GT.shape: ", gts)
heat_map = np.zeros((6,6))
'''
transition = True
ps = 35
datasets = [2]
for dataset_num in datasets:
    for chr_num in [2]:
        GT = np.load("./../data/dataset_{}/data_{}/data_gt_chr{}_{}.npy".format(dataset_num, dim, chr_num, dim)).astype(np.float32)
        PR = np.load("./../final_prediction/HiC4D/dataset_{}/HiC4D_d{}_chr{}_predicted_final.npy".format(dataset_num, dataset_num, chr_num)).astype(np.float32)
        #PR = np.load("./../final_prediction/HiCForecast/dataset_{}/HiCForecast_d{}_pred_chr{}_final.npy".format(dataset_num, dataset_num, chr_num)).astype(np.float32)
        print("GT.shape: ", GT.shape)
        print("PR.shape: ", PR.shape)
        print("sum(GT[-1]): ", np.sum(GT[-1]))
        print("sum(PR[-1]): ", np.sum(PR[-1]))
        #print("PR[2][500:510, 500:510]: ", PR[2][500:510, 500:510])
        #print("GT[5][500:510, 500:510]: ", GT[5][500:510, 500:510])
        #print("For loop: ")
        for k in range(3):
            GT[3 + k, :, :] = PR[k, :, :]
        gts = GT.shape
        print("GT.shape: ", gts)
        print("sum(GT[-1]: ", np.sum(GT[5]))
        print("sum(PR[-1]): ", np.sum(PR[-1]))
        #print("GT[5][500:510, 500:510]: ", GT[5][500:510, 500:510])
        heat_map = np.zeros((6,6))
        for i in range(6):
            for j in range(i, 6):
                 
                m1 = GT[i]
                m2 = GT[j]
                #disco = compute_reproducibility(m1, m2, transition, tmax=3, tmin=3)
                disco = compute_disco_avg(m1, m2, transition, ps, num_pred=3)

                print("{}, {}: {}".format(i, j, disco))
                heat_map[i][j] = disco
                heat_map[j][i] = disco

        print("heat_map: ", heat_map)
        #np.save("/scratch/dpinchuk_scratch/HiCForecast/dmvfn/results/plot_data/disco_compare_avgps_{}_ds{}_chr{}".format(ps, dataset_num, chr_num), heat_map)
        np.save("/scratch/dpinchuk_scratch/HiCForecast/dmvfn/results/plot_data/disco_compare_HiC4D_avgps_{}_ds{}_chr{}".format(ps, dataset_num, chr_num), heat_map)



