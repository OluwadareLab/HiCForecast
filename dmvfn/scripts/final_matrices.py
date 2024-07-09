import numpy as np
from GenomeDISCO import *

def compute_disco_avg(pred_mx, gt_mx, transition, ps, num_pred=3):
    disco_list = [[] for i in range(num_pred)]
    max_index_list = []
    for j in range(num_pred):
        #patches = range(1, gt_mx.shape[1] - ps, 1)
        #patches = [3950]
        #patches = [1910]
        patch_ind = 3362
        patches = [patch_ind]
        for i in patches:
            pred_patch = pred_mx[j][i:ps+i, i:ps+i]
            gt_patch = gt_mx[j+3][i:ps+i, i:ps+i]
            if np.sum(gt_patch) == 0:
                continue
            disco = compute_reproducibility(pred_patch, gt_patch, transition, tmax=3, tmin=3)
            print("disco: ", np.round(disco, 3))
            disco_list[j].append(disco)
            #print("j: {} i: {} disco: {}".format(j, i, disco))
        #max_index_list.append(np.argmax(disco_list[j]))
    disco_avg = [[] for i in range(num_pred)]
    #for j in range(num_pred):
        #disco_avg[j] = sum(disco_list[j]) / len(disco_list[j]) 
    #disco_list = np.array(disco_list)
    #print("disco_list shape: ", disco_list.shape)
    #print("max_index_list: ", max_index_list)
    return disco_avg

ps = 35
trans=True

for dataset_num in [6]:
    for chr_num in [2, 6]:
        #pred_path = "./../final_prediction/{}/dataset_{}/HiCForecast_d{}_pred_chr{}_final.npy".format(model, dataset_num, dataset_num, chr_num)
        #pred_path = "./../final_prediction/{}/dataset_{}/HiC4D_d{}_chr{}_predicted_final.npy".format(model, dataset_num, dataset_num, chr_num)
        #gt_path =  "./../data/dataset_{}/data_64/data_gt_chr{}_64.npy".format(dataset_num, chr_num)
        #x = np.load(gt_path)[:, 5:, 5:].astype(np.float32)

        gt_path = "./../final_matrices/dataset_{}/data_gt_chr{}.npy".format(dataset_num, chr_num)
        load_path = "./../final_matrices/dataset_{}/HiCForecast_d{}_pred_chr{}.npy".format(dataset_num, dataset_num, chr_num)
        #load_path = "./../final_matrices/dataset_{}/HiC4D_pred_d{}_chr{}.npy".format(dataset_num, dataset_num, chr_num)
        pred_mx = np.load(load_path).astype(np.float32)
        gt_mx = np.load(gt_path).astype(np.float32)
        print("dataset {} chr {} shape: ".format(dataset_num, chr_num), pred_mx.shape)
        
        compute_disco_avg(pred_mx, gt_mx, trans, ps, num_pred=3)
