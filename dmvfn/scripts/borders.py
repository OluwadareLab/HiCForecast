import numpy as np
import sys
import os


np.set_printoptions(threshold=sys.maxsize)
#k=1
#model="HiCForecast"
#model="HiC4D"
for model in ["HiC4D"]:
    for dataset_num in [5,6]:
        for chr_num in [2, 6]:
            pred_path = "./../final_prediction/{}/dataset_{}/{}_d{}_pred_chr{}_final.npy".format(model, dataset_num, model, dataset_num, chr_num)
            #pred_path = "./../final_prediction/{}/dataset_{}/HiC4D_d{}_chr{}_predicted_final.npy".format(model, dataset_num, dataset_num, chr_num)
            #gt_path =  "./../data/dataset_{}/data_64/data_gt_chr{}_64.npy".format(dataset_num, chr_num)
            mx = np.load(pred_path).astype(np.float32)
            #print("mx shape: ", mx.shape)
            mx = mx * 100.0
            #save_path_gt = "./../final_matrices/dataset_{}/data_gt_chr{}.npy".format(dataset_num, chr_num)
            save_path_pred = "./../final_matrices/dataset_{}/{}_d{}_pred_chr{}_final.npy".format(dataset_num, model, dataset_num, chr_num)
            #save_path = "./../final_matrices/dataset_{}/HiCForecast_pred_d{}_chr{}.npy".format(dataset_num, dataset_num, chr_num)
            #save_path = "./../final_matrices/dataset_{}/HiC4D_pred_d{}_chr{}.npy".format(dataset_num, dataset_num, chr_num)
            #np.save(save_path, mx)
            print("dataset {} chr {}".format(dataset_num, chr_num))
            np.save(save_path_pred, mx)
            #os.system('cp {} {}'.format(gt_path, save_path_gt))
            #os.system('cp {} {}'.format(pred_path, save_path_pred))
            #print(np.diag(gt_mx[0], k=k)[-10:])
            '''
            for t in range(6):
                print(np.diag(mx[t], k=k)[-10:])
            '''
