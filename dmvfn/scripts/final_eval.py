import os
import sys
import numpy as np

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from hicrep import *
from disco_eval import *
from pearson_eval import *
from ssim_eval import * 
from psnr_eval import *

patch_sizes = [35, 60]
#patch_sizes = [35]
#dataset_num =8
model = "HiCForecast"
batch_max = False
#model = "HiC4D"
datasets = [4]
for i in datasets:
    dataset_num = i
    if dataset_num == 4 or dataset_num == 7:
        num_pred = 2
    else:
        num_pred = 3
    print("dataset_num: ", dataset_num)
    for chr_num in [2,6]:
        print("chr_num: ", chr_num)
        if model == "HiCForecast":
            if batch_max == True:
                pred_path = "./../final_prediction/{}/batch_max_trained/dataset_{}/HiCForecast_d{}_pred_chr{}_final.npy".format(model, dataset_num, dataset_num, chr_num)
            else:
                pred_path = "./../final_prediction/{}/dataset_{}/HiCForecast_d{}_pred_chr{}_final.npy".format(model, dataset_num, dataset_num, chr_num)
        elif model == "HiC4D":
            pred_path = "./../final_prediction/{}/dataset_{}/HiC4D_d{}_chr{}_predicted_final.npy".format(model, dataset_num, dataset_num, chr_num)
        gt_path =  "./../data/dataset_{}/data_64/data_gt_chr{}_64.npy".format(dataset_num, chr_num)
        if batch_max == True:
            csv_file_path = "./../final_results/{}/batch_max_trained/dataset_{}/{}_d{}_chr{}_bm.csv".format(model, dataset_num, model, dataset_num, chr_num)
        else:
            csv_file_path = "./../final_results/{}/dataset_{}/{}_d{}_chr{}.csv".format(model, dataset_num, model, dataset_num, chr_num)
        gt_mx = np.load(gt_path)
        pred_mx = np.load(pred_path)

        with open (csv_file_path, 'w', newline='') as f:
            for ps in patch_sizes:
                print("Patch Size: ", ps)
                writer = csv.writer(f)
                writer.writerow(["ps: {}".format(ps), "t4","t5","t6"])
                
                m = np.max(gt_mx)

                disco = compute_disco_avg(pred_mx, gt_mx, True, ps, num_pred = num_pred)
                print("GenomeDISCO: ", np.round(disco, 3))
                row = disco
                row.insert(0, "GenomeDISCO")
                writer.writerow(row)

                pearson = compute_pearson_avg(pred_mx, gt_mx, ps, num_pred=num_pred)
                print("Pearson: ", np.round(pearson, 3))
                row = pearson
                row.insert(0, "Pearson")
                writer.writerow(row)
                
                '''
                ssim = compute_ssim_avg(pred_mx, gt_mx, ps, m)
                print("SSIM: ", np.round(ssim, 3))
                row = ssim
                row.insert(0, "SSIM")
                writer.writerow(row)
                '''

                psnr = compute_psnr_avg(pred_mx, gt_mx, ps, m, num_pred=num_pred)
                print("PSNR: ", np.round(psnr, 3))
                row = psnr 
                row.insert(0, "PSNR")
                writer.writerow(row)

