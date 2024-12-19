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
datasets = [1]
start_idx = 300
for i in datasets:
    dataset_num = i
    if dataset_num == 4 or dataset_num == 7:
        num_pred = 2
    else:
        num_pred = 3
    print("dataset_num: ", dataset_num)
    for chr_num in [2,6]:
        print("chr_num: ", chr_num)
        pred_path = "/scratch/dpinchuk_scratch/HiCForecast/new_resolution_data/10kb/data_64/dataset_{}/final_prediction/HiCForecast_d{}_pred_chr{}_final.npy".format(dataset_num, dataset_num, chr_num)
        gt_path =  "/scratch/dpinchuk_scratch/HiCForecast/new_resolution_data/10kb/data_64/dataset_{}/data_gt_chr{}_64.npy".format(dataset_num, chr_num)
        csv_file_path = "/scratch/dpinchuk_scratch/HiCForecast/new_resolution_data/10kb/data_64/dataset_{}/results/HiCForecast_10kb_d{}_chr{}_si{}.csv".format(dataset_num, dataset_num, chr_num, start_idx)
        gt_mx = np.load(gt_path)[:, start_idx:, start_idx:]
        pred_mx = np.load(pred_path)[:, start_idx:, start_idx:]

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

