import os
import sys
import numpy as np

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

#from hicrep import *
from disco_eval import *
from pearson_eval import *
#from ssim_eval import * 
from psnr_eval import *

#patch_sizes = [35]
ps = 60
#dataset_num =8
model = "HiCForecast"
#model = "HiC4D"
#start_idx = 0
datasets = [4]
for i in datasets:
    dataset_num = i
    if dataset_num == 4 or dataset_num == 7:
        num_pred = 2
    else:
        num_pred = 3
    print("dataset_num: ", dataset_num)
    csv_file_path = "./../final_results/{}/dataset_{}/{}_d{}_ps{}_sidiag_v3.csv".format(model, dataset_num, model, dataset_num, ps)
    #csv_file_path = "./../final_results/{}/dataset_{}/{}_d{}_ps{}_si{}_v3.csv".format(model, dataset_num, model, dataset_num, ps, start_idx)

    with open (csv_file_path, 'w', newline='') as f:
        #for chr_num in [2, 6]:
        for chr_num in range(1, 23):
            #if chr_num in [2,6]:
            #    continue #skipping 2 and 6 because they have already been evaluated

            print("chr_num: ", chr_num)
            if model == "HiCForecast":
                pred_path = "./../final_prediction/{}/dataset_{}/HiCForecast_d{}_pred_chr{}_final.npy".format(model, dataset_num, dataset_num, chr_num)
                gt_path =  "./../data/dataset_{}/data_64/data_gt_chr{}_64.npy".format(dataset_num, chr_num)
            elif model == "HiC4D":
                pred_path = "./../final_matrices/dataset_{}/HiC4D_d{}_pred_chr{}.npy".format(dataset_num, dataset_num, chr_num)
                #pred_path = "./../final_matrices/dataset_{}/HiC4D_pred_d{}_chr{}.npy".format(dataset_num, dataset_num, chr_num)
                gt_path =  "./../final_matrices/dataset_{}/data_gt_chr{}.npy".format(dataset_num, chr_num)
            gt_mx = np.load(gt_path)
            pred_mx = np.load(pred_path)
            print("gt_mx.shape:", gt_mx.shape)
            print("pred_mx.shape: ", pred_mx.shape) 
            
            d = np.diag(gt_mx[0], k=0)
            #if flip == True:
            #    d = np.flip(d)
            if np.max(d) == 0:
                print("maximum is 0")
            start_idx = np.argmax((d !=0))
            print("start_idx: ", start_idx)


            gt_mx = np.load(gt_path)[:, start_idx:, start_idx:]
            pred_mx = np.load(pred_path)[:, start_idx:, start_idx:]
            print("gt_mx.shape:", gt_mx.shape)
            print("pred_mx.shape: ", pred_mx.shape) 

            #Write to CSV
            writer = csv.writer(f)
            writer.writerow(["chr: {}".format(chr_num), "t4","t5","t6"])
            
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

