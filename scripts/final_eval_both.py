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
#ps = 48
#dataset_num =8
#models = ["HiCForecast"]
#model = "HiC4D"
models = ["HiCForecast", "HiC4D"]
#datasets = [1,2,5,6,8]
datasets = [1, 2, 4, 5, 6, 8]
#datasets = [4]
start_idx = 0
#csv_file_path = "./../final_results/final_eval_HiC4D_all_chr2_6_ps{}.csv".format(ps)
csv_file_path = "./../final_results/final_eval_HiC4D_all_chr2_6_ps_60_48_3_clip.csv"
#csv_file_path = "./../final_results/final_eval_HiCForecast_d4_chr2_6_ps{}_3.csv".format(ps)
with open (csv_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    for i in datasets:
        dataset_num = i
        if dataset_num == 7:
            num_pred = 2
        else:
            num_pred = 3
        print("dataset_num: ", dataset_num)
        writer.writerow(["Dataset: {}".format(i), "t4","t5","t6"])
        for chr_num in [2, 6]:
        #for chr_num in range(1, 20):
            #if chr_num in [2,6]:
            #    continue #skipping 2 and 6 because they have already been evaluated
            disco_both = []
            pcc_both = []
            psnr_both = []
            for model in models:

                print("chr_num: ", chr_num)
                if model == "HiCForecast":
                    ps = 60
                    #pred_path = "./../final_prediction/{}/dataset_{}/HiCForecast_d{}_pred_chr{}_final.npy".format(model, dataset_num, dataset_num, chr_num)
                    if i == 5 or i == 6:
                        pred_path = "./../final_matrices/dataset_{}/HiCForecast_d{}_pred_chr{}.npy".format(dataset_num, dataset_num, chr_num)
                    else:
                        pred_path = "./../final_matrices/dataset_{}/HiCForecast_pred_d{}_chr{}.npy".format(dataset_num, dataset_num, chr_num)
                    gt_path =  "./../final_matrices/dataset_{}/data_gt_chr{}.npy".format(dataset_num, chr_num)
                elif model == "HiC4D":
                    ps = 48
                    start_idx = 0 # just in case
                    if i ==5 or i==6:
                        pred_path = "./../final_matrices/dataset_{}/HiC4D_d{}_pred_chr{}.npy".format(dataset_num, dataset_num, chr_num)
                    else:
                        pred_path = "./../final_matrices/dataset_{}/HiC4D_pred_d{}_chr{}.npy".format(dataset_num, dataset_num, chr_num)
                    gt_path =  "./../final_matrices/dataset_{}/data_gt_chr{}.npy".format(dataset_num, chr_num)
                    #if dataset_num == 4:
                    #    pred_path = "./../HiC4D_d4_test/HiC4D_d4_chr{}_test_assembled.npy".format(chr_num)
                print("start_idx: ", start_idx)
                gt_mx = np.load(gt_path)[:, start_idx:, start_idx:]
                #gt_mx = np.load(gt_path)
                pred_mx = np.load(pred_path)[:, start_idx:, start_idx:]
                if model == "HiC4D" and dataset_num == 4:
                    pred_mx = np.clip(pred_mx, a_min=0, a_max=None)
                print("pred_path: ", pred_path)
                print("gt_mx.shape:", gt_mx.shape)
                print("pred_mx.shape: ", pred_mx.shape) 
                
                m = np.max(gt_mx)
                #Write to CSV

                disco = compute_disco_avg(pred_mx, gt_mx, True, ps, num_pred = num_pred)
                print("GenomeDISCO: ", np.round(disco, 3))

                pearson = compute_pearson_avg(pred_mx, gt_mx, ps, num_pred=num_pred)
                print("Pearson: ", np.round(pearson, 3))
                
                '''
                ssim = compute_ssim_avg(pred_mx, gt_mx, ps, m)
                print("SSIM: ", np.round(ssim, 3))
                row = ssim
                row.insert(0, "SSIM")
                writer.writerow(row)
                '''

                psnr = compute_psnr_avg(pred_mx, gt_mx, ps, m, num_pred=num_pred)
                print("PSNR: ", np.round(psnr, 3))
                if i != 7:
                    for j in range(3):
                        disco_both.append(disco[j])
                        pcc_both.append(pearson[j])
                        psnr_both.append(psnr[j])
                '''
                else:
                    for j in range(2):
                        disco_both.append(disco[j])
                        pcc_both.append(pearson[j])
                        psnr_both.append(psnr[j])
                '''
                

                
                '''
                if model == "HiCForecast":
                    for j in range(2):
                        disco_both[j] = disco[j]
                        pcc_both[j] = pearson[j]
                        psnr_both[j] = psnr[j]
                    if i != 4:
                        j = 2
                        disco_both[j] = disco[j]
                        pcc_both[j] = pearson[j]
                        psnr_both[j] = psnr[j]
                elif model == "HiC4D":
                    for j in range(3, 5):
                        disco_both[j] = disco[j - 3]
                        pcc_both[j] = pearson[j - 3]
                        psnr_both[j] = psnr[j - 3]
                    if i != 4:
                        j = 5
                        disco_both[j] = disco[j - 3]
                        pcc_both[j] = pearson[j - 3]
                        psnr_both[j] = psnr[j - 3]
                '''
                            
            writer.writerow(["chr: {}".format(chr_num), "t4","t5","t6", "t4","t5","t6"])

            row = disco_both
            row.insert(0, "GenomeDISCO")
            writer.writerow(row)

            row = pcc_both
            row.insert(0, "Pearson")
            writer.writerow(row)

            row = psnr_both 
            row.insert(0, "PSNR")
            writer.writerow(row)
