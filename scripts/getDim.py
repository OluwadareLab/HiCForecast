import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

dataset_num = 4
chr_num = 2
#file_name = "/scratch/dpinchuk_scratch/HiCForecast/data/dataset_2/data_64/data_gt_chr1_64.npy"
gt_path =  "./../final_matrices/dataset_{}/data_gt_chr{}.npy".format(dataset_num, chr_num)
pred_path = "./../final_matrices/dataset_{}/HiC4D_pred_d{}_chr{}.npy".format(dataset_num, dataset_num, chr_num)
HiCForecast_pred_path = "./../final_matrices/dataset_{}/HiCForecast_pred_d{}_chr{}.npy".format(dataset_num, dataset_num, chr_num)

gt = np.load(gt_path)
print("gt shape: ", gt.shape)
pred = np.load(pred_path)
print("pred.shape: ", pred.shape)
HiCForecast = np.load(HiCForecast_pred_path)
print("HiCForecast.shape: ", HiCForecast.shape)
print(pred[0][200:250, 200:250])
