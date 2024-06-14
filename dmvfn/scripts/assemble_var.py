from inference import my_assemble
import numpy as np
#file_predict = './../data/data_50/predictions/norm_100/'
file_predict = "/scratch/dpinchuk_scratch/HiCForecast/dmvfn/final_prediction/HiC4D/dataset_7/HiC4D_d7_chr6_predicted.npy" 
gt_file = "/scratch/dpinchuk_scratch/HiCForecast/dmvfn/data/dataset_7/data_50/data_gt_chr6_50.npy"
file_index = "/scratch/dpinchuk_scratch/HiCForecast/dmvfn/data/dataset_7/data_50/test/data_test_index_chr6_50.npy"
output_path = "/scratch/dpinchuk_scratch/HiCForecast/dmvfn/final_prediction/HiC4D/dataset_7/HiC4D_d7_chr6_predicted_final"
dat_predict = np.load(file_predict)
gt_mx = np.load(gt_file)
num_bins = gt_mx.shape[1]
sub_mat_n = 50
hic4d=True
my_assemble(dat_predict, output_path, file_index,  num_bins, sub_mat_n, num_predictions=3, hic4d=hic4d)
