import numpy as np
from assemble import *


file_predict = "./../HiC4D_d4_test/HiC4D_d4_chr6_test"
num_bins = np.load("./../data/dataset_4/data_50/data_gt_chr6_50.npy").shape[1]
sub_mat_n = 50

get_predictions(file_predict, num_bins, sub_mat_n, num_predictions=2, hic4d=True)
