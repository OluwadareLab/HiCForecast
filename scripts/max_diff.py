import numpy as np

d_num = 4
chr_num = 6
fileB = "/scratch/dpinchuk_scratch/HiCForecast/data/dataset_{}/data_64/data_gt_chr{}_64.npy".format(d_num, chr_num)

fileA =  "./../final_matrices/dataset_{}/data_gt_chr{}.npy".format(d_num, chr_num)

A = np.load(fileA)
B = np.load(fileB)

start_idx = 5

B = B[:, start_idx: , start_idx:]

d = np.max(np.abs(A - B))
print("d: ", d)


