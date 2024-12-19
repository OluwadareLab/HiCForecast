import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
d=4
#for d in [2,4,5,6,8]:
#print("d_num: ", d)
#A = np.load("./../HiC4D_d4_test/HiC4D_d4_chr6_test.npy")
A = np.load("./../final_matrices/dataset_4/HiCForecast_pred_d4_chr2.npy")
print(A.shape)
A = np.load("./../final_matrices/dataset_4/HiC4D_pred_d4_chr2.npy")
print(A.shape)
A = np.load("./../final_matrices/dataset_{}/data_gt_chr2.npy".format(d))
#A = np.load("./../data/dataset_{}/data_50/data_gt_chr6_50.npy".format(d))
#A = np.load("./../data/dataset_4/data_50/test/data_test_chr2_50.npy")
print(A.shape)
#print(A[0][1000:1010, 1000:1010])
#print(np.max(A))

'''
for chr_num in [2, 6]:
    print("chr_num: ", chr_num)
    #file1 = "./../final_matrices/dataset_{}/data_gt_chr{}.npy".format(d_num, chr_num)
    file1 = "./../final_matrices/dataset_{}/data_gt_chr{}.npy".format(d_num, chr_num)
    #file2 = "./../data/dataset_{}/data_64/data_gt_chr{}_64.npy".format(d_num, chr_num)
    #file3 = "./../final_matrices/dataset_{}/HiC4D_pred_d{}_chr{}.npy".format(d_num, d_num, chr_num)
    #file3 = "./../final_matrices/dataset_{}/HiC4D_d{}_pred_chr{}.npy".format(d_num, d_num, chr_num)

    A = np.load(file1)
    #B = np.load(file2)
    #C = np.load(file3)

    print("A.shape: ", A.shape)
    #print("B.shape: ", B.shape)
    #print("C.shape: ", C.shape)
'''
