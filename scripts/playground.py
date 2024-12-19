import os
import sys
import numpy as np
#A = np.load("/scratch/dpinchuk_scratch/HiCForecast/data/dataset_7/data_64/data_gt_chr5_64.npy")
np.set_printoptions(threshold=sys.maxsize)
#file_name = "./../new_data/GSE82185_PN5_rep1234_allValidPairs_downsampled_40000_chr2.npy"
#file_name = "./../hamster/test/hamstertest_2_64.npy"
file_name = "/scratch/dpinchuk_scratch/HiCForecast/new_resolution_data/10kb/data_64/dataset_1/data_gt_chr2_64.npy"
#dataset_num = 1
#chr_num =6
#gt_path =  "/scratch/dpinchuk_scratch/HiCForecast/new_resolution_data/10kb/data_64/dataset_{}/data_gt_chr{}_64.npy".format(dataset_num, chr_num)
#A = np.load(gt_path)
'''
file_name = "./../hamster/data_gt_hamster_6_64.npy"
A = np.load(file_name)
print(A.shape)
'''
d_num = 5
print("d_num: ", d_num)
flip = True
#for chr_num in range(1,25):
for chr_num in [2]:
    #file_name = "/scratch/dpinchuk_scratch/HiCForecast/data/dataset_{}/data_64/data_gt_chr{}_64.npy".format(d_num, chr_num)
    A = np.load(file_name)
    print(A.shape)
    print("Chr_num: ", chr_num)
    print(" ")
    for i in range(5):
        d = np.diag(A[i], k=2)
        if flip == True:
            d = np.flip(d)
        if np.max(d) == 0:
            print("maximum is 0")
        idx = np.argmax((d !=0))
        print(idx)
    #print(A)
    #print(np.diag(A, k=0))

quit()
mx1 = np.load("./../final_prediction/chr19/pred_chr19_final.npy")
mx2 = np.load("./../final_prediction/chr19/pred_chr19_final_test.npy")
max_diff = np.max(np.abs(mx1 - mx2))
print("max diff: ", max_diff)


quit()
train_folder = "/scratch/dpinchuk_scratch/HiCForecast/dmvfn/data/data_64/train/"

train_list = os.listdir(train_folder)
for file_name in train_list:
    data = np.load(train_folder + file_name)
    print("name: ", file_name)
    for i in range(1,6):
        print("i={}, max = {}".format(i, np.max(data[:, i, :, :])))

val_file = "/scratch/dpinchuk_scratch/HiCForecast/dmvfn/data/data_64/val/data_val_chr19_64.npy"
data = np.load(val_file)
print("val:")
for i in range(1,6):
    print("i={}, max = {}".format(i, np.max(data[:, i, :, :])))
quit()
with np.printoptions(threshold=np.inf):
    print("diag 0: ", d)
    print("np.where: ", np.where(data > 250)) 
    print("np.where: ", len(np.where(data > 250)[0])) 
