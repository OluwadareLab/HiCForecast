import os
import numpy as np
A = np.load("./../data/dataset_5/data_64/test/data_test_chr20_64.npy")
print(A.shape)

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
