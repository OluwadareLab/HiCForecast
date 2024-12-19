import os
import sys
import csv
import numpy as np

d_num = 8
#ps = 60
model = "HiCForecast"
csv_file_path = "./../final_results/{}/dataset_{}/start_index_d{}_40kb.csv".format(model, d_num, d_num)
print("d_num: ", d_num)
flip = False
with open (csv_file_path, 'w', newline='') as f:
    for chr_num in range(1,20):
        file_name = "/scratch/dpinchuk_scratch/HiCForecast/data/dataset_{}/data_64/data_gt_chr{}_64.npy".format(d_num, chr_num)
        A = np.load(file_name)
        print(A.shape)
        print("Chr_num: ", chr_num)
        print(" ")
        d = np.diag(A[0], k=0)
        if flip == True:
            d = np.flip(d)
        if np.max(d) == 0:
            print("maximum is 0")
        idx = np.argmax((d !=0))
        idx = int(idx)
        print(idx)
        #print(A)
        #print(np.diag(A, k=0))

        #Write to CSV
        writer = csv.writer(f)
        writer.writerow(["chr: {}".format(chr_num), "["+str(idx) +", -1]"])
        writer.writerow(["", "["+str(idx) +", -1]"])
        writer.writerow(["", "["+str(idx) +", -1]"])
        writer.writerow(["", ""])

