import os
import numpy as np
import cooler

##### get testing input data

dataset_num = 2

# time point ids
if dataset_num == 1:
    ids = ["PN5","early_2cell","late_2cell","8cell","ICM","mESC_500"]
    ficool_dir = "./data/cool_40kb_downsample/"
if dataset_num == 2:
    ids = ["zygote", 62e6]
    meta_data = [62e6, 23] #[number of allValidRead pairs after downsampling, num of chromosomes in genome]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 3:
    ids = ["12hpa", "Early-2-cell", "Late-2-cell", "8-cell", "ICM", "TE" ]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 4:
    ids = [ "2-cell", "8-cell", "morula", "blastocyst", "6-week", "hESC" ]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 5:
    ids = ["st11", "st12", "st13", "st14", "st18", "st27"]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 6:
    ids = ["s8", "s9", "s10", "s12", "s15", "s23"]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 7:
    ids = ["hESC", "MES", "CP", "CM", "Fetal"]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 8:
    ids = [ "B", "Ba", "D2", "D4", "D6", "D8" ]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
    

timePoint = ids[0]
ficool = ficool_dir +timePoint + ".cool"
clr = cooler.Cooler(ficool)
names = clr.chromnames
#print("chromnames: ", names)
chrid = 'chr1'
#print("chr_len: ", chr_len)
chr_lengths = []

for chr in range(1,20):
    chrid = 'chr'+str(chr)
    chr_len = clr.chromsizes[chrid]
    chr_lengths.append(chr_len)
    print("{}: {}".format(chrid, chr_len))
avg_len = sum(chr_lengths)/len(chr_lengths)
print("avg_len: ",  avg_len)
#maxHiC = meta_data[0] / (meta_data[1] * avg_len)
maxHiC = meta_data[0]/ sum(chr_lengths)
print("maxHiC: ", maxHiC)

