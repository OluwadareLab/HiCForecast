#import sys
#import cv2
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np

#pearson_mean_list  = np.load('./../results/data_96/single_channel_no_vgg_96/batch_8/epoch_49/pearson_chr19.npy')
#print(np.mean(pearson_mean_list, axis=1))
train_path = "./../data/data_96/train/"
train_list = os.listdir(train_path)
bins=np.arange(0,1000, 10)
for file_name in train_list:
   dataset = np.load(train_path + file_name)
   hist, bin_edges = np.histogram(dataset, bins=bins)
   np.save("./../data/histograms/" + file_name + "hist.npy", hist)
   np.save("./../data/histograms/" + file_name + "bin_edges.npy", bin_edges)
   print("hist: ", hist)
   print('bind_edges: ', bin_edges)


