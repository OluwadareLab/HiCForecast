#import sys
#import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np

pearson_mean_list  = np.load('./../results/data_96/single_channel_no_vgg_96/batch_8/epoch_49/pearson_chr19.npy')
print(np.mean(pearson_mean_list, axis=1))
