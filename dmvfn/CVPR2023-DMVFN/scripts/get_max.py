import numpy as np
import os


train_path = "./../data/data_96/train/"
train_list = os.listdir(train_path)
max_value = 0
for chr_file in train_list:
    dataset = np.load(train_path + chr_file)
    current_max = np.max(dataset)
    print("current_max: ", current_max)
    print("argmax: ", np.unravel_index(np.argmax(dataset), dataset.shape))
    if current_max > max_value:
        max_value = current_max

print("max_value: ", max_value)
    

