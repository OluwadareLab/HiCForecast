import numpy as np

data = np.load("./../data/data_96/train/data_train_chr14_96.npy")

print("data.shape: ", data.shape)
d = np.diag(data[1011][3], 2)
with np.printoptions(threshold=np.inf):
    print("diag 0: ", d)
    print("np.where: ", np.where(data > 250)) 
    print("np.where: ", len(np.where(data > 250)[0])) 
