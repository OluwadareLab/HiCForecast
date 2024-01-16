import numpy as np

data = np.load("./../data/data_96/data_gt_chr19_96.npy")

print("data.shape: ", data.shape)
d =np.diag(data[2], k=1)
print("max: ", np.max(data[2]))
quit()
with np.printoptions(threshold=np.inf):
    print("diag 0: ", d)
    print("np.where: ", np.where(data > 250)) 
    print("np.where: ", len(np.where(data > 250)[0])) 
