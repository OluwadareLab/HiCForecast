import numpy as np

dat_test = np.load("./../data/data_test_chr6_224.npy").astype(np.float32)
print("dat_test.shape: ", dat_test.shape)
dat_test = np.stack((dat_test, dat_test, dat_test), axis=0)
print("dat_test.shape: ", dat_test.shape)
dat_test = dat_test.transpose(1,2,0,3,4)
print("dat_test.shape: ", dat_test.shape)
np.save("./../data/data_test_chr6_224_3_chan.pny", dat_test)

