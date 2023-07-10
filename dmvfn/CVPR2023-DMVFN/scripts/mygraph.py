import matplotlib.pyplot as plt
import numpy as np

dat_test = np.load("./../data/data_test_chr6_224.npy").astype(np.float32)
image_1 = dat_test[600][1]
image_2 = dat_test[600][2]
image_3 = dat_test[600][3]
min_val = 0
max_val = 100
plt.imsave(fname='./../images/hic_red_0.png', arr=image_1, cmap='Reds', format='png', vmin=min_val, vmax=max_val)
plt.imsave(fname='./../images/hic_red_1.png', arr=image_2, cmap='Reds', format='png', vmin=min_val, vmax=max_val)
plt.imsave(fname='./../images/hic_red_2.png', arr=image_3, cmap='Reds', format='png', vmin=min_val, vmax=max_val)

