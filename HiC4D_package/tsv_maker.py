import numpy as np

m = np.load("./data/data_64/data_gt_chr19_64.npy")
with open('tsv_test.txt', 'w') as f:
    it = np.nditer(m[0], flags=['multi_index'])
    while not it.finished:
        f.write('{}\t{}\t{}\n'.format(it.multi_index[0], it.multi_index[1], it[0]))
        it.iternext()
