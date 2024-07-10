import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
base = importr('base')
utils = importr("utils")
hicrep = importr("hicrep")

dim = 96
lower_bound = 400000

GT = np.load("./../data/data_{}/data_gt_chr19_{}.npy".format(dim, dim))
gts = GT.shape
print("GT.shape: ", gts)
heat_map = np.zeros((6,6))

rpy2.robjects.numpy2ri.activate()


nr, nc = gts[1], gts[2]
print("nr: ", nr)
print("nc: ", nc)

for i in range(6):
    GTR1 = ro.r.matrix(GT[i], nrow=nr, ncol=nc)
    for j in range(i, 6):
        GTR2 = ro.r.matrix(GT[j], nrow=nr, ncol=nc)
    
        get_scc = ro.r['get.scc']
        scc_score = get_scc(GTR1, GTR2 , resol = 40000, h = 5, lbr = lower_bound, ubr = 1600000)
        print("i, j: {} {}".format(i, j))
        print("scc_score[2]: ", scc_score[2][0][0])
        heat_map[i][j] = scc_score[2][0][0]
        heat_map[j][i] = scc_score[2][0][0]

print("heat_map: ", heat_map)
np.save("./../data/HiCRep_compare_19_{}".format(lower_bound), heat_map)
