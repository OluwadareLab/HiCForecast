import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
base = importr('base')
utils = importr("utils")
hicrep = importr("hicrep")

GT = np.load("./../data/data_96/data_gt_chr19_96.npy")
B = np.load("./../data/data_96/predictions/single_channel_MSE_VGG_96/batch_8/epoch_99/pred_chr19_final.npy")

rpy2.robjects.numpy2ri.activate()

bs = B.shape
print("bs: ", bs)
nr, nc = bs[1], bs[2]
print("nr: ", nr)
print("nc: ", nc)

PredR = ro.r.matrix(B[0], nrow=nr, ncol=nc)
GTR = ro.r.matrix(GT[0], nrow=nr, ncol=nc)

#ro.r.assign("mat1", Br1)
#ro.r.assign("mat2", Br2)
#ro.r.assign("/home/dmitryp/dpinchuk/dmvfn/CVPR2023-DMVFN/scripts/b.matrix", Br)

#ro.r('mat1 <- read.table("/home/dmitryp/dpinchuk/dmvfn/CVPR2023-DMVFN/scripts/b.matrix")')
#mat1 = base.read_table("B")
#mat1 = Br
#mat2 = mat1

get_scc = ro.r['get.scc']
scc_score = get_scc(GTR, PredR , resol = 40000, h = 5, lbr = 400000, ubr = 1600000)


#scc_out = ro.r.get.scc(mat1, mat2, resol = 40000, h = 5, lbr = 400000, ubr = 1600000)
print("scc type: ", type(scc_score))
print("scc_scor: ", scc_score[2])
