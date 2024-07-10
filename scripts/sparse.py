import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
base = importr('base')
utils = importr("utils")
hiccompare = importr("HiCcompare")

#installation:
#utils.install_packages("BiocManager")
#biocmanager = importr("BiocManager")
#biocmanager.install("HiCcompare")

file_path = './../data/dataset_1/data_64/data_gt_chr2_64.npy'

GT = np.load(file_path)
T = GT[0]

bs = GT.shape
print("bs: ", bs)
nr, nc = bs[1], bs[2]
print("nr: ", nr)
print("nc: ", nc)
rpy2.robjects.numpy2ri.activate()


TR = ro.r.matrix(T, nrow=nr, ncol=nc)
#get_colnames = ro.r['colnames']
#print(get_colnames(TR))
#base.colnames(TR)[0] = "a"
print(ro.r('colnames(TR)'))
#colnames(T)
