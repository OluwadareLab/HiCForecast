import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
base = importr('base')
utils = importr("utils")
hicrep = importr("hicrep")

dim = 96
epoch = 49
batch = 256
#loss = "single_channel_MSE_VGG"
loss = "single_channel_no_vgg"
#loss = "single_channel_default_VGG"
#loss = "single_channel_L1_no_vgg"
#loss = "single_channel_MSE_no_vgg"
GT = np.load("./../data/data_{}/data_gt_chr19_{}.npy".format(dim, dim))
B = np.load("./../data/data_{}/predictions/{}_{}/batch_{}/epoch_{}/pred_chr19_final.npy".format(dim, loss, dim, batch, epoch))

rpy2.robjects.numpy2ri.activate()

bs = B.shape
print("bs: ", bs)
nr, nc = bs[1], bs[2]
print("nr: ", nr)
print("nc: ", nc)

for i in range(3):
    PredR = ro.r.matrix(B[i], nrow=nr, ncol=nc)
    GTR = ro.r.matrix(GT[i], nrow=nr, ncol=nc)

    #ro.r.assign("mat1", Br1)
    #ro.r.assign("mat2", Br2)
    #ro.r.assign("/home/dmitryp/dpinchuk/dmvfn/CVPR2023-DMVFN/scripts/b.matrix", Br)

    #ro.r('mat1 <- read.table("/home/dmitryp/dpinchuk/dmvfn/CVPR2023-DMVFN/scripts/b.matrix")')
    #mat1 = base.read_table("B")
    #mat1 = Br
    #mat2 = mat1

    get_scc = ro.r['get.scc']
    scc_score = get_scc(GTR, PredR , resol = 40000, h = 5, lbr = 400000, ubr = 1600000)

    print("scc_scor: ", scc_score[2])
