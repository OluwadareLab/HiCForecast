import argparse
import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
base = importr('base')
utils = importr("utils")
hicrep = importr("hicrep")

#dim = 96
#epoch = 86
#batch = 8
#max_HiC = "255_cut_off"
#max_HiC = 255
#loss = "single_channel_MSE_VGG"
#loss = "single_channel_no_vgg"
#loss = "single_channel_default_VGG"
#loss = "single_channel_L1_no_vgg"
#loss = "single_channel_MSE_no_vgg"
#loss = "single_channel_L1_VGG"
#loss = "HiC4D"

def get_hicrep(file_predict, dim, lower_bound, ubr = 1600000, num_predictions=3):
    #lower_bound = 40000
    GT = np.load("./../data/data_{}/data_gt_chr19_{}.npy".format(dim, dim))
    B = np.load(file_predict)

    #GT = np.load("./../data/data_{}/data_gt_chr19_{}.npy".format(dim, dim))
    print("GT.shape: ", GT.shape)
    print("B.shape: ", B.shape)

    #if max_HiC != 255:
    #    GT[GT > max_HiC] = max_HiC
    #    print("Performed HiC_Max cutoff")

    '''
    if max_HiC != 255:
        GT[GT > max_HiC] = max_HiC
        print("Setting max_HiC cutoff")
    '''
    rpy2.robjects.numpy2ri.activate()

    bs = B.shape
    print("bs: ", bs)
    nr, nc = bs[1], bs[2]
    print("nr: ", nr)
    print("nc: ", nc)

    scores = np.zeros(num_predictions)
    for i in range(num_predictions):
        PredR = ro.r.matrix(B[i], nrow=nr, ncol=nc)
        GTR = ro.r.matrix(GT[i + 6 - num_predictions], nrow=nr, ncol=nc)

        get_scc = ro.r['get.scc']
        scc_score = get_scc(GTR, PredR , resol = 40000, h = 5, lbr = lower_bound, ubr = ubr)
        scores[i] = scc_score[2][0][0]
        print("scc_scor: ", scc_score[2][0][0])
    print("scores: ", scores)
    return scores

