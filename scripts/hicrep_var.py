from hicrep import *
dim = 96
file_predict = './../data/data_50/predictions/norm_100/pred_chr19_final.npy'
#GT = "./../data/data_{}/data_gt_chr19_{}.npy".format(dim, dim)
#dim = 50
lower_bound = 0
ubr = 47*40000
get_hicrep(file_predict, dim, lower_bound, ubr = ubr, num_predictions=3)
