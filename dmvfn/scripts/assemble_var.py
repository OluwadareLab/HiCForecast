from assemble import *
file_predict = './../data/data_50/predictions/norm_100/'
num_bins = 1534
sub_mat_n = 50
hic4d=True
get_predictions(file_predict, num_bins, sub_mat_n, num_predictions=3, hic4d=hic4d)
