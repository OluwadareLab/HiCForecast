import numpy as np


def get_predictions(file_predict, file_index, num_bins, sub_mat_n):
    
    dat_predict = np.load(file_predict)
    dat_index = np.load(file_index)
    
    predictions = []
    for i in range(-3,0):
        tid = "t"+str(i+7)
        mat_chr = np.zeros((num_bins, num_bins))
        mat_n = np.zeros((num_bins, num_bins))
        for j in range(dat_predict.shape[0]):
            i1, i2 = dat_index[j, i]
            #print("j: ", j) 
            mat_chr[i1:(i1+sub_mat_n), i2:(i2+sub_mat_n)] += dat_predict[j, i]
            mat_n[i1:(i1+sub_mat_n), i2:(i2+sub_mat_n)] += 1
     
        mat_chr2 = np.divide(mat_chr, mat_n, out=np.zeros_like(mat_chr), where=mat_n!=0)
        predictions.append(mat_chr2)
    
    return predictions

dim = 128
file_predict = "/home/ubuntu/dpinchuk/dmvfn/CVPR2023-DMVFN/data/data_{}/predictions/single_channel_no_vgg_{}/batch_8/epoch_49/pred_chr19.npy".format(dim, dim)
file_index = "/home/ubuntu/dpinchuk/dmvfn/CVPR2023-DMVFN/data/data_{}/val/data_val_index_chr19_{}.npy".format(dim, dim)
file_out = "./../data/data_{}/predictions/single_channel_no_vgg_{}/batch_8/epoch_49/pred_chr19_final.npy".format(dim, dim)

predict_mat = get_predictions(file_predict, file_index, 1534, dim)

# for t4, t5, and t6
print(np.array(predict_mat).shape)
np.save(file_out, np.array(predict_mat))
