import os
import numpy as np

def get_predictions(file_predict, num_bins, sub_mat_n, file_index, num_predictions=3, hic4d=False):
    
    #if not os.path.exists(file_predict):
    #    os.makedirs(file_predict)

    dim = sub_mat_n
    #file_index = "./../data/data_{}/val/data_val_index_chr19_{}.npy".format(dim, dim)
    #file_index = "./../data/dataset_4/data_50/test/data_test_index_chr6_50.npy"
    
    dat_predict = np.load(file_predict + ".npy")
    #dat_index = np.load(file_index) 
    dat_index = np.load(file_index) 
    #print("dat_index.shape: ", dat_index.shape)
    
    predictions = []
    for i in range(-num_predictions,0):
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
    
    print(np.array(predictions).shape)
    np.save(file_predict + "_final.npy", np.array(predictions))
    return predictions

#dim = 96
#max_HiC = 255
#epoch = 49
#batch = 8
#max_HiC = "255_cut_off"
#loss = "single_channel_no_vgg"
#loss = "single_channel_default_VGG"
#loss = "single_channel_L1_no_vgg"
#loss = "single_channel_MSE_no_vgg"
#loss = "single_channel_MSE_VGG"
#loss = "single_channel_L1_VGG"
#loss = "HiC4D"
#predict_mat = get_predictions(file_predict, file_index, 1534, dim)
# for t4, t5, and t6
#print(np.array(predict_mat).shape)
#np.save(file_out + "pred_chr19_final.npy", np.array(predict_mat))
