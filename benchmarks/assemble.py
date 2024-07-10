import numpy as np

def get_predictions(file_predict, file_index, num_bins, sub_mat_n=50):
    
    dat_predict = np.load(file_predict)
    print("dat_predict.shape: ", dat_predict.shape)
    dat_index = np.load(file_index)
    print("dat_index.shape: ", dat_index.shape)
    
    predictions = []
    for i in range(-3,0):
        tid = "t"+str(i+7)
        mat_chr = np.zeros((num_bins, num_bins))
        mat_n = np.zeros((num_bins, num_bins))
        for j in range(dat_predict.shape[0]):
            i1, i2 = dat_index[j, i]
            
            mat_chr[i1:(i1+sub_mat_n), i2:(i2+sub_mat_n)] += dat_predict[j, i]
            mat_n[i1:(i1+sub_mat_n), i2:(i2+sub_mat_n)] += 1
     
        mat_chr2 = np.divide(mat_chr, mat_n, out=np.zeros_like(mat_chr), where=mat_n!=0)
        predictions.append(mat_chr2)
    
    return predictions
dataset_num = 6
chr_num = 6
#num_bins = 1534 #19
#num_bins = 4544 #2
#num_bins = 3738 #6

#num_bins = 4553 #2 #d2
#num_bins = 3744 #6 d2 

file_predict = "./../dmvfn/final_prediction/HiC4D/dataset_{}/HiC4D_d{}_chr{}_predicted.npy".format(dataset_num, dataset_num, chr_num)
file_index = "./data/dataset_{}/data_50/test/data_test_index_chr{}_50.npy".format(dataset_num, chr_num)
file_out = "./../dmvfn/final_prediction/HiC4D/dataset_{}/HiC4D_d{}_chr{}_predicted_final.npy".format(dataset_num, dataset_num, chr_num)
gt_path =  "./data/dataset_{}/data_50/data_gt_chr{}_50.npy".format(dataset_num, chr_num)

gt_mx = np.load(gt_path)
num_bins = gt_mx.shape[1]
print("num_bins: ", num_bins)

predict_mat = get_predictions(file_predict, file_index, num_bins)

# for t4, t5, and t6
print(np.array(predict_mat).shape)
np.save(file_out, np.array(predict_mat))
