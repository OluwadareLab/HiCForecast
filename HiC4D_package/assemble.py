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

file_predict = "./data/chr6_predicted.npy"
file_index = "./data/data_test_index_chr6.npy"
file_out = "./data/chr6_predicted_final.npy"


predict_mat = get_predictions(file_predict, file_index, 1230)

# for t4, t5, and t6
print(np.array(predict_mat).shape)
np.save(file_out, np.array(predict_mat))
