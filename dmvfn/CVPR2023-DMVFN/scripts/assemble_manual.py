
if training == True:
    file_predict = file_predict
    file_index = "./../data/data_{}/val/data_val_index_chr19_{}.npy".format(dim, dim)
    file_out = file_out
else:
    if loss != 'HiC4D':
        file_predict = "./../data/data_{}/predictions/{}_{}/norm_{}/batch_{}/epoch_{}/pred_chr19.npy".format(dim, loss, dim, max_HiC, batch, epoch)
        file_index = "./../data/data_{}/val/data_val_index_chr19_{}.npy".format(dim, dim)
        file_out = "./../data/data_{}/predictions/{}_{}/norm_{}/batch_{}/epoch_{}/".format(dim, loss, dim, max_HiC, batch, epoch)
        print("loss: ", loss)
    else:
        #Different file structure for HiC4D
        file_predict = "./../data/data_{}/predictions/{}_{}/norm_{}/chr19_predicted.npy".format(dim, loss, dim, max_HiC, batch, epoch)
        file_index = "./../data/data_{}/val/data_val_index_chr19_{}.npy".format(dim, dim)
        file_out = "./../data/data_{}/predictions/{}_{}/norm_{}/".format(dim, loss, dim, max_HiC, batch, epoch)
        print("loss: ", loss)
