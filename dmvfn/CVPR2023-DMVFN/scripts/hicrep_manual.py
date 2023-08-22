
        if loss ==  "HiC4D":
            #HiC4D file structure
            print("Loss: {}".format(loss))
            GT = np.load("./../data/data_{}/data_gt_chr19_{}.npy".format(dim, dim))
            B = np.load("./../data/data_{}/predictions/{}_{}/norm_{}/chr19_predicted_final.npy".format(dim, loss, dim, max_HiC))
        else:
            #default file structure:
            print("Loss: {}".format(loss))
            GT = np.load("./../data/data_{}/data_gt_chr19_{}.npy".format(dim, dim))
            B = np.load("./../data/data_{}/predictions/{}_{}/norm_{}/batch_{}/epoch_{}/pred_chr19_final.npy".format(dim, loss, dim, max_HiC, batch, epoch))
