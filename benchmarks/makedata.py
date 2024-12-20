import os
import numpy as np
import cooler

##### get testing input data

dataset_num = 7

# time point ids
if dataset_num == 1:
    ids = ["PN5","early_2cell","late_2cell","8cell","ICM","mESC_500"]
    ficool_dir = "./data/cool_40kb_downsample/"
if dataset_num == 2:
    ids = ["zygote", "2-cell", "4-cell", "8-cell", "E3.5", "E7.5" ]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 3:
    ids = ["12hpa", "Early-2-cell", "Late-2-cell", "8-cell", "ICM", "TE" ]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 4:
    ids = [ "2-cell", "8-cell", "morula", "blastocyst", "6-week", "hESC" ]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 5:
    ids = ["st11", "st12", "st13", "st14", "st18", "st27"]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 6:
    ids = ["s8", "s9", "s10", "s12", "s15", "s23"]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 7:
    ids = ["hESC", "MES", "CP", "CM", "Fetal"]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
if dataset_num == 8:
    ids = [ "B", "Ba", "D2", "D4", "D6", "D8" ]
    ficool_dir = "./data/HiC4D_datasets2-8/{}/".format(dataset_num)
    
chrs_test = ['chr2','chr6']
chrs_val = ['chr19']

#resolution = 31250  #31250 for 64  #8928 for 224? #used to be 40000
#sub_mat_n = int(2_000_000 // resolution)
sub_mat_n = 64
dir_data = "./data/dataset_{}/data_{}/".format(dataset_num, sub_mat_n)
dir_data_dmvfn = "./../dmvfn/data/dataset_{}/data_{}/".format(dataset_num, sub_mat_n)
num_bins_all = []


if not os.path.exists(dir_data + "train"):
    os.makedirs(dir_data + "train")

if not os.path.exists(dir_data + "val"):
    os.makedirs(dir_data + "val")

if not os.path.exists(dir_data_dmvfn + "train"):
    os.makedirs(dir_data_dmvfn + "train")

if not os.path.exists(dir_data_dmvfn + "val"):
    os.makedirs(dir_data_dmvfn + "val")

if not os.path.exists(dir_data + "test"):
    os.makedirs(dir_data + "test")

if not os.path.exists(dir_data_dmvfn + "test"):
    os.makedirs(dir_data_dmvfn + "test")

#chr2: num_bins_all[0] = 4544 with 224
#chr6: num_bins_all[0] = 3738 with 224
#train data with 224 in order: [4930, 3990, 3891, 3814, 3814, 3294, 3102, 3250, 3047, 3032, 3008, 3130, 2588, 2458, 2382, 2270, 4167]
#chr19: num_bins_all[0] = 1534


#64
#test: [4544, 3738]
#val: [1534]
#train: [4930, 3990, 3891, 3814, 3814, 3294, 3102, 3250, 3047, 3032, 3008, 3130, 2588, 2458, 2382, 2270, 4167]

def make_test_data():
    for chr in range(1,11):

        chrid = 'chr'+str(chr)
        #if chr == 20:
        #    chrid = 'chrX'
        #if chrid not in chrs_test:
        #   continue

        dat_timePoints = []
        dat_index = []
        for idx, timePoint in enumerate(ids):

            ficool = ficool_dir +timePoint + ".cool"
            clr = cooler.Cooler(ficool)

            chr_len = clr.chromsizes[chrid]
            mat_chr = clr.matrix(balance=False).fetch(chrid)
            mat_chr2 = np.nan_to_num(mat_chr)
            #print("mat_chr2.shape: ", mat_chr2.shape)

            bins = mat_chr2.shape[0]
            if idx == 0:
                num_bins_all.append(bins)
            subMats = []
            index = []
            #print(chrid, chr_len, mat.shape)
            for i in range(0, bins, 3):
                if i+sub_mat_n >= bins:
                    continue
                subMat = mat_chr2[i:i+sub_mat_n, i:i+sub_mat_n]
                subMats.append(subMat)
                index.append((i, i))

            subMats = np.array(subMats)
            index = np.array(index)
            dat_timePoints.append(subMats)
            dat_index.append(index)


        dat_timePoints = np.array(dat_timePoints)
        dat_timePoints2 = np.transpose(dat_timePoints, (1,0,2,3))
        dat_index = np.array(dat_index)
        dat_index2 = np.transpose(dat_index, (1,0,2))

        print("num_bins_all[0]: ", num_bins_all[0])
        print("numb_bins_all ", num_bins_all)
        print(chrid, dat_timePoints.shape, dat_timePoints2.shape, dat_index.shape, dat_index2.shape)
        np.save(dir_data +"test/"+"data_test_"+chrid+"_"+str(sub_mat_n), dat_timePoints2)
        np.save(dir_data+"test/"+"data_test_index_"+chrid+"_"+str(sub_mat_n), dat_index2)
        np.save(dir_data_dmvfn+"test/"+"data_test_"+chrid+"_"+str(sub_mat_n), dat_timePoints2)
        np.save(dir_data_dmvfn+"test/"+"data_test_index_"+chrid+"_"+str(sub_mat_n), dat_index2)
        print("Chromosome saved.")


def make_train_data():
    for chr in range(1,21):

        chrid = 'chr'+str(chr)
        if chr == 20:
            chrid = 'chrX'
        if chrid in chrs_test:
            continue
        if chrid in chrs_val:
            continue

        dat_timePoints = []
        dat_index = []
        for idx, timePoint in enumerate(ids):

            ficool = "./data/cool_40kb_downsample/"+timePoint + ".cool"
            clr = cooler.Cooler(ficool)

            chr_len = clr.chromsizes[chrid]
            mat_chr = clr.matrix(balance=False).fetch(chrid)
            mat_chr2 = np.nan_to_num(mat_chr)
            print("mat_chr2.shape: ", mat_chr2.shape)

            bins = mat_chr2.shape[0]
            if idx == 0:
                num_bins_all.append(bins)
            subMats = []
            index = []
            #print(chrid, chr_len, mat.shape)
            for i in range(0, bins, 3):
                if i+sub_mat_n >= bins:
                    continue
                subMat = mat_chr2[i:i+sub_mat_n, i:i+sub_mat_n]
                subMats.append(subMat)
                index.append((i, i))

            subMats = np.array(subMats)
            index = np.array(index)
            dat_timePoints.append(subMats)
            dat_index.append(index)


        dat_timePoints = np.array(dat_timePoints)
        dat_timePoints2 = np.transpose(dat_timePoints, (1,0,2,3))
        dat_index = np.array(dat_index)
        dat_index2 = np.transpose(dat_index, (1,0,2))

        print("num_bins_all[0]: ", num_bins_all[0])
        print("numb_bins_all ", num_bins_all)
        print(chrid, dat_timePoints.shape, dat_timePoints2.shape, dat_index.shape, dat_index2.shape)
        np.save(dir_data+"train/"+"data_train_"+chrid+"_"+str(sub_mat_n), dat_timePoints2)
        np.save(dir_data_dmvfn+"train/"+"data_train_"+chrid+"_"+str(sub_mat_n), dat_timePoints2)
        #np.save(dir_data+"data_train_index_"+chrid+"_64", dat_index2)
        print("Chromosome saved.")



def make_val_data():
    for chr in range(1,21):

        chrid = 'chr'+str(chr)
        if chr == 20:
            chrid = 'chrX'
        if chrid not in chrs_val:
            continue

        dat_timePoints = []
        dat_index = []
        for idx, timePoint in enumerate(ids):

            ficool = "./data/cool_40kb_downsample/"+timePoint + ".cool"
            clr = cooler.Cooler(ficool)

            chr_len = clr.chromsizes[chrid]
            mat_chr = clr.matrix(balance=False).fetch(chrid)
            mat_chr2 = np.nan_to_num(mat_chr)
            #print("mat_chr2.shape: ", mat_chr2.shape)

            bins = mat_chr2.shape[0]
            if idx == 0:
                num_bins_all.append(bins)
            subMats = []
            index = []
            #print(chrid, chr_len, mat.shape)
            for i in range(0, bins, 3):
                if i+sub_mat_n >= bins:
                    continue
                subMat = mat_chr2[i:i+sub_mat_n, i:i+sub_mat_n]
                subMats.append(subMat)
                index.append((i, i))

            subMats = np.array(subMats)
            index = np.array(index)
            dat_timePoints.append(subMats)
            dat_index.append(index)


        dat_timePoints = np.array(dat_timePoints)
        dat_timePoints2 = np.transpose(dat_timePoints, (1,0,2,3))
        dat_index = np.array(dat_index)
        dat_index2 = np.transpose(dat_index, (1,0,2))

        print("num_bins_all[0]: ", num_bins_all[0])
        print("numb_bins_all ", num_bins_all)
        print(chrid, dat_timePoints.shape, dat_timePoints2.shape, dat_index.shape, dat_index2.shape)
        np.save(dir_data+"val/" + "data_val_"+chrid+"_"+ str(sub_mat_n), dat_timePoints2)
        np.save(dir_data+"val/" + "data_val_index_"+chrid+"_"+str(sub_mat_n), dat_index2)
        np.save(dir_data_dmvfn+"val/" + "data_val_"+chrid+"_"+ str(sub_mat_n), dat_timePoints2)
        np.save(dir_data_dmvfn+"val/" + "data_val_index_"+chrid+"_"+str(sub_mat_n), dat_index2)
        print("Chromosome saved.")


def get_predictions(file_predict, file_index, num_bins, sub_mat_n=50):
    
    dat_predict = np.load(file_predict)
    dat_index = np.load(file_index)
    
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


def make_ground_truth():
 for chr in range(1,11):
    chrid = 'chr'+str(chr)
    #if chr == 20:
    #    chrid = 'chrX'
    #if chrid not in chrs_test:
    #    continue

    dat_timePoints = []
    dat_index = []
    ground_truth = []
    for idx, timePoint in enumerate(ids):

        ficool = ficool_dir+timePoint + ".cool"
        clr = cooler.Cooler(ficool)

        chr_len = clr.chromsizes[chrid]
        mat_chr = clr.matrix(balance=False).fetch(chrid)
        mat_chr2 = np.nan_to_num(mat_chr)
        mat_chr2 = np.expand_dims(mat_chr2, 0)
        print("mat_chr2.shape: ", mat_chr2.shape)
        ground_truth.append(mat_chr2)
        
    ground_truth = np.concatenate(ground_truth, axis=0)
    print("ground_truth.shape: ", ground_truth.shape)
    np.save(dir_data + "data_gt_"+chrid+"_" + str(sub_mat_n) + ".npy", ground_truth)
    np.save(dir_data_dmvfn + "data_gt_"+chrid+"_" + str(sub_mat_n) + ".npy", ground_truth)
    print("Ground truth saved.")


if __name__ == "__main__":
    make_ground_truth()
    make_test_data()
      
