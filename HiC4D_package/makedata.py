import numpy as np
import cooler

##### get testing input data


# time point ids
ids = ["PN5","early_2cell","late_2cell","8cell","ICM","mESC_500"]

chrs_test = ['chr2','chr6']
chrs_val = ['chr19']
resolution = 8928 #31250 for 64  #8928 for 224? #used to be 40000
sub_mat_n = int(2_000_000 // resolution)
dir_data = "./data/data_224/test/"
num_bins_all = []

#chr2: num_bins_all[0] = 4544 with 224
#chr6: num_bins_all[0] = 3738 with 224
#train data with 224 in order: [4930, 3990, 3891, 3814, 3814, 3294, 3102, 3250, 3047, 3032, 3008, 3130, 2588, 2458, 2382, 2270, 4167]
#chr19: num_bins_all[0] = 1534

def make_test_data():
    for chr in range(1,21):

        chrid = 'chr'+str(chr)
        if chr == 20:
            chrid = 'chrX'
        if chrid not in chrs_test:
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
        np.save(dir_data+"data_test_"+chrid+"_224", dat_timePoints2)
        np.save(dir_data+"data_test_index_"+chrid+"_224", dat_index2)
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
        np.save(dir_data+"data_train_"+chrid+"_224", dat_timePoints2)
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
        np.save(dir_data+"data_val_"+chrid+"_224", dat_timePoints2)
        np.save(dir_data+"data_val_index_"+chrid+"_224", dat_index2)
        print("Chromosome saved.")

def make_ground_truth():
    for chr in range(1,21):

        chrid = 'chr'+str(chr)
        if chr == 20:
            chrid = 'chrX'
        if chrid not in chrs_test:
            continue
        if chrid not in chrs_val:
            continue

        dat_timePoints = []
        dat_index = []
        ground_truth = []
        for idx, timePoint in enumerate(ids):

            ficool = dir_data+"cool_40kb_downsample/"+timePoint + ".cool"
            clr = cooler.Cooler(ficool)

            chr_len = clr.chromsizes[chrid]
            mat_chr = clr.matrix(balance=False).fetch(chrid)
            mat_chr2 = np.nan_to_num(mat_chr)
            mat_chr2 = np.expand_dims(mat_chr2, 0)
            print("mat_chr2.shape: ", mat_chr2.shape)
            ground_truth.append(mat_chr2)
            
        ground_truth = np.concatenate(ground_truth, axis=0)
        np.save("./data/data_train_"+chrid+"_64.npy", ground_truth)
        np.save(dir_data+"data_test_index_"+chrid+"_64", dat_index2)




if __name__ == "__main__":
    make_val_data()
    make_train_data()

      
