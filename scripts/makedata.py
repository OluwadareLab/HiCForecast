import os
import numpy as np
import cooler
import argparse

num_bins_all = []

def make_data_patches(ids, chromosomes, ficool_dir,  output_folder, sub_mat_n):
    for chr in chromosomes:
        chrid = chr

        dat_timePoints = []
        dat_index = []
        for idx, timePoint in enumerate(ids):

            ficool = ficool_dir +timePoint + ".cool"
            clr = cooler.Cooler(ficool)

            chr_len = clr.chromsizes[chrid]
            mat_chr = clr.matrix(balance=False).fetch(chrid)
            mat_chr2 = np.nan_to_num(mat_chr)

            bins = mat_chr2.shape[0]
            if idx == 0:
                num_bins_all.append(bins)
            subMats = []
            index = []
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

        print(chrid, dat_timePoints.shape, dat_timePoints2.shape, dat_index.shape, dat_index2.shape)
        np.save(output_folder +"input_patches/data_"+chrid+"_"+str(sub_mat_n), dat_timePoints2)
        np.save(output_folder+"input_patches/data_index_"+chrid+"_"+str(sub_mat_n), dat_index2)
        print("Chromosome saved.")

def make_data_ground_truth(ids, chromosomes, ficool_dir,  output_folder, sub_mat_n):
    for chr in chromosomes:
        chrid = chr

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
        np.save(output_folder + "data_gt_"+chrid+"_" + str(sub_mat_n) + ".npy", ground_truth)
        print("Ground truth saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ficool_dir', type=str)
    parser.add_argument('--sub_mat_n', type=int, default=64) 
    parser.add_argument('--output_folder', type=str)
    parser.add_argument("--timepoints", nargs="*", help="Names of the timepoints in .cool file in order")
    parser.add_argument("--chromosomes", nargs="*", help="Names of the chromosomes in .cool file in order")
    args = parser.parse_args()
   
    ficool_dir = args.ficool_dir
    sub_mat_n = args.sub_mat_n
    output_folder = args.output_folder
    timepoints = args.timepoints
    chromosomes = args.chromosomes

    if not os.path.exists(output_folder+"input_patches/"):
        os.makedirs(output_folder+"input_patches/")
    
    make_data_patches(timepoints, chromosomes, ficool_dir, output_folder, sub_mat_n)
    make_data_ground_truth(timepoints, chromosomes, ficool_dir, output_folder, sub_mat_n)

