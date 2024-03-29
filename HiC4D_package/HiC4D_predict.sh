python3 HiC4D_predict.py \
-f ./data/dataset_7/data_50/test/data_test_chr6_50.npy \
-m ResConvLSTM.pt \
-il 3 -nl 25 -hd 32 -ks 7 --GPU-index 1 -ps 1 --max-HiC 100 \
-o ./../dmvfn/final_prediction/HiC4D/dataset_7/HiC4D_d7_chr6_predicted




