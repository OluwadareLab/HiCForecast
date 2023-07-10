


python HiC4D_predict.py \
-f ./data/data_test_chr6.npy \
-m ResConvLSTM.pt \
-il 3 -nl 25 -hd 32 -ks 7 --GPU-index 0 -ps 1 --max-HiC 100 \
-o ./data/chr6_predicted




