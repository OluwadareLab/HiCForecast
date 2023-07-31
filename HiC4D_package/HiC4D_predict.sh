python3 HiC4D_predict.py \
-f ./data/data_50/val/data_val_chr19_50.npy \
-m ResConvLSTM.pt \
-il 3 -nl 25 -hd 32 -ks 7 --GPU-index 0 -ps 1 --max-HiC 400 \
-o ./data/data_50/predictions/norm_400/chr19_predicted




