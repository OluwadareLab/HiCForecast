python3 ./predict.py \
--data_path ./../data/data_96/val/data_val_chr19_96.npy \
--load_path ./../models/hic_train_log/20230728-215246/dmvfn_49.pkl \
--output_dir ./../data/data_96/predictions/single_channel_MSE_VGG_96/norm_100/batch_8/epoch_49/pred_chr19.npy \
--single_channel \
--max_HiC 100

