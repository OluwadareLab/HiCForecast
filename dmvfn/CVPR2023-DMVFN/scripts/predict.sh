python3 ./predict.py \
--data_path ./../data/data_48/val/data_val_chr19_48.npy \
--load_path ./../models/hic_train_log/20230728-214223/dmvfn_49.pkl \
--output_dir ./../data/data_48/predictions/single_channel_L1_VGG_48/norm_400/batch_8/epoch_49/pred_chr19.npy \
--single_channel \
--max_HiC 400


