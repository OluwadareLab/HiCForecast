python3 ./predict.py \
--data_path ./../data/data_96/val/data_val_chr19_96.npy \
--load_path ./../models/hic_train_log/20230720-211655/dmvfn_49.pkl \
--output_dir ./../data/data_96/predictions/single_channel_no_vgg_96/norm_255/batch_16/epoch_49/pred_chr19.npy \
--single_channel \
--max_HiC 255

