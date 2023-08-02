python3 ./predict.py \
--data_path ./../data/data_96/val/data_val_chr19_96.npy \
--load_path ./../models/hic_train_log_cache/20230717-164102/dmvfn_54.pkl \
--output_dir ./../data/data_96/predictions/single_channel_no_vgg_96/norm_255_cut_off/batch_8/epoch_54/pred_chr19.npy \
--single_channel \
--max_HiC 255

