python3 ./predict.py \
--data_path ./../data/data_96/val/data_val_chr19_96.npy \
--load_path ./../models/hic_train_log/20230718-110826/dmvfn_99.pkl \
--output_dir ./../data/data_96/predictions/single_channel_no_vgg_96/batch_256/epoch_99/pred_chr19.npy \
--single_channel

