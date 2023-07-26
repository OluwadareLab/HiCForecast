python3 ./predict.py \
--data_path ./../data/data_96/val/data_val_chr19_96.npy \
--load_path ./../models/hic_train_log/20230719-110557/dmvfn_49.pkl \
--output_dir ./../data/data_96/predictions/single_channel_no_vgg_96/batch_256/epoch_49/pred_chr19.npy \
--single_channel

