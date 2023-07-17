torchrun --nproc_per_node=1 \
--master_port=4321 ./train_1d.py \
--epoch 35 \
--num_gpu 1 \
--device_number 1 \
--num_workers 0 \
--batch_size 8 \
--train_dataset hic \
--val_datasets hic \
--resume_path ./../models/hic_train_log_cache/20230714-195051/dmvfn_34.pkl \
--resume_epoch 34 \
--data_train_path ./../data/data_128/train/ \
--no_rgb \
--code_test

