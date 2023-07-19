torchrun --nproc_per_node=1 \
--master_port=4321 ./train_1d.py \
--epoch 1 \
--num_gpu 1 \
--device_number 1 \
--num_workers 1 \
--batch_size 32 \
--train_dataset hic \
--val_datasets hic \
--data_train_path ./../data/data_96/train/ \
--code_test

