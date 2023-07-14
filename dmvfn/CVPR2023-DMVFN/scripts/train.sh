torchrun --nproc_per_node=1 \
--master_port=4321 ./train_1d.py \
--epoch 1 \
--train_dataset hic \
--val_datasets hic \
--batch_size 8 \
--num_gpu 1 \
--num_workers 0 \
--data_train_path ./../data/data_64/train/ \
--no_rgb \
--code_test

