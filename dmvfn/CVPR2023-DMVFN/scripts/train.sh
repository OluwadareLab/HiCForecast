torchrun --nproc_per_node=1 \
--master_port=4321 ./train_1d.py \
--epoch 150 \
--train_dataset hic \
--val_datasets hic \
--batch_size 8 \
--num_gpu 1 \
--num_workers 0 \
--data_train_path ./../data/data_64/train/ \
--no_rgb \
--resume_path /home/ubuntu/dpinchuk/dmvfn/CVPR2023-DMVFN/models/hic_train_log/20230713-223916/dmvfn_19.pkl \
--resume_epoch 19 \
--no_code_test

