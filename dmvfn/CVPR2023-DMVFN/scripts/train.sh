torchrun --nproc_per_node=1 \
--master_port=4321 ./train.py \
--epoch 5 \
--train_dataset hic \
--val_datasets hic \
--batch_size 8 \
--num_gpu 1 \
--num_workers 1 \
--resume_path /home/dmitryp/dpinchuk/dmvfn/CVPR2023-DMVFN/models/hic_train_log/20230712-225137/dmvfn_14.pkl \
--resume_epoch 1 \
--code_test

