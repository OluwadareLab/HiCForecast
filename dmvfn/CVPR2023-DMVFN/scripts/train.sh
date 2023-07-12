torchrun --nproc_per_node=1 \
--master_port=4321 ./train.py \
--epoch 2 \
--train_dataset hic \
--val_datasets hic \
--batch_size 8 \
--resume_path ./../pretrained_models/dmvfn_city.pkl \
--num_gpu 1 \
--num_workers 1 \
--resume_epoch 1
