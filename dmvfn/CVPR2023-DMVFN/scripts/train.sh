python3 -m torch.distributed.launch --nproc_per_node=8 \
--master_port=4321 ./scripts/train.py \
--train_dataset CityTrainDataset \
--val_datasets CityValDataset \
--batch_size 8 \
--num_gpu 8
