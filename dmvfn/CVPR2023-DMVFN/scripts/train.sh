torchrun --nproc_per_node=1 \
<<<<<<< HEAD
--master_port=4321 ./train_1d.py \
--epoch 1 \
=======
--master_port=4321 ./train.py \
--epoch 5 \
>>>>>>> be0a2f113ced8a3895bb38f94138cf8ccd531843
--train_dataset hic \
--val_datasets hic \
--batch_size 8 \
--num_gpu 1 \
<<<<<<< HEAD
--num_workers 0 \
--data_train_path ./../data/data_64/train/ \
--no_rgb \
--code_test
=======
--num_workers 1 \
--resume_path /home/dmitryp/dpinchuk/dmvfn/CVPR2023-DMVFN/models/hic_train_log/20230712-225137/dmvfn_14.pkl \
--resume_epoch 1 \
--code_test

>>>>>>> be0a2f113ced8a3895bb38f94138cf8ccd531843
