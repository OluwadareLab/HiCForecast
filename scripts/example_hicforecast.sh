mkdir ./../example_data/processed

python3 ./makedata.py  --ficool_dir ./../example_data/ \
--sub_mat_n 64 \
--output_folder ./../example_data/processed/ \
--timepoints PN5 early_2cell late_2cell 8cell ICM mESC_500 \
--chromosomes chr19

mkdir ./../example_data/processed/train_patches
cp ./../example_data/processed/input_patches/* ./../example_data/processed/train_patches/

torchrun --nproc_per_node=1 \
--master_port=4321 ./train.py \
--epoch 1 \
--max_HiC 300 \
--patch_size 64 \
--num_gpu 1 \
--device_id 0 \
--num_workers 1 \
--batch_size 8 \
--lr_scale 1.0 \
--block_num 9 \
--data_val_path ./../example_data/processed/input_patches/data_chr19_64.npy \
--data_train_path ./../example_data/processed/train_patches/ \
--resume_epoch 0 \
--early_stoppage_epochs 5 \
--early_stoppage_start 400 \
--loss single_channel_L1_no_vgg \
--val_gt_path ./../example_data/processed/data_gt_chr19_64.npy \
--val_file_index_path ./../example_data/processed/input_patches/data_index_chr19_64.npy \
--no_cut_off \
--dynamics \
--no_max_cut_off \
--no_batch_max \
--code_test 

python3 ./inference.py --max_HiC 300 \
--patch_size 64 \
--cut_off \
--model_path ./../trained_model/HiCForecast.pkl \
--data_path ./../example_data/processed/input_patches/data_chr19_64.npy \
--output_path ./../HiCForecast_prediction \
--file_index_path ./../example_data/processed/input_patches/data_index_chr19_64.npy \
--no_batch_max \
--gt_path ./../example_data/processed/data_gt_chr19_64.npy