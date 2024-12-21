python3 ./inference.py --max_HiC 300 \
--patch_size 64 \
--cut_off \
--model_path ./../trained_model/HiCForecast.pkl \
--data_path ./../processed_data/data_patches/data_chr19_64.npy \
--output_path ./../HiCForecast_prediction \
--file_index_path ./../processed_data/data_patches/data_index_chr19_64.npy \
--no_batch_max \
--gt_path ./../processed_data/data_gt_chr19_64.npy 



