python3 ./inference.py --max_HiC 300 \
--patch_size 64 \
--cut_off \
--model_path ./../final_model/HiCForecast.pkl \
--data_path ./../data/data_64/val/data_val_chr19_64.npy \
--output_path ./test_output_2 \
--file_index_path /scratch/dpinchuk_scratch/HiCForecast/data/dataset_1/data_64/val/data_val_index_chr19_64.npy \
--no_batch_max \
--gt_path /scratch/dpinchuk_scratch/HiCForecast/data/dataset_1/data_64/data_gt_chr19_64.npy 



