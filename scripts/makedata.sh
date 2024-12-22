mkdir ./../example_data/processed

python3 ./makedata.py  --ficool_dir ./../example_data/ \
--sub_mat_n 64 \
--output_folder ./../example_data/processed/ \
--timepoints PN5 early_2cell late_2cell 8cell ICM mESC_500 \
--chromosomes chr19

mkdir ./../example_data/processed/train_patches
cp ./../example_data/processed/input_patches/data_chr19_64.npy ./../example_data/processed/train_patches/