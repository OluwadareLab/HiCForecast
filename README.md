
# HiCForecast: Dynamic Network Optical Flow Estimation Algorithm for Spatiotemporal Hi-C Data Forecasting
***
#### [OluwadareLab, University of Colorado, Colorado Springs](https://uccs-bioinformatics.com/)
***
#### Developers:

Dmitry Pinchuk <br>
Department of Computer Science <br>
University of Wisconsin-Madison <br>
Email: dpinchuk@wisc.edu <br>
<br>
H M A Mohit Chowdhury<br>
Department of Computer Science<br>
University of Colorado Colorado Springs<br>
Email: hchowdhu@uccs.edu<br>
<br>
Abhishek Pandeya<br>
Department of Computer Science<br>
University of Colorado Colorado Springs<br>
Email: apandeya@uccs.edu<br>
<br>

#### Contact:

Dr. Oluwatosin Oluwadare <br>
Department of Computer Science <br>
University of Colorado, Colorado Springs <br>
Email: ooluwada@uccs.edu <br>
***

## Build Instructions
HiCForecast runs in a Docker-containerized environment. Before cloning this repository and attempting to build, install the Docker engine. To install and build HiCForecast follow these steps. 
1. Clone this repository locally using the command `https://github.com/OluwadareLab/HiCForecast.git && cd HiCForecast`.
2. Pull the HiCForecast docker image from docker hub using the command `docker pull oluwadarelab/hicforecast:latest`. This may take a few minutes. Once finished, check that the image was sucessfully pulled using `docker image ls`.
3. Pull the data preprocessing and benchmark model image from docker hub using the command `docker pull oluwadarelab/hicforecastdata-preprocessing:latest`. This may take a few minutes. Once finished, check that the image was sucessfully pulled using `docker image ls`.
4. Run the HiCForecast container and mount the present working directory to the container using `docker run --rm --gpus all -it --name hicforecast -v ${PWD}:${PWD} oluwadarelab/hicforecast`.
5. Run the data preprocessing and benchmark model container and mount the present working directory to the container using `docker run --rm --gpus all -it --name hicforecast_data -v ${PWD}:${PWD} oluwadarelab/hicforecastdata-preprocessing:latest`. This may take a few minutes. Once finished, check that the image was sucessfully pulled using `docker image ls`.
6.  `cd` to your home directory.
***

## Data Preprocessing
The data preprocessing extracts patches from the .cool dataset and converts them into .npy files accepted by the HiCForecast model. To run data preprocessing follow these steps.
1. Enter the Docker container for data preprocessing by using the command `docker exec -it hicforecast_data bash`.
2. `cd` to the HiCForecast directory.
<!--3. If it does not exist yet, make a new directory for data by using the command `mkdir data` and `cd ./data`.
<!--4. Install the Mouse Embryogenesis (Dataset 1) file cool_40kb_downsample.tar.gz from https://biomlearn.uccs.edu/Data/HiCForecast/ by running the command `wget https://biomlearn.uccs.edu/Data/HiCForecast/cool_40kb_downsample.tar.gz` and then extract it by running the command `tar -xf cool_40kb_downsample.tar.gz`.
5. Enter the scripts folder by running `cd ./../scripts`.
6. Run the data extraction by using the command `python3 makedata.py`. -->
3.  Download your Hi-C data in a `.cool` format to a location of your choice.
4.  Edit the **makedata.sh** bash script file to include the required arguments. Include a space followed by a backslash to indicate a new line at the end of each argument (e.g. `--sub_mat_n 64 \`). Arguments of all types including strings should be included without quotes around them (e.g. `--ficool_dir my_file_path/folder/ \`).
    * `--ficool_dir`: The folder containing the input `.cool` files.
    * `--sub_mat_n`: The size of the patches to be used by the model. HiCForecast uses 64.
    * `--output_folder`: The location of the folder where the processed data will be stored.
    * `--timepoints`: These are the names of the `.cool` files in the `ficoo_dir` folder, where every file represents a timpoint. This should be a list of the names separated by a space and without the `.cool` extension (e.g `--timepoints 2-cell 4-cell 8-cell \`).
    * `--chromosomes`: These are the chromosome ids as they appear in the `.cool` files that need to be processed. They should be included in a similar format as `--timepoints` above (e.g `--chromosomes chr1 chr2 chr3 chr4 \`).
6. Run data preprocessing by using the command `sh makedata.sh`.
7. Exit the data preprocessing Docker container by running the command `exit`.

## Using Our Processed Data
To download our preprocessed data for chromosomes 2, 6 and 19 from Mouse Embryogenesis (Dataset 1) follow these steps.
1. `cd` to the data folder in the HiCForecast repository.
2. Make a new folder called data_64 by running the command `mkdir ./data_64` and then enter it with `cd ./data_64`.
3. Download and extract the files by running the commands
   ```
   wget https://biomlearn.uccs.edu/Data/HiCForecast/chr2.tar.gz
   wget https://biomlearn.uccs.edu/Data/HiCForecast/chr6.tar.gz
   wget https://biomlearn.uccs.edu/Data/HiCForecast/chr19.tar.gz
   wget https://biomlearn.uccs.edu/Data/HiCForecast/test.tar.gz
   wget https://biomlearn.uccs.edu/Data/HiCForecast/val.tar.gz
   tar -xf chr2.tar.gz
   tar -xf chr6.tar.gz
   tar -xf chr19.tar.gz
   tar -xf test.tar.gz
   tar -xf val.tar.gz
   ```
## Training
To train the HiCForecast model follow these steps
1. If you have not done so yet, enter the HiCForecast Docker container by running the command `docker exec -it hicforecast bash`.
2. `cd` into the HiCForecast/scripts folder.
3. Edit the **train.sh** bash script file to include the required arguments. Include a space followed by a backslash to indicate a new line at the end of each argument (e.g. `--epoch 100 \`). An example, of the train.sh bash script is included in the scripts folder of the repository.
    * `--master_port=4321 ./train_1d.py \`: Leave the first argument unchanged.
    * `--epoch`: The number of epochs to train the model for.
    * `--max_HiC`: The normalization constant. The data is cut off at this maximum value and divided by it to normalize into the range [0, 1]. HiCForecast default is 300.
    * `--patch_size`: The size of the patches to be used by the model. HiCForecast default is 64.
    * `--num_gpu`: The number of GPU's the training will utilize.
    * `--device_id`: The device id of the GPU to be used.
    * `--num_workers`: The numbe of workers to use in the dataloader.
    * `--batch_size`: The batch size. HiCForecast default is 8.
    * `--lr_scale`: The learning rate scale, which multiplies the learning rate by this value. HiCForecast default is 1.0.
    * `--block_num`: The block number is the number of MVFB blocks the architecrue will include. HiCForecast default is 9.
    * `--train_dataset hic \`: Leave this argument unchanged.
    * `--val_datasets hic \`: Leave this argument unchanged.
    * `--data_val_path`: The path to the validation data. 
    * `--data_train_path`: The path to the training dataset.
    * `--resume_epoch`: The epoch from which to resume training the model.
    * `--early_stoppage_epochs`: The number of epochs to wait before validation imporvement happens to terminate training with early stoppage. HiCForecast default is 5.
    * `--early_stoppage_start`: Epoch from which to start applying early stoppage. HiCForecast default is 400 (effectively early stoppage was not used).
    * `--loss`: The loss function used in training. Choices include: `single_channel_L1_no_vgg`, `single_channel_default_VGG`, `single_channel_MSE_no_vgg`, `single_channel_MSE_VGG`, and `single_channel_L1_VGG`. HiCForecast default is `single_channel_L1_no_vgg`.
    * `--val_gt_path`: The path to the ground truth for the validation file.
    * `--val_file_index_path`: The path to the file index for the validation file, which was generated in the Data Preprocessing step. 
    * `--cut_off`: Indicates the presence of data normalization by cuting off all values obove max_HiC and then normalizing into the range [0, 1]. Switch the argument to `--no_cut_off` to turn off the this normalization feature. HiCForecast default includes the `--cut_off` argument.
    * `--dynamics`: Indicates the presence of the routing module and dynamic aspect of the architecture. Switch the argument to `--no_dynamics` to turn off the routing module and dynamic aspect of the architecture. HiCForecast default includes the `--dynamics` argument.
    * `--max_cut_off`: Indicates that data normalization will happen by dividing by the maximum of the input data instead of by HiC_max. Switch this argument to `--no_max_cut_off` to turn off this feature. HiCForecast default includes the `--no_max_cut_off` argument.
    * `--batch_max`: Normalization happens by dividing by the batch maximum. To turn off replace the argument with `--no_batch_max`. HiCForecast includes the `--no_batch_max` argument.
    * `--code_test`: Indicates that the training process will run in test mode, cycling through only a few batches during each epoch, to quickly test the entire training pipeline. In test mode the model will save the logs in a separate test log folder. To turn off test mode and enable the regular training process, replace this argument with `--no_code_test`.
4. Run the bash script with the command `sh train.sh` to initiate training.

## Inference
To run inference follow these steps.
1. If you have not done so yet, enter the HiCForecast Docker container by running the command `docker exec -it hicforecast bash`.
2. `cd` into the HiCForecast/scripts folder.
3. Edit the **inference.sh** file to include the required arguments. Include a space followed by a backslash to indicate a new line at the end of each argument (e.g. `--max_HiC 300 \`).
   * `--max_HiC`: The normalization value. The HiCForecast default is 300.
   * `--batch_max`: Normalization happens by dividing by the batch maximum. To turn off replace the argument with `--no_batch_max`. HiCForecast uses the `--no_batch_max` argument.
   * `--cut_off`:  Indicates the presence of data normalization by cuting off all values obove max_HiC and then normalizing into the range [0, 1]. Switch the argument to `--no_cut_off` to turn off the this normalization feature. HiCForecast uses the `--cut_off` argument.
   * `--sub_mat_n`: Size of the patches that the model takes as input. The HiCForecast default is 64.
   * `--model_path`: Path to the model weights location.
   * `--data_path`: Path to the input dataset location processed via steps in the Data Processing section.
   * `--output_path`: Path to the prediction output location.
   * `--file_index`: Path to input data indeces, which are needed to reassemble the prediction output into a single final matrix. These files are generated during data preprocessing.
   * `--gt_path`: Path to original ground truth matrix with shape (T, N, N), where T is the number of timesteps in the timeseries and N is the dimension of each NxN Hi-C matrix.
4. Run inference by using the command `sh inference.sh`.


