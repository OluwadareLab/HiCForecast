
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

#### Contact:
Dr. Oluwatosin Oluwadare <br>
Department of Computer Science <br>
University of Colorado, Colorado Springs <br>
Email: ooluwada@uccs.edu <br>
***

## Installation
HiCForecast is written in **Python 3.8.10** and uses **R 4.3.1**. User can use `CLI` or `Docker` container to run HiCForecast. All the packages are listed below:

R Package:
* hicrep (Follow `https://github.com/TaoYang-dev/hicrep` steps to install)
  
PIP Packages:
* scikit-learn==1.3.0
* scikit-image==0.21.0
* torch==2.0.1
* torchvision==0.15.2
* opencv-python==4.8.0.74
* lpips==0.1.4
* pytorch-msssim==1.0.0
* tensorboard==2.13.0
* rpy2==3.5.13
* cooler

### Pip installation
1. First clone the git repository
   ```
   git clone https://github.com/OluwadareLab/HiCForecast.git
   cd HiCForecast
   ```
2. Install R and hicrep
3. Run the following command to install all the PIP packages..
   ```
   pip install -r requirements.txt
   ``` 

### OR

### Docker
HiCForecast runs in a Docker-containerized environment. User do not need to install anything inside container. Our image is prebuild with all the necessary packages. To run HiCForecast in a docker container, follow these steps:
1. Pull the HiCForecast docker image from docker hub using the command `docker pull oluwadarelab/hicforecast:latest`.
2. Run the HiCForecast container and mount the present working directory to the container using `docker run --rm --gpus all -itd --name hicforecast -v ${PWD}:${PWD} oluwadarelab/hicforecast`.
3. Enter into HiCForecast container using `docker exec -it hicforecast bash`.
***

## Running HiCForecast
All the scripts are available at `HiCForecast/scripts` this directory.

### Data Preprocessing
1.  Prepare Hi-C data in `.cool` format.
2.  Run `python3 makedata.py` with the following arguments:
    * `--ficool_dir`: The folder containing the input `.cool` files.
    * `--sub_mat_n`: The size of the patches to be used by the model. HiCForecast uses 64.
    * `--output_folder`: The location of the folder where the processed data will be stored.
    * `--timepoints`: These are the names of the `.cool` files in the `ficoo_dir` folder, where every file represents a timpoint. This should be a list of the names separated by a space and without the `.cool` extension (e.g `--timepoints 2-cell 4-cell 8-cell`).
    * `--chromosomes`: These are the chromosome ids as they appear in the `.cool` files that need to be processed. They should be included in a similar format as `--timepoints` above (e.g `--chromosomes chr1 chr2 chr3 chr4`).

**Note:**
We provided a bash script **makedata.sh** for data preparation. Users can run this script by updating the arguments.

#### Data Preprocessing Example With HiCForecast Data
1. Download the raw `.cool` files from ...
2. `cd` into `./HiCForecast/scripts` directory and run `sh makedata.sh` with the provided `makedata.sh` file.
3. Run `cd ./../processed_data` to go into the directory that `makedata.py` generated.
4. Run the following commands to generate a separate folder for training data
   ```
   mkdir ./train_patches/
   mv ./data_patches/data_chr1_64.npy ./train_patches/data_chr1_64.npy
   mv ./data_patches/data_chr3_64.npy ./train_patches/data_chr3_64.npy
   mv ./data_patches/data_chr4_64.npy ./train_patches/data_chr4_64.npy
   mv ./data_patches/data_chr5_64.npy ./train_patches/data_chr5_64.npy
   mv ./data_patches/data_chr7_64.npy ./train_patches/data_chr7_64.npy
   mv ./data_patches/data_chr8_64.npy ./train_patches/data_chr8_64.npy
   mv ./data_patches/data_chr9_64.npy ./train_patches/data_chr9_64.npy
   mv ./data_patches/data_chr10_64.npy ./train_patches/data_chr10_64.npy
   mv ./data_patches/data_chr11_64.npy ./train_patches/data_chr11_64.npy
   mv ./data_patches/data_chr12_64.npy ./train_patches/data_chr12_64.npy
   mv ./data_patches/data_chr13_64.npy ./train_patches/data_chr13_64.npy
   mv ./data_patches/data_chr14_64.npy ./train_patches/data_chr14_64.npy
   mv ./data_patches/data_chr15_64.npy ./train_patches/data_chr15_64.npy
   mv ./data_patches/data_chr16_64.npy ./train_patches/data_chr16_64.npy
   mv ./data_patches/data_chr17_64.npy ./train_patches/data_chr17_64.npy
   mv ./data_patches/data_chr18_64.npy ./train_patches/data_chr18_64.npy
   ```

#### Our Processed Data
We provided our raw data for chromosomes 2, 6 and 19 from Mouse Embryogenesis (Dataset 1) in the follow links:
* https://biomlearn.uccs.edu/Data/HiCForecast/chr2.tar.gz
* https://biomlearn.uccs.edu/Data/HiCForecast/chr6.tar.gz
* https://biomlearn.uccs.edu/Data/HiCForecast/chr19.tar.gz
* https://biomlearn.uccs.edu/Data/HiCForecast/test.tar.gz
* https://biomlearn.uccs.edu/Data/HiCForecast/val.tar.gz

### Train
To train the HiCForecast model follow these steps:
1. Preprocess HiC data following the *Data Preprocessing* steps.
2. Run `python3 train_1d.py` with the following arguments:
    * `--epoch`: The number of epochs to train the model for.
    * `--max_HiC`: The normalization constant. The data is cut off at this maximum value and divided by it to normalize into the range [0, 1]. HiCForecast default is 300.
    * `--patch_size`: The size of the patches to be used by the model. HiCForecast default is 64.
    * `--num_gpu`: The number of GPU's the training will utilize.
    * `--device_id`: The device id of the GPU to be used.
    * `--num_workers`: The numbe of workers to use in the dataloader.
    * `--batch_size`: The batch size. HiCForecast default is 8.
    * `--lr_scale`: The learning rate scale, which multiplies the learning rate by this value. HiCForecast default is 1.0.
    * `--block_num`: The block number is the number of MVFB blocks the architecrue will include. HiCForecast default is 9.
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

**Note:**
We provided a bash script **train.sh** for training. Users can run this script by updating the arguments.

#### Training Example with HiCForecast Data
1. Follow the steps in the *Data Preprocessing Example With HiCForecast Data* section to generate the training and validation data.
2. `cd` to the `./HiCForecast/scripts` directory and run `sh train.sh` with the provided example `train.sh` file.

### Inference
To run inference follow step:
1. Run `python3 inference.py` with the following arguments:
   * `--max_HiC`: The normalization value. The HiCForecast default is 300.
   * `--batch_max`: Normalization happens by dividing by the batch maximum. To turn off replace the argument with `--no_batch_max`. HiCForecast uses the `--no_batch_max` argument.
   * `--cut_off`:  Indicates the presence of data normalization by cuting off all values obove max_HiC and then normalizing into the range [0, 1]. Switch the argument to `--no_cut_off` to turn off the this normalization feature. HiCForecast uses the `--cut_off` argument.
   * `--sub_mat_n`: Size of the patches that the model takes as input. The HiCForecast default is 64.
   * `--model_path`: Path to the model weights location.
   * `--data_path`: Path to the input dataset location processed via steps in the Data Processing section.
   * `--output_path`: Path to the prediction output location.
   * `--file_index`: Path to input data indeces, which are needed to reassemble the prediction output into a single final matrix. These files are generated during data preprocessing.
   * `--gt_path`: Path to original ground truth matrix with shape (T, N, N), where T is the number of timesteps in the timeseries and N is the dimension of each NxN Hi-C matrix.

**Note:**
We provided a bash script **inference.sh** for inference. Users can run this script by updating the arguments.

#### Inference Example with HiCForecast Data
1. Follow the steps in the *Data Preprocessing Example With HiCForecast Data* section to generate the input data.
2. `cd` into the `./HiCForecast/scripts` directory and run the `sh inference.sh` command with the provided `inference.sh` file.
