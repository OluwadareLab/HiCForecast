
# HiCForecast: Dynamic Network Optical Flow Estimation Algorithm for Spatiotemporal Hi-C Data Forecasting
***
#### [OluwadareLab, University of Colorado, Colorado Springs](https://uccs-bioinformatics.com/)
***
#### Developers:

Dmitry Pinchuk <br>
Department of Computer Science <br>
University of Wisconsin-Madison <br>
Email: dpinchuk@wisc.edu <br>

Mohit and Abhishek add their contact info here: <br>

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
3. If it does not exist yet, make a new directory for data by using the command `mkdir data` and `cd ./data`.
4. Install the Mouse Embryogenesis (Dataset 1) file cool_40kb_downsample.tar.gz from https://biomlearn.uccs.edu/Data/HiCForecast/ by running the command `wget https://biomlearn.uccs.edu/Data/HiCForecast/cool_40kb_downsample.tar.gz` and then extract it by running the command `tar -xf cool_40kb_downsample.tar.gz`.
5. Enter the scripts folder by running `cd ./../scripts`.
6. Run the data extraction by using the command `python3 makedata.py`.
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
3. Edit the train.sh bash script file to include the required arguments. Include a space followed by a backslash to indicate a new line at the end of each argument (e.g. `--epoch 100 \`).
    * `--master_port=4321 ./train_1d.py \`: Leave the first argument unchanged.
    * `--epoch`: The number of epochs to train the model for.
    * `--max_HiC`: The normalization constant. The data is cut off at this maximum value and divided by it to normalize into the range [0, 1].
    * `--patch_size`: The size of the patches to be used by the model.
    * `--num_gpu`: The number of GPU's the training will utilize.
    * `--device_id`: The device id of the GPU to be used.
    * `--num_workers`: The numbe of workers to use in the dataloader.
    * `--batch_size`: The batch size.
    * `--lr_scale`: The learning rate scale, which multiplies the learning rate by this value.
    * `--block_num`: The block number is the number of MVFB blocks the architecrue will include.
    * `--train_dataset hic \`: Leave this argument unchanged.
    * `--val_datasets hic \`: Leave this argument unchanged.
    * `--data_val_path`: The path to the validation data.
    * `--data_train_path`: The path to the training dataset.
    * `--resume_epoch`: The epoch from which to resume training the model.
    * `--early_stoppage_epochs`: The number of epochs to wait before validation imporvement happens to terminate training with early stoppage.
    * `--early_stoppage_start`: Epoch from which to start applying early stoppage.
    * `--loss`: The loss function used in training. Choices include: `single_channel_L1_no_vgg`, `single_channel_default_VGG`, `single_channel_MSE_no_vgg`, `single_channel_MSE_VGG`, and `single_channel_L1_VGG`. 
    * `--no_cut_off`: I don't remember.
    * `--dynamics`: Indicates the presence of the routing module and dynamic aspect of the architecture. Switch the argument to `--no_dynamics` to turn off the routing module and dynamic aspect of the architecture.
    * `--no_max_cut_off`: I don't remember.
    * `--batch_max`: Normalization happens by dividing by the batch maximum. To turn off replace the argument with `--no_batch_max`.
    * `--code_test`: Indicates that the training process will run in test mode, cycling through only a few batches during each epoch, to quickly test the entire training pipeline. In test mode the model will save the logs in a separate test log folder. To turn off test mode and enable the regular training process, replace this argument with `--no_code_test`.
   An example, of the train.sh bash script is included in the scripts folder of the repository:
4. Run the bash script with the command `sh train.sh` to initiate training.


