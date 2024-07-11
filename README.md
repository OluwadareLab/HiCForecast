
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


