!git clone https://github.com/megvii-research/CVPR2023-DMVFN.git
import os
os.chdir("/content/CVPR2023-DMVFN/")
!pip3 install -r requirements.txt
print("Cell 1 executed.")

#@title Download pretrained weights
!mkdir pretrained_models
import os
os.chdir("./pretrained_models/")
!gdown --id 1jILbS8Gm4E5Xx4tDCPZh_7rId0eo8r9W
!gdown --id 1WrV30prRiS4hWOQBnVPUxdaTlp9XxmVK
!gdown --id 14_xQ3Yl3mO89hr28hbcQW3h63lLrcYY0
os.chdir("../")
print("Cell 2 executed.")

#@title Download test dataset [Optional]
#@markdown We use the validation set of cityscapes for testing. If you want to inference your own image, please ignore this block.
!mkdir ./data/
!mkdir ./data/cityscapes
!mkdir ./data/cityscapes/test
os.chdir("./data/cityscapes/test/")
!gdown "10zCt-uZFOqgF3tpdhluRqbs-4aScvGR4&confirm=t"
!unzip -q test.zip
!rm -rf test.zip
os.chdir("/content/CVPR2023-DMVFN/")
print("Cell 3 executed.")

!python3 ./scripts/test.py --val_datasets CityValDataset --load_path ./pretrained_models/dmvfn_city.pkl
print("Testing complete.")
