import os
import cv2
import sys
import torch
import random
import argparse
import numpy as np

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from utils.util import *
from model.model import Model

#from torchsummary import summary


device = torch.device("cuda")
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--image_0_path', required=True, type=str, help='image 0 path')
parser.add_argument('--image_1_path', required=True, type=str, help='image 1 path')
parser.add_argument('--load_path', required=True, type=str, help='model path')
parser.add_argument('--output_dir', default="pred.png", type=str, help='output path')

args = parser.parse_args()

def evaluate(model, args):

    with torch.no_grad():
        #dat_test = np.load(args.image_0_path).astype(np.float32)
        #dat_test = np.expand_dims(dat_test, axis=4)
        #image_0 = dat_test[600][1]
        #image_1 = dat_test[600][2]
        #img_0 = np.repeat(image_0[np.newaxis, :], 3, axis=0)
        #img_0 = img_0.transpose(1,2,0)
        
        #img_1 = np.repeat(image_1[np.newaxis, :], 3, axis=0)
        #img_1 = img_1.transpose(1,2,0)

        img_0 = cv2.imread(args.image_0_path)
        print("img_0.shape: ", img_0.shape)#me
        img_1 = cv2.imread(args.image_1_path)
        #print("img_1.shape: ", img_1.shape)#ME
        #print("dat_test.shape: ", dat_test.shape)

        if img_0 is None or img_1 is None:
            raise Exception("Images not found.")
        img_0 = img_0.transpose(2, 0, 1).astype('float32')
        img_1 = img_1.transpose(2, 0, 1).astype('float32')
        img = torch.cat([torch.tensor(img_0),torch.tensor(img_1)], dim=0)
        img = img.unsqueeze(0).unsqueeze(0).to(device, non_blocking=True) # NCHW
        img = img.to(device, non_blocking=True) / 255.

        pred = model.eval(img, 'single_test') # 1CHW
        pred = np.array(pred.cpu().squeeze() * 255).transpose(1, 2, 0) # CHW -> HWC
        print("thre channels: ", pred[100][104][0])
        print("#2: ", pred[100][104][1])
        print("#3: ", pred[100][104][2])
        quit()
        cv2.imwrite(args.output_dir, pred)
            
if __name__ == "__main__":    
    model = Model(load_path=args.load_path, training=False)
    #model = Model(training=True)
    #print(summary(model, (3, 250, 250))
    #net = DMVFN(num_feature=250).cuda()
    #x = torch.randn((2,6,250,250)).cuda()
    #y = net(x, scale=[4,4,4,2,2,2,1,1,1])
    #print("y.shape: ", y.shape)
    evaluate(model, args)
