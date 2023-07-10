import os
import cv2
import sys
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm

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
parser.add_argument('--data_path', required=True, type=str, help='data path')
#parser.add_argument('--image_1_path', required=True, type=str, help='image 1 path')
parser.add_argument('--load_path', required=True, type=str, help='model path')
parser.add_argument('--output_dir', default="pred.pny", type=str, help='output path')

args = parser.parse_args()

def evaluate(model, args):

    with torch.no_grad():
        dat_test = np.load(args.data_path).astype(np.float32)
        #dat_test = dat_test / 225.
        #print("dat_test.shape: ", dat_test.shape)
        #print("dat_test[:10,1:3].shape: ", dat_test[:10, 1:3].shape)
        test_loader = torch.utils.data.DataLoader(torch.from_numpy(dat_test[:,1:3]), batch_size=1, shuffle=False)
        print("dat_test.shape: ", dat_test.shape)
        
        '''
        image_0 = dat_test[600][1]
        image_1 = dat_test[600][2]
        img_0 = np.repeat(image_0[np.newaxis, :], 3, axis=0).astype('float32')
        print("img_0.shape: ", img_0.shape)
        
        img_1 = np.repeat(image_1[np.newaxis, :], 3, axis=0).astype('float32')

        print("img_0.shape: ", img_0.shape)#me
        print("img_1.shape: ", img_1.shape)#ME

        if img_0 is None or img_1 is None:
            raise Exception("Images not found.")

        img = torch.stack((torch.tensor(img_0), torch.tensor(img_1)), 0)
        print("img.shape: ", img.shape)
        img = img.unsqueeze(0).to(device, non_blocking=True) #BNCHW
        img = img.to(device, non_blocking=True) / 255.
        print("img.shape: ", img.shape)
        '''
        predictions = []
        for i, X in enumerate(tqdm(test_loader)):
            # X = X.unsqueeze(0).to(device, non_blocking=True)
            X = np.stack((X, X, X), axis=0)
            X = np.transpose(X, (1, 2, 0, 3, 4))
            #print("X.shape: ", X.shape)
            X = torch.from_numpy(X).to(device, non_blocking=True)
            X = X / 255.

            pred = model.eval(X, 'hic') # BNCHW
            #print("pred.shape: ", pred.shape)
            pred = np.array(pred.cpu() * 255)
            #print("pred.shape: ", pred.shape)
            #pred = torch.cat(pred)
            #print("pred.shape after concat: ", pred.shape)
            pred = pred[:,:,0,:,:]
            #print("pred.shape: ", pred.shape)
            predictions.append(pred)

        predictions = np.concatenate(predictions, axis=0)
        print("predictions.shape: ", predictions.shape)
        np.save(args.output_dir, predictions)
            
if __name__ == "__main__":    
    model = Model(load_path=args.load_path, training=False)
    evaluate(model, args)
