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
print("Executing predict.py")

device = torch.device("cuda")
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

'''
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, type=str, help='data path')
#parser.add_argument('--image_1_path', required=True, type=str, help='image 1 path')
parser.add_argument('--load_path', required=True, type=str, help='model path')
parser.add_argument('--output_dir', type=str, help='output path')
parser.add_argument('--rgb', action="store_true")
parser.add_argument('--single_channel', dest='rgb', action='store_false')
parser.add_argument('--max_HiC', type=int)
parser.add_argument('--cut_off', action='store_true')
parser.add_argument('--no_cut_off', dest='cut_off', action='store_false')
args = parser.parse_args()

rgb = args.rgb
max_HiC = args.max_HiC
'''

def evaluate(model, args):

    with torch.no_grad():
        dat_test = np.load(args.data_path).astype(np.float32)
        if args.cut_off == True:
            dat_test[dat_test > max_HiC] = max_HiC
            print("Performed cut off for prediction")
        dat_test = dat_test / max_HiC
        #dat_test = dat_test / 225.
        #print("dat_test.shape: ", dat_test.shape)
        #print("dat_test[:10,1:3].shape: ", dat_test[:10, 1:3].shape)

        test_loader = torch.utils.data.DataLoader(dat_test[:,1:3], batch_size=1, shuffle=False)
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
            if rgb == True:
                #untested
                X = torch.stack((X, X, X), dim=0)
                X = torch.permute(X, (1, 2, 0, 3, 4))
            else:
            #print("X.shape: ", X.shape)
                X = torch.unsqueeze(X, 2)

            X = X.to(device, dtype=torch.float32, non_blocking=True)

            pred = model.eval(X, 'hic') # BNCHW
            #print("pred.shape: ", pred.shape)
            #pred = torch.cat(pred)
            #print("pred.shape after concat: ", pred.shape)
            if rgb == True:
                pred = pred[:,:,0,:,:]
            #print("pred.shape: ", pred.shape)
            pred = torch.squeeze(pred, dim=2)
            pred = np.array(pred.cpu() * max_HiC)
            predictions.append(pred)

        predictions = np.concatenate(predictions, axis=0)
        print("predictions.shape: ", predictions.shape)
        np.save(args.output_dir, predictions)
            
if __name__ == "__main__":    
    model = Model(load_path=args.load_path, training=False, rgb=rgb)
    evaluate(model, args)
