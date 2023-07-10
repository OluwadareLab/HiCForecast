import sys
import numpy as np
import pickle
import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from ResConvLSTMNet import *
from utils import *

def test(model, device, test_loader, total_len, input_len, input_dim, imgSizes):
	model.eval()
	loss_sum = 0.0
	predictions = []
	c, h, w = imgSizes
	with torch.no_grad():
		mask_true_test = np.zeros((1, total_len-input_len-1, c, h, w))
		mask_true_test = torch.FloatTensor(mask_true_test).to(device)
		for i, X in enumerate(test_loader):
			output, loss = model(X.to(device), mask_true_test)
			#print(output.shape)
			predictions.append(output.cpu().detach().numpy())
			loss_sum = loss_sum + loss.item()

	return predictions, loss_sum/i

def main():
	parser = argparse.ArgumentParser(description='HiC4D testing process')
	parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	optional = parser.add_argument_group('optional arguments')
	required.add_argument('-f', '--file-test-data', type=str, metavar='FILE', required=True,
                        help='file name of the test data, npy format and shape=n1*L*40*40')
	required.add_argument('-o', '--file-output', type=str, metavar='FILE', required=True,
	                      help='path to output file')
	required.add_argument('-m', '--best-model', type=str, metavar='FILE', required=True,
                        help='path to the best model')
	required.add_argument('-il', '--input-length', type=int, metavar='N', required=True,
                        help='the input/known sequence length')
	required.add_argument('--max-HiC', type=int, metavar='N', required=True,
	                      help='the maximum Hi-C contact counts for scaling')
	optional.add_argument('-nl', '--num-layers', type=int, default=4, metavar='N',
	                      help='input batch size for training (default: 4)')
	optional.add_argument('-hd', '--hidden-dim', type=int, default=64, metavar='N',
	                      help='hidden dimension for network (default: 64)')
	optional.add_argument('-ps', '--patch-size', type=int, default=2, metavar='N',
	                      help='input patch size for training (default: 2)')
	optional.add_argument('-ks', '--kernel-size', type=int, default=3, metavar='N',
	                      help='kernel size for network (default: 3)')
	optional.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
	optional.add_argument('--GPU-index', type=int, default=0, metavar='N',
	                      help='GPU index (default: 0)')
	
	args = parser.parse_args()

	# load data
	dat_test = np.load(args.file_test_data).astype(np.float32) 

	maxHiC = args.max_HiC
	print("max_HiC: ", maxHiC)

	dat_test[dat_test > maxHiC] = maxHiC
	dat_test = dat_test / maxHiC

	dat_test = np.expand_dims(dat_test, axis=4)
	dat_test = patch_image(dat_test, args.patch_size)
	dat_test = np.transpose(dat_test, [0,1,4,2,3])

	print("Input data", flush=True)
	print("Test data: ", dat_test.shape, flush=True)

	test_loader = torch.utils.data.DataLoader(torch.from_numpy(dat_test), batch_size=1, shuffle=False)

	# check if CUDA is available
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:"+str(args.GPU_index) if use_cuda else "cpu")

	# info from data itself
	input_dim = dat_test.shape[2]
	img_size = dat_test.shape[3]
	imgSizes = (input_dim, img_size, img_size)
	channel = input_dim
	total_length = dat_test.shape[1]

	# load network
	model = RNN(args.num_layers, input_dim, args.hidden_dim, 
	            args.kernel_size, (img_size, img_size), channel, 
							device, (total_length, args.input_length), sampling=2)
	#stat_dict = torch.load(args.best_model, map_location='cpu')
	model.load_state_dict(torch.load(args.best_model, map_location='cpu'))
	model.to(device)
	print("Loading model. Done!")	
	
	frames, loss_test = test(model, device, test_loader,
		                   total_length, args.input_length, input_dim, imgSizes)
	frames2 = patch_image_back(np.transpose(np.concatenate(frames, axis=0), [0,1,3,4,2]), args.patch_size)
	frames2 = np.squeeze(frames2)
	print(loss_test, frames2.shape)
	# save predictions
	np.save(args.file_output, frames2)

if __name__ == '__main__':
	main()
