
import torch
import torch.nn as nn
import torch.nn.functional as F

from convlstmcell import *

# the RNN class modified based on https://github.com/thuml/predrnn-pytorch/blob/master/core/models/predrnn_v2.py

class Inception(nn.Module):
	def __init__(self, input_dim, hidden_dim, kernel_size, img_size, bias=False, norm=True, ns=2, last=False):        
		super(Inception, self).__init__()

		self.ns = ns
		self.last = last

		layers = []
		for i in range(ns):
			if i == 0:
				layers.append(ConvLSTMCell(input_dim, hidden_dim, kernel_size, img_size, bias=bias, norm=norm))
			else:
				layers.append(ConvLSTMCell(hidden_dim, hidden_dim, kernel_size, img_size, bias=bias, norm=norm))		

		self.layers = nn.Sequential(*layers)
		
		if last:
			self.conv_last = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0, bias=False)
	
	def forward(self, x, hs_pre, cs_pre):
		'''
		number of hs in hs_pre is equal to length(kernel_size)
		'''

		residual = x

		hs, cs = [], []
		for i, layer in enumerate(self.layers):
			if i == 0:
				h_i, c_i = layer(x, hs_pre[i], cs_pre[i])
			else:
				h_i, c_i = layer(hs[i-1], hs_pre[i], cs_pre[i])
			hs.append(h_i)
			cs.append(c_i)

		h_o = hs[self.ns-1] + residual
		
		if self.last:
			h_o = self.conv_last(h_o)
		
		return hs, cs, h_o


class RNN(nn.Module):

	def __init__(self, num_layers, input_dim, hidden_dim, kernel_size, img_size, channel, device, lengths, 
	                   sampling=2, norm=True, ns=2):
		super(RNN, self).__init__()

		self.num_layers = num_layers
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.kernel_size = kernel_size
		self.width, self.height = img_size
		self.channel = channel
		self.device = device
		self.total_len, self.input_len = lengths
		self.sampling = sampling
		self.ns = ns

		self.loss = nn.MSELoss()

		self.conv_1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)

		self.cells = nn.ModuleList()
		for i in range(num_layers):
			if i != num_layers-1:
				self.cells.append(Inception(hidden_dim, hidden_dim, kernel_size, img_size, norm=norm))
			else:
				self.cells.append(Inception(hidden_dim, hidden_dim, kernel_size, img_size, norm=norm, last=True))
		
		#self.conv_last = nn.Conv2d(hidden_dim, channel, kernel_size=1, stride=1, padding=0, bias=False)

	def forward(self, X, mask_true):
		# X: batch,length,channel,width,height

		batch, width, height = X.shape[0], X.shape[3], X.shape[4]

		next_frames = []
		hs, cs = [], []
		h_output = []

		for i in range(self.num_layers):
			zeros = torch.zeros([batch, self.hidden_dim, width, height]).to(self.device)
			zeros_inception = [zeros] * self.ns
			hs.append(zeros_inception)
			cs.append(zeros_inception)
			h_output.append(zeros)
		
		for t in range(self.total_len - 1):
			if self.sampling == 1:
				# reverse schedule sampling
				if t == 0:
					xt = X[:, t] # b*c*w*h
				else:
					xt = mask_true[:, t-1] * X[:, t] + (1 - mask_true[:, t-1]) * x_gen
			else:
				# scheduled sampling
				if t < self.input_len:
					xt = X[:, t]
				else:
					mask_true_t = torch.squeeze(mask_true[:, t-self.input_len], dim=1)
					xt = mask_true_t * X[:, t] + (1 - mask_true_t) * x_gen
					#xt = mask_true[:, t-self.input_len] * X[:, t] + (1 - mask_true[:, t-self.input_len]) * x_gen

			# the first conv layer
			xt2 = self.relu(self.conv_1(xt))

			# the first ConvLSTM layer
			hs[0], cs[0], h_output[0] = self.cells[0](xt2, hs[0], cs[0])

			# the other following layers if any
			for i in range(1, self.num_layers):
				hs[i], cs[i], h_output[i] = self.cells[i](h_output[i-1], hs[i], cs[i])

			#x_gen = self.conv_last(hs[self.num_layers - 1])
			x_gen = h_output[self.num_layers-1]
			next_frames.append(x_gen)

		next_frames = torch.stack(next_frames, dim=0).permute(1,0,2,3,4).contiguous()
		loss = self.loss(next_frames, X[:, 1:])

		return next_frames, loss

