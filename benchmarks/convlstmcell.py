import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):

	def __init__(self, input_dim, hidden_dim, kernel_size, img_size, bias=True, hadamard=True, norm=True):

		super(ConvLSTMCell, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim

		self.kernel_size = kernel_size
		self.padding = kernel_size // 2
		self.bias = bias
		self.hadamard = hadamard

		if norm:
			self.conv = nn.Sequential(
										nn.Conv2d(self.input_dim + self.hidden_dim, self.hidden_dim * 4,
                            kernel_size=self.kernel_size, padding=self.padding, bias=self.bias),
										nn.GroupNorm(4, self.hidden_dim * 4))
		else:
			self.conv = nn.Conv2d(self.input_dim + self.hidden_dim, self.hidden_dim * 4,
			                      kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

		# weight matrices for Hadamard product
		if self.hadamard:
			self.Wci = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(hidden_dim, *img_size)))
			self.Wcf = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(hidden_dim, *img_size)))
			self.Wco = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(hidden_dim, *img_size)))
 

	def forward(self, x, h_pre, c_pre):

		conv_outputs = self.conv(torch.cat([x, h_pre], dim=1))
		conv_i, conv_f, conv_g, conv_o = torch.split(conv_outputs, self.hidden_dim, dim=1)
		
		if self.hadamard:
			i = torch.sigmoid(conv_i + self.Wci * c_pre)
			f = torch.sigmoid(conv_f + self.Wcf * c_pre)
		else:
			i = torch.sigmoid(conv_i)
			f = torch.sigmoid(conv_f)

		c = f * c_pre + i * torch.tanh(conv_g)
		if self.hadamard:
			o = torch.sigmoid(conv_o + self.Wco * c)
		else:
			o = torch.sigmoid(conv_o)

		h = o * torch.tanh(c)

		return h, c


