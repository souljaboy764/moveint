import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils import *

class RMDVAE(nn.Module):
	def __init__(self, input_dim:int, output_dim:int, args:argparse.ArgumentParser) -> None:
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		for key in args.__dict__:
			setattr(self, key, args.__dict__[key])
		self.activation = getattr(nn, args.__dict__['activation'])()
	
		enc_sizes = [self.input_dim] + self.hidden_sizes + [args.latent_dim]
		enc_layers = []
		for i in range(len(enc_sizes)-1):
			enc_layers.append(nn.Linear(enc_sizes[i], enc_sizes[i+1]))
			enc_layers.append(self.activation)
		self.human_encoder = nn.Sequential(*enc_layers)
		self.human_mean_w = nn.Parameter(torch.Tensor(args.latent_dim, 3))
		self.human_mean_b = nn.Parameter(torch.Tensor(args.latent_dim, 3))
		self.human_std = nn.Parameter(torch.Tensor(1, args.latent_dim, 3)) 

		self.human_rnn = nn.GRU(input_size=enc_sizes[-1], hidden_size=enc_sizes[-1], num_layers=1, batch_first=True)
		self.segment_logits = nn.Linear(enc_sizes[-1], 3) # reach, transfer, retreat

		enc_sizes = [self.output_dim] + self.hidden_sizes
		enc_layers = []
		for i in range(len(enc_sizes)-1):
			enc_layers.append(nn.Linear(enc_sizes[i], enc_sizes[i+1]))
			enc_layers.append(self.activation)
		self.robot_encoder = nn.Sequential(*enc_layers)
		self.robot_mean = nn.Sequential(nn.Linear(self.hidden_sizes[-1], args.latent_dim), self.activation)
		self.robot_std = nn.Linear(self.hidden_sizes[-1], args.latent_dim)

		dec_sizes = [args.latent_dim] + enc_sizes[::-1]
		dec_layers = []
		for i in range(len(dec_sizes)-1):
			dec_layers.append(nn.Linear(dec_sizes[i], dec_sizes[i+1]))
			dec_layers.append(self.activation)
		self.robot_decoder = nn.Sequential(*dec_layers)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.ones_(self.human_std)
		nn.init.kaiming_uniform_(self.human_mean_w)
		nn.init.kaiming_uniform_(self.human_mean_b)
		

	def forward(self, x_in:torch.Tensor, x_out:torch.Tensor=None) -> (torch.Tensor,torch.Tensor,torch.Tensor):
		h_enc = self.human_encoder(x_in)
		h_mean = self.human_mean_w[None] * h_enc[...,None] + self.human_mean_b
		
		h_rnn, _ = self.human_rnn(h_enc)
		h_alpha = self.segment_logits(h_rnn)

		if x_out is not None:
			r_enc = self.robot_encoder(x_out)
			r_mean = self.robot_mean(r_enc)
			r_std = self.robot_std(r_enc).exp() + self.std_reg

			if self.training:
				r_samples_r = r_mean[None] + torch.randn((self.mce_samples,)+r_std.shape, device=x_in.device)*r_std[None]
			else:
				r_samples_r = r_mean

			r_out_r = self.robot_decoder(r_samples_r)
		else:
			r_mean = None
			r_std = None
			r_out_r = None

		# h_alpha_sample = torch.distributions.categorical.Categorical(probs=alpha).sample()
		h_alpha_sample = gumbel_rao(h_alpha, k=100,temp=0.01)
		if self.training:
			eps = torch.randn((self.mce_samples,)+h_mean.shape[:-1], device=x_in.device)
			r_samples_h = (h_mean*h_alpha_sample[:, None]).sum(-1) + eps * (self.human_std*h_alpha_sample[:, None]).sum(-1)
		else:
			r_samples_h = (h_mean*h_alpha_sample[:, None]).sum(-1)
		r_out_h = self.robot_decoder(r_samples_h)

		return h_mean, h_alpha, r_mean, r_std, r_out_r, r_out_h
	
	# def forward_step(self:BaseNet, x:torch.Tensor, hidden:torch.Tensor=None) -> (torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor):
	# 	enc, hidden = self._encoder(hidden)
	# 	return self.policy(enc), self.policy_std(enc).exp() + self.std_reg, self.segment_logits(enc), hidden
