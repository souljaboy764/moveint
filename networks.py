import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from utils import *

class BaseNet(nn.Module):
	def __init__(self:nn.Module, input_dim:int, output_dim:int, args:argparse.ArgumentParser) -> None:
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		for key in args.__dict__:
			setattr(self, key, args.__dict__[key])
		self.activation = getattr(nn, args.__dict__['activation'])()
	
	def forward(self, x):
		raise NotImplementedError
	
class FeedFwdNet(BaseNet):
	def __init__(self:BaseNet, input_dim:int, output_dim:int, args:argparse.ArgumentParser) -> None:
		super().__init__(input_dim, output_dim, args)

		enc_sizes = [self.input_dim] + self.hidden_sizes
		enc_layers = []
		for i in range(len(enc_sizes)-1):
			enc_layers.append(nn.Linear(enc_sizes[i], enc_sizes[i+1]))
			enc_layers.append(self.activation)
		self._encoder = nn.Sequential(*enc_layers)

		self.policy = nn.Linear(enc_sizes[-1], self.output_dim)
		self.segment_logits = nn.Linear(enc_sizes[-1], 3)# reach, transfer, retreat

	def forward(self:BaseNet, x:torch.Tensor) -> (torch.Tensor,torch.Tensor,torch.Tensor):
		enc = self._encoder(x)
		return self.policy(enc), None, self.segment_logits(enc)

class MixtureDensityNet(BaseNet):
	def __init__(self:BaseNet, input_dim:int, output_dim:int, args:argparse.ArgumentParser) -> None:
		super().__init__(input_dim, output_dim, args)

		enc_sizes = [self.input_dim] + self.hidden_sizes
		enc_layers = []
		for i in range(len(enc_sizes)-1):
			enc_layers.append(nn.Linear(enc_sizes[i], enc_sizes[i+1]))
			enc_layers.append(self.activation)
		self._encoder = nn.Sequential(*enc_layers)

		self.policy = nn.Linear(enc_sizes[-1], args.num_components*self.output_dim)
		self.policy_std = nn.Linear(enc_sizes[-1], args.num_components*self.output_dim)
		self.segment_logits = nn.Linear(enc_sizes[-1], 3) # reach, transfer, retreat
					
	def forward(self:BaseNet, x:torch.Tensor) -> (torch.Tensor,torch.Tensor,torch.Tensor):
		enc = self._encoder(x)
		return self.policy(enc), self.policy_std(enc).exp() + self.std_reg, self.segment_logits(enc)

class GRUNet(BaseNet):
	def __init__(self:BaseNet, input_dim:int, output_dim:int, args:argparse.ArgumentParser) -> None:
		super().__init__(input_dim, output_dim, args)

		self._encoder = nn.GRU(input_size=input_dim, hidden_size=self.hidden_sizes[0], num_layers=len(self.hidden_sizes), batch_first=True)
		
		self.policy = nn.Linear(self.hidden_sizes[0], self.output_dim)
		self.segment_logits = nn.Linear(self.hidden_sizes[0], 3) # reach, transfer, retreat

	def forward(self:BaseNet, x:torch.Tensor) -> (torch.Tensor,torch.Tensor):
		enc,_ = self._encoder(x)
		return self.policy(enc), None, self.segment_logits(enc)
	
	def forward_step(self:BaseNet, x:torch.Tensor, hidden:torch.Tensor=None) -> (torch.Tensor,torch.Tensor,torch.Tensor):
		enc, hidden = self._encoder(hidden)
		return self.policy(enc), self.segment_logits(enc), hidden

class GRUMixtureDensityNet(BaseNet):
	def __init__(self:BaseNet, input_dim:int, output_dim:int, args:argparse.ArgumentParser) -> None:
		super().__init__(input_dim, output_dim, args)

		self._encoder = nn.GRU(input_size=input_dim, hidden_size=self.hidden_sizes[0], num_layers=len(self.hidden_sizes), batch_first=True)
		
		self.policy = nn.Linear(self.hidden_sizes[0], args.num_components*self.output_dim)
		self.policy_std = nn.Linear(self.hidden_sizes[0], args.num_components*self.output_dim)
		self.segment_logits = nn.Linear(self.hidden_sizes[0], 3) # reach, transfer, retreat

	def forward(self:BaseNet, x:torch.Tensor) -> (torch.Tensor,torch.Tensor,torch.Tensor):
		enc,_ = self._encoder(x)
		return self.policy(enc), self.policy_std(enc).exp() + self.std_reg, self.segment_logits(enc)
	
	def forward_step(self:BaseNet, x:torch.Tensor, hidden:torch.Tensor=None) -> (torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor):
		enc, hidden = self._encoder(hidden)
		return self.policy(enc), self.policy_std(enc).exp() + self.std_reg, self.segment_logits(enc), hidden

class RMDVAE(BaseNet):
	def __init__(self:BaseNet, input_dim:int, output_dim:int, args:argparse.ArgumentParser) -> None:
		super().__init__(input_dim, output_dim, args)

		enc_sizes = [self.input_dim] + self.hidden_sizes
		enc_layers = []
		for i in range(len(enc_sizes)-1):
			enc_layers.append(nn.Linear(enc_sizes[i], enc_sizes[i+1]))
			enc_layers.append(self.activation)
		self.human_encoder = nn.Sequential(*enc_layers)
		self.human_mean = nn.Sequential(nn.Linear(self.hidden_sizes[-1], 3*args.latent_dim), self.activation)
		self.human_std = nn.Linear(self.hidden_sizes[-1], 3*args.latent_dim)
		self.human_rnn = nn.GRU(input_size=self.hidden_sizes[-1], hidden_size=self.hidden_sizes[-1], num_layers=1, batch_first=True)
		self.segment_logits = nn.Linear(self.hidden_sizes[-1], 3) # reach, transfer, retreat
		
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

	def forward(self:BaseNet, x_in:torch.Tensor, x_out:torch.Tensor=None) -> (torch.Tensor,torch.Tensor,torch.Tensor):
		h_enc = self.human_encoder(x_in)
		h_mean = self.human_mean(h_enc).reshape(-1, self.latent_dim, 3)
		h_std = (self.human_std(h_enc).exp() + self.std_reg).reshape(-1, self.latent_dim, 3)
		
		h_rnn, _ = self.human_rnn(h_enc)
		h_alpha = self.segment_logits(h_rnn)

		if x_out is not None:
			r_enc = self.robot_encoder(x_out)
			r_mean = self.robot_mean(r_enc)
			r_std = self.robot_std(r_enc).exp() + self.std_reg

			if self.training:
				r_samples_r = r_mean[None] + torch.randn((self.mce_samples,)+r_std.shape, device=r_std.device)*r_std[None]
			else:
				r_samples_r = r_mean

			r_out_r = self.robot_decoder(r_samples_r)
		else:
			r_mean = None
			r_std = None
			r_out_r = None

		# idx = torch.distributions.categorical.Categorical(probs=alpha).sample()
		idx = gumbel_rao(h_alpha, k=100,temp=0.01)
		if self.training:
			eps = torch.randn((self.mce_samples,)+h_mean.shape[:-1], device=h_mean.device)
			r_samples_h = (h_mean*idx[:, None]).sum(-1) + eps * (h_std*idx[:, None]).sum(-1)
		else:
			r_samples_h = (h_mean*idx[:, None]).sum(-1)
		r_out_h = self.robot_decoder(r_samples_h)

		return h_mean, h_std, h_alpha, r_mean, r_std, r_out_r, r_out_h
	
	# def forward_step(self:BaseNet, x:torch.Tensor, hidden:torch.Tensor=None) -> (torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor):
	# 	enc, hidden = self._encoder(hidden)
	# 	return self.policy(enc), self.policy_std(enc).exp() + self.std_reg, self.segment_logits(enc), hidden
