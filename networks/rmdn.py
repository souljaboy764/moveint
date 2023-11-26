import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

# Recurrent Mixture Density Network
class RMDN(nn.Module):
	def __init__(self, input_dim:int, output_dim:int, args:argparse.ArgumentParser) -> None:
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		for key in args.__dict__:
			setattr(self, key, args.__dict__[key])
		self.activation = getattr(nn, args.__dict__['activation'])()
	
		enc_sizes = [self.input_dim] + self.hidden_sizes
		enc_layers = []
		for i in range(len(enc_sizes)-1):
			enc_layers.append(nn.Linear(enc_sizes[i], enc_sizes[i+1]))
			enc_layers.append(self.activation)
		self.human_encoder = nn.Sequential(*enc_layers)

		# head_sizes = [enc_sizes[-1]] + self.hidden_sizes + [args.num_components*self.output_dim]
		head_sizes = [enc_sizes[-1], args.num_components*self.output_dim]
		mean_layers = []
		logstd_layers = []
		for i in range(len(head_sizes)-1):
			mean_layers.append(nn.Linear(head_sizes[i], head_sizes[i+1]))
			mean_layers.append(self.activation)
			logstd_layers.append(nn.Linear(head_sizes[i], head_sizes[i+1]))
			logstd_layers.append(self.activation)
		self.human_mean = nn.Sequential(*mean_layers)
		self.human_logstd = nn.Sequential(*logstd_layers)
		
		self.human_rnn = nn.GRU(input_size=enc_sizes[-1], hidden_size=head_sizes[-1], num_layers=len(head_sizes)-1, batch_first=True)
		self.segment_logits = nn.Sequential(nn.Linear(head_sizes[-1], args.num_components), nn.Softmax(-1))

	def forward(self, x_in:torch.Tensor, x_out:torch.Tensor=None) -> (torch.Tensor,torch.Tensor,torch.Tensor):
		h_enc = self.human_encoder(x_in)
		h_mean = self.human_mean(h_enc).reshape((-1, self.num_components, self.output_dim))
		h_std = self.human_logstd(h_enc).reshape((-1, self.num_components, self.output_dim)).exp() + self.std_reg
		
		h_rnn, _ = self.human_rnn(h_enc)
		h_alpha = self.segment_logits(h_rnn)

		return h_mean, h_std, h_alpha
	

	def forward_step(self, x_in:torch.Tensor, hidden:torch.Tensor=None) -> (torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor):
		h_enc = self.human_encoder(x_in)
		h_mean = self.human_mean(h_enc).reshape((-1, self.num_components, self.output_dim))
		h_std = self.human_logstd(h_enc).reshape((-1, self.num_components, self.output_dim)).exp() + self.std_reg
		
		h_rnn, hidden = self.human_rnn(h_enc, hidden)
		h_alpha = self.segment_logits(h_rnn)
		return h_mean, h_std, h_alpha, hidden

	def run_iteration(self, iterator:DataLoader, optimizer:torch.optim.Optimizer, args:argparse.ArgumentParser, epoch:int):
		mse_loss, bce_loss, ae_loss, kl_loss = [], [], [], []
		
		for it, (x_in, x_out) in enumerate(iterator):
			if self.training:
				optimizer.zero_grad()

			x_in = torch.Tensor(x_in[0]).to(device)
			x_out = torch.Tensor(x_out[0]).to(device)
			h_mean, h_std, h_alpha = self(x_in, x_out)
			# Combined
			if self.training:
				h_mean_combined = (h_mean*h_alpha[..., None]).sum(1)
				h_std_combined = (h_std*h_alpha[..., None]).sum(1)
				nll = -Normal(h_mean_combined, h_std_combined).log_prob(x_out).sum(-1)
			else:
				nll = (((h_mean*h_alpha[..., None]).sum(1) - x_out)**2).sum(-1)
			
			# # GMM
			# if self.training:
			# 	nll = -(Normal(h_mean, h_std).log_prob(x_out[:,None,:]).exp()*h_alpha[..., None]).sum(1).log().sum(-1)
			# else:
			# 	h_alpha_sample = gumbel_rao(h_alpha, k=100, temp=0.01)
			# 	h_mean_combined = (h_mean*h_alpha_sample[..., None]).sum(1)
			# 	nll = ((h_mean_combined - x_out)**2).sum(-1)

			# 	# h_alpha_sample = gumbel_rao(h_alpha, k=100, temp=0.01)
			# 	# h_mean_combined = (h_mean*h_alpha_sample[..., None]).sum(1)
			# 	# h_std_combined = (h_std*h_alpha_sample[..., None]).sum(1)
			# 	# x_out_gen = Normal(h_mean_combined, h_std_combined).rsample((15,)).mean(0)
			# 	# nll = ((x_out_gen - x_out)**2).sum(-1)
			
			if self.num_components>1:
				interclass_dist = []
				for i in range(self.num_components-1):
					for j in range(i+1, self.num_components):
						interclass_dist.append(torch.exp(-((h_mean[:, i] - h_mean[:, j])**2).sum(-1))[None])

				intraclass_dist = torch.exp(-((torch.diff(h_mean, dim=0, prepend=h_mean[0:1]))**2).sum(-2)).mean(-1)
				kld = (h_alpha * torch.log(h_alpha)).sum(-1) + intraclass_dist - torch.vstack(interclass_dist).sum(0)
			else:
				kld = torch.zeros_like(nll)
			if it==0:
				mse_loss = nll
				kl_loss = kld
			else:
				mse_loss = torch.cat([mse_loss, nll])
				kl_loss = torch.cat([kl_loss, kld])
				
			loss = nll + self.beta*kld
			loss = loss.mean()
			if self.training:
				loss.backward()
				optimizer.step()
		return {'pred_mse':mse_loss, 'kl_loss':kl_loss}
