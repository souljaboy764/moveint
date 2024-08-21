from typing import Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

def kl_divergence(mu_p, std_p, mu_q, std_q):
    var_ratio = (std_p / std_q).pow(2)
    t1 = ((mu_p - mu_q) / std_q).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

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

	def forward(self, x_in:torch.Tensor, x_out:torch.Tensor=None) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
		h_enc = self.human_encoder(x_in)
		h_mean = self.human_mean(h_enc).reshape((-1, self.num_components, self.output_dim))
		h_std = self.human_logstd(h_enc).reshape((-1, self.num_components, self.output_dim)).exp() + self.std_reg
		
		h_rnn, _ = self.human_rnn(h_enc)
		h_alpha = self.segment_logits(h_rnn)

		return h_mean, h_std, h_alpha
	
	def forward_step(self, x_in:torch.Tensor, hidden:torch.Tensor=None) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
		h_enc = self.human_encoder(x_in)
		h_mean = self.human_mean(h_enc).reshape((-1, self.num_components, self.output_dim))
		h_std = self.human_logstd(h_enc).reshape((-1, self.num_components, self.output_dim)).exp() + self.std_reg
		
		h_rnn, hidden = self.human_rnn(h_enc, hidden)
		h_alpha = self.segment_logits(h_rnn)
		return h_mean, h_std, h_alpha, hidden


# HRI VAE with a Recurrent Mixture Density Network as the Human Encoder and a standard VAE for the robot
class MoVEInt(nn.Module):
	def __init__(self, input_dim:int, output_dim:int, args:argparse.ArgumentParser) -> None:
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		for key in args.__dict__:
			setattr(self, key, args.__dict__[key])
		self.activation = getattr(nn, args.__dict__['activation'])()
	
		self.human_encoder = RMDN(input_dim, self.latent_dim, args)

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

	def forward(self, x_in:torch.Tensor, x_out:torch.Tensor=None) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
		h_mean, h_std, h_alpha = self.human_encoder(x_in)
		h_mean_combined = (h_mean*h_alpha[..., None]).sum(1)
		h_std_combined = (h_std.pow(2)*h_alpha[..., None]).sum(1).sqrt()

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

		# h_alpha_sample = gumbel_rao(h_alpha, k=100, temp=0.01)
		# h_mean_combined = (h_mean*h_alpha_sample[..., None]).sum(1)
		# h_std_combined = (h_std*h_alpha_sample[..., None]).sum(1)
		
		if self.training:
			eps = torch.randn((self.mce_samples,)+h_mean_combined.shape, device=x_in.device)
			r_samples_h = h_mean_combined + eps * h_std_combined
		else:
			r_samples_h = h_mean_combined
			# eps = torch.randn((15,)+h_mean_combined.shape, device=x_in.device)
			# r_samples_h = (h_mean_combined + eps * h_std_combined).mean(0)
		r_out_h = self.robot_decoder(r_samples_h)

		return h_mean, h_std, h_alpha, h_mean_combined, h_std_combined, r_mean, r_std, r_out_r, r_out_h, r_samples_h, r_samples_r
	
	def forward_step(self, x_in:torch.Tensor, hidden:torch.Tensor=None) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
		h_mean, h_std, h_alpha, hidden = self.human_encoder.forward_step(x_in, hidden)
		h_mean_combined = (h_mean*h_alpha[..., None]).sum(1)
		h_std_combined = (h_std*h_alpha[..., None]).sum(1)

		if self.training:
			eps = torch.randn((self.mce_samples,)+h_mean_combined.shape, device=x_in.device)
			r_samples_h = h_mean_combined + eps * h_std_combined
		else:
			r_samples_h = h_mean_combined

		r_out_h = self.robot_decoder(r_samples_h)
		r_out_components = self.robot_decoder(h_mean)

		return h_mean, h_std, h_alpha, r_out_h, r_out_components, hidden

	def run_iteration(self, iterator:DataLoader, optimizer:torch.optim.Optimizer, args:argparse.ArgumentParser, epoch:int):
		mse_loss, ae_loss, kl_loss = [], [], []
		
		for it, (x_in, x_out) in enumerate(iterator):
			if self.training:
				optimizer.zero_grad()

			x_in = torch.Tensor(x_in[0]).to(device)
			x_out = torch.Tensor(x_out[0]).to(device)
			h_mean, h_std, h_alpha, h_mean_combined, h_std_combined, r_mean, r_std, r_out_r, r_out_h, r_samples_h, r_samples_r = self(x_in, x_out)
			
			kld = kl_divergence(r_mean, r_std, h_mean_combined, h_std_combined).mean(-1)
			
			ae_recon = ((r_out_r - x_out)**2).sum(-1)
			pred_mse = ((r_out_h - x_out)**2).sum(-1)
			if self.training:
				ae_recon = ae_recon.mean(0)
				pred_mse = pred_mse.mean(0)

			if self.num_components>1:
				interclass_dist = []
				for i in range(self.num_components-1):
					for j in range(i+1, self.num_components):
						interclass_dist.append(torch.exp(-((h_mean[:, i] - h_mean[:, j])**2).sum(-1))[None])

				intraclass_dist = 1-torch.exp(-((torch.diff(h_mean, dim=0, prepend=h_mean[0:1]))**2).sum(-2)).mean(-1)
				kld += (h_alpha * torch.log(h_alpha)).sum(-1) 
				kld += intraclass_dist 
				kld += torch.vstack(interclass_dist).sum(0)
			else:
				kld += torch.zeros_like(ae_recon)

			if it==0:
				mse_loss = pred_mse
				ae_loss = ae_recon
				kl_loss = kld
			else:
				mse_loss = torch.cat([mse_loss, pred_mse])
				ae_loss = torch.cat([ae_loss, ae_recon])
				kl_loss = torch.cat([kl_loss, kld])

			loss = ae_recon + pred_mse + args.beta*kld
			loss = loss.mean()
			if self.training:
				loss.backward()
				optimizer.step()
		if self.training:
			return {'pred_mse':mse_loss, 'ae_loss':ae_loss, 'kl_loss':kl_loss}
		else:
			return {'pred_mse':mse_loss, 'kl_loss':kl_loss}