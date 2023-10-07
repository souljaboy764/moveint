import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pbdlib_torch as pbd_torch
import pbdlib as pbd

from rmdn_hri.utils import *

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
		self.human_logstd = nn.Parameter(torch.Tensor(1, args.latent_dim, 3)) 

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
		nn.init.zeros_(self.human_logstd)
		nn.init.kaiming_uniform_(self.human_mean_w)
		nn.init.kaiming_uniform_(self.human_mean_b)
		

	def forward(self, x_in:torch.Tensor, x_out:torch.Tensor=None) -> (torch.Tensor,torch.Tensor,torch.Tensor):
		h_enc = self.human_encoder(x_in)
		h_mean = self.human_mean_w[None] * h_enc[...,None] + self.human_mean_b
		h_std = self.human_logstd.exp() + self.std_reg
		
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
		h_alpha_sample = gumbel_rao(h_alpha, k=100, temp=0.01)
		if self.training:
			eps = torch.randn((self.mce_samples,)+h_mean.shape[:-1], device=x_in.device)
			r_samples_h = (h_mean*h_alpha_sample[:, None]).sum(-1) + eps * (h_std*h_alpha_sample[:, None]).sum(-1)
		else:
			r_samples_h = (h_mean*h_alpha_sample[:, None]).sum(-1)
		r_out_h = self.robot_decoder(r_samples_h)

		return h_mean, h_alpha, r_mean, r_std, r_out_r, r_out_h
	
	# def forward_step(self:BaseNet, x:torch.Tensor, hidden:torch.Tensor=None) -> (torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor):
	# 	enc, hidden = self._encoder(hidden)
	# 	return self.policy(enc), self.policy_std(enc).exp() + self.std_reg, self.segment_logits(enc), hidden

	def run_iteration(self, iterator:DataLoader, optimizer:torch.optim.Optimizer, args:argparse.ArgumentParser, epoch:int):
		mse_loss, bce_loss, ae_loss, kl_loss = [], [], [], []
		
		for i, (x_in, x_out, label) in enumerate(iterator):
			if self.training:
				optimizer.zero_grad()

			label_dist = np.array([np.sum(label[0].numpy()==0), np.sum(label[0].numpy()==1), np.sum(label[0].numpy()==2)])
			w = 1/label_dist
			w /= w.sum()
			w = torch.Tensor(w).to(device)
			x_in = torch.Tensor(x_in[0]).to(device)
			x_out = torch.Tensor(x_out[0]).to(device)
			label_onehot = torch.eye(3, device=device)[label[0]]
			label = torch.Tensor(label[0]).to(device)
			h_mean, h_alpha, r_mean, r_std, r_out_r, r_out_h = self(x_in, x_out)
			h_std = self.human_logstd.exp() + self.std_reg

			
			kld = kl_divergence(Normal(r_mean, r_std), Normal((h_mean*label_onehot[:, None]).sum(-1), (h_std*label_onehot[:, None]).sum(-1))).mean(-1)
			
			ae_recon = ((r_out_r - x_out)**2).sum(-1)
			pred_mse = ((r_out_h - x_out)**2).sum(-1)
			if self.training:
				ae_recon = ae_recon.mean(0)
				pred_mse = pred_mse.mean(0)

			interclass_dist = (
								torch.exp(-((h_mean[..., 0] - h_mean[..., 1])**2).sum(-1)) + \
								torch.exp(-((h_mean[..., 2] - h_mean[..., 1])**2).sum(-1)) + \
								torch.exp(-((h_mean[..., 0] - h_mean[..., 2])**2).sum(-1))
			)/3

			intraclass_dist = torch.exp(-((torch.diff(h_mean, dim=0, prepend=h_mean[0:1]))**2).sum(-2)).mean(-1)
			
			kld += intraclass_dist - interclass_dist
			if self.training:
				bce = F.binary_cross_entropy_with_logits(h_alpha, label_onehot, weight=w, reduction='none').sum(-1)
			else:
				bce = (h_alpha.argmax(1) == label).to(float)

			if i==0:
				mse_loss = pred_mse
				ae_loss = ae_recon
				kl_loss = kld
				bce_loss = bce
			else:
				mse_loss = torch.cat([mse_loss, pred_mse])
				ae_loss = torch.cat([ae_loss, ae_recon])
				kl_loss = torch.cat([kl_loss, kld])
				bce_loss = torch.cat([bce_loss, bce])

			loss = ae_recon + pred_mse + bce + args.beta*kld
			loss = loss.mean()
			if self.training:
				loss.backward()
				optimizer.step()
		if self.training:
			return {'pred_mse':mse_loss, 'alpha_bce':bce_loss, 'ae_loss':ae_loss, 'kl_loss':kl_loss}
		else:
			return {'pred_mse':mse_loss, 'accuracy':bce_loss}

class MILDVAE(nn.Module):
	def __init__(self, input_dim:int, args:argparse.ArgumentParser) -> None:
		super().__init__()
		self.input_dim = input_dim
		for key in args.__dict__:
			setattr(self, key, args.__dict__[key])
		self.activation = getattr(nn, args.__dict__['activation'])()
	
		self.enc_sizes = [self.input_dim] + self.hidden_sizes
		enc_layers = []
		for i in range(len(self.enc_sizes)-1):
			enc_layers.append(nn.Linear(self.enc_sizes[i], self.enc_sizes[i+1]))
			enc_layers.append(self.activation)
		self._encoder = nn.Sequential(*enc_layers)

		self.post_mean = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		self.post_logstd = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		
		self.dec_sizes = [self.latent_dim] + self.hidden_sizes[::-1]
		dec_layers = []
		for i in range(len(self.dec_sizes)-1):
			dec_layers.append(nn.Linear(self.dec_sizes[i], self.dec_sizes[i+1]))
			dec_layers.append(self.activation)
		self._decoder = nn.Sequential(*dec_layers)
		self._output = nn.Linear(self.dec_sizes[-1], self.input_dim)

	def forward(self, x, encode_only = False, dist_only=False):
		enc = self._encoder(x)
		z_mean = self.post_mean(enc)
		if encode_only:
			return z_mean
		z_std = self.post_logstd(enc).exp() + 1e-8
		if dist_only:
			return MultivariateNormal(z_mean, scale_tril=torch.diag_embed(z_std))
			
		if self.training:
			eps = torch.randn((self.mce_samples,)+z_mean.shape, device=z_mean.device)
			zpost_samples = z_mean + eps*z_std
			zpost_samples = torch.concat([zpost_samples, z_mean[None]], dim=0)
		else:
			zpost_samples = z_mean
		
		x_gen = self._output(self._decoder(zpost_samples))
		# return x_gen, zpost_samples, z_mean, torch.diag_embed(z_std**2)
		return x_gen, zpost_samples, MultivariateNormal(z_mean, torch.diag_embed(z_std**2))
	
class MILD(nn.Module):
	def __init__(self, input_dim:int, output_dim:int, args:argparse.ArgumentParser) -> None:
		super().__init__()
		self.model_h = MILDVAE(input_dim, args)
		self.model_r = MILDVAE(output_dim, args)
		nb_dim=2*args.latent_dim
		nb_states=3
		self.hmm = pbd_torch.HMM(nb_dim=nb_dim, nb_states=nb_states)
		self.hmm.mu = torch.zeros((nb_states, nb_dim), device=device)
		self.hmm.sigma = torch.eye(nb_dim, device=device)[None].repeat(nb_states,1,1)
		self.hmm.init_priors = torch.ones((nb_states,), device=device)/nb_states
		self.hmm.Trans = torch.ones((nb_states, nb_states), device=device)/nb_states
		
	def run_iteration(self, iterator:DataLoader, optimizer:torch.optim.Optimizer, args:argparse.ArgumentParser, epoch:int):
		mse_loss, kl_loss = [], []
		
		for i, (x_in, x_out, label) in enumerate(iterator):
			if self.training:
				optimizer.zero_grad()
			
			x_in = torch.Tensor(x_in[0]).to(device)
			x_out = torch.Tensor(x_out[0]).to(device)
			label = torch.Tensor(label[0]).to(device)
			xh_gen, zh_samples, zh_post = self.model_h(x_in)
			xr_gen, zr_samples, zr_post = self.model_h(x_out)
			
			if self.training:
				recon_loss = ((xh_gen - x_in)**2).sum(-1) + ((xr_gen - x_out)**2).sum(-1)

				if epoch!=0:
					with torch.no_grad():
						zh_prior = torch.distributions.MultivariateNormal(self.hmm.mu[label, :self.model_h.latent_dim], covariance_matrix=self.hmm.sigma[label, :self.model_h.latent_dim, :self.model_h.latent_dim])
						zr_prior = torch.distributions.MultivariateNormal(self.hmm.mu[label, self.model_h.latent_dim:], covariance_matrix=self.hmm.sigma[label, self.model_h.latent_dim:, self.model_h.latent_dim:])
					kld = kl_divergence(zh_post, zh_prior) + kl_divergence(zr_post, zr_prior)
					loss = recon_loss + args.beta*kld
				else:
					loss = recon_loss
				
				loss.backward()
				optimizer.step()
			else:
				alpha = self.hmm.forward_variable(zh_post.mean, marginal=slice(0, self.model_h.latent_dim))
				zr_cond = self.hmm.condition(zh_post.mean, dim_in=slice(0, self.model_h.latent_dim), dim_out=slice(self.model_h.latent_dim, 2*self.model_h.latent_dim),
												h=alpha, return_cov=False, data_Sigma_in=None)
				xr_cond = self.model_r._output(self.model_r._decoder(zr_cond))
				recon_loss = ((xr_cond - x_out)**2).sum(-1)
				bce = (alpha.argmax(0) == label).to(float)

			if i==0:
				mse_loss = recon_loss
				if self.training:
					kl_loss = kld
				else:
					kl_loss = bce
			else:
				mse_loss = torch.cat([mse_loss, recon_loss])
				if self.training:
					kl_loss = torch.cat([kl_loss, kld])
				else:
					kl_loss = torch.cat([kl_loss, bce])

		if self.training:
			return {'pred_mse':mse_loss, 'kl_loss':kl_loss}
		else:
			return {'pred_mse':mse_loss, 'accuracy':kl_loss}
