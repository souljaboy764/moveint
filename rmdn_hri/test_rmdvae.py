import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import sklearn

import os

from rmdn_hri.utils import *
import rmdn_hri.dataset
import rmdn_hri.networks

parser = argparse.ArgumentParser(description='RMDVAE Testing')
parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
						help='Checkpoint to evaluate')

args = training_argparse()
ckpt = torch.load(args.ckpt)
training_args = ckpt['args']

test_iterator = DataLoader(getattr(rmdn_hri.dataset,training_args.dataset)(train=False), batch_size=1, shuffle=False)
model = getattr(rmdn_hri.networks,args.model)(test_iterator.dataset.input_dims, test_iterator.dataset.output_dims, training_args).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
mse_loss = []#[] for i in test_iterator.dataset.actidx]
# if True:
for idx in test_iterator.dataset.dataset.actidx:
	alpha_dist = []
	mse_loss.append([])
	# for i in range(len(test_iterator.dataset)):
	for i in range(idx[0],idx[1]):
		x_in, x_out = test_iterator.dataset.__getitem__(i)
		x_in = torch.Tensor(x_in).to(device)
		x_out = torch.Tensor(x_out).to(device)
		with torch.no_grad():
			h_mean, h_std, h_alpha, h_mean_combined, h_std_combined, r_mean, r_std, r_out_r, r_out_h, r_samples_h, r_samples_r = model(x_in, x_out)

		h_alpha = F.one_hot(h_alpha.argmax(1), num_classes=training_args.num_components).sum(0)
		alpha_dist.append(h_alpha)
		
		# BP/NuiSI HH
		pred_mse = ((r_out_h - x_out)**2).reshape((x_out.shape[0], 5, 6, 3)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy()

		# # # Handover HH
		# pred_mse = ((r_out_h - x_out)**2).reshape((x_out.shape[0], 5, 12, 3)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy()
		
		# # BP/NuiSI Pepper
		# pred_mse = ((r_out_h - x_out)**2).reshape((x_out.shape[0], 5, 4, 1)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy()
		
		# # # BP/NuiSI Yumi
		# pred_mse = ((r_out_h - x_out)**2).reshape((x_out.shape[0], 5, 7, 1)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy()

		mse_loss[-1] += pred_mse.tolist()
		# print(np.mean(pred_mse))
	# print(torch.sum(torch.vstack(alpha_dist), 0))
	mse_loss[-1] = np.array(mse_loss[-1])*100
	# print('')

	# print(mse_loss[-1].shape)
	print(f'{np.mean(mse_loss[-1]):.3f} $\pm$ {np.std(mse_loss[-1]):.3f}')