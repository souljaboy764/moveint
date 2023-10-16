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
model = getattr(rmdn_hri.networks, 'RMDVAE')(test_iterator.dataset.input_dims, test_iterator.dataset.output_dims, training_args).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
# i = np.random.randint(0, 12)#, len(test_iterator.dataset))
# i = 7 # Best bimanual
# i = 23 # Best unimanual
# print(i)
inputs = []
gt = []
outputs = []
alphas = []
latents = []
r_traj_i = []
# for idx in [[4], [6], [12], [23]]: # handovers
# for idx in [[4], [8], [23], [23]]: # handovers
for idx in test_iterator.dataset.dataset.actidx:
    i = idx[0]
    x_in, x_out = test_iterator.dataset.__getitem__(i)
    inputs.append(x_in)
    gt.append(x_out)
    x_in = torch.Tensor(x_in).to(device)
    x_out = torch.Tensor(x_out).to(device)
    with torch.no_grad():
        h_mean, h_std, h_alpha, h_mean_combined, h_std_combined, r_mean, r_std, r_out_r, r_out_h, r_samples_h, r_samples_r = model(x_in, x_out)
        # h_mean, h_std, h_alpha = model(x_in, x_out)
        # h_mean = (h_mean*h_alpha[..., None]).sum(1, keepdims=True)
        # print(h_alpha)
        # print(h_alpha.argmax(1))

        r_traj_i.append(model.robot_decoder(h_mean).cpu().numpy())
        outputs.append(r_out_h.cpu().numpy())
        print(outputs[-1].shape)
        latents.append(h_mean.cpu().numpy())
        alphas.append(h_alpha.cpu().numpy())
            # r_traj_i.append(h_mean[:, i].cpu().numpy())

np.savez_compressed('logs/samples/bp_hh.npz', xh=np.array(inputs, dtype=object), xr=np.array(outputs, dtype=object), xr_i=np.array(r_traj_i, dtype=object), zh=np.array(latents, dtype=object), alpha=np.array(alphas, dtype=object), xr_gt=np.array(gt, dtype=object))