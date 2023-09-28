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

from utils import *
from dataset import HumanHandoverDataset
import networks

parser = argparse.ArgumentParser(description='RMDVAE Testing')
parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
						help='Checkpoint to evaluate')

args = training_argparse()
ckpt = torch.load(args.ckpt)
training_args = ckpt['args']

print("Reading Data")
test_iterator = DataLoader(HumanHandoverDataset(training_args, train=False), batch_size=1, shuffle=False)

print("Creating Model and Optimizer")
model = networks.RMDVAE(test_iterator.dataset.input_dims, test_iterator.dataset.output_dims, training_args).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
mse_loss = []
hand_preds = []
alpha_preds = []
accuracy = []
confusion_matrix = np.zeros((3, 3))
for i, (x_in, x_out, label) in enumerate(test_iterator):
	x_in = torch.Tensor(x_in[0]).to(device)
	x_out = torch.Tensor(x_out[0]).to(device)
	label = label[0]

	with torch.no_grad():
		h_mean, h_std, h_alpha, r_mean, r_std, r_out_r, r_out_h = model(x_in, None)
	
	pred_mse = ((r_out_h - x_out)**2).mean(-1)
	hand_preds.append(r_out_h.cpu().numpy())
	label_preds = h_alpha.argmax(1)
	h_alpha = nn.Softmax(-1)(h_alpha)
	threshold = 0.95
	one_boundary = torch.logical_and(h_alpha[:, 0]<threshold, h_alpha[:, 1]>1-threshold)
	two_boundary = torch.logical_and(h_alpha[:, 1]<threshold, h_alpha[:, 2]>1-threshold)
	label_preds[one_boundary] = 1
	label_preds[two_boundary] = 2
	confusion_matrix += sklearn.metrics.confusion_matrix(label, label_preds.cpu().numpy())
	accuracy.append(sklearn.metrics.accuracy_score(label, label_preds.cpu().numpy()))

	if i==0:
		mse_loss = pred_mse
	else:
		mse_loss = torch.cat([mse_loss, pred_mse])

accuracy = np.vstack(accuracy)
mse_loss = mse_loss.cpu().numpy()
num_samples = accuracy.shape[0]
# print(nn.Softmax(-1)(h_alpha))
# print(label)
print(confusion_matrix)
print(accuracy.mean(), accuracy.std())
print(mse_loss.mean(), mse_loss.std())
