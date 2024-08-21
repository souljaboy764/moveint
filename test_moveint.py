import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np

from utils import *
import dataset
from networks import MoVEInt

parser = argparse.ArgumentParser(description='MoVEInt Testing')
parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
						help='Checkpoint to evaluate')

args = training_argparse()
ckpt = torch.load(args.ckpt)
training_args = ckpt['args']

test_iterator = DataLoader(getattr(dataset,training_args.dataset)(train=False), batch_size=1, shuffle=False)
model = MoVEInt(test_iterator.dataset.input_dims, test_iterator.dataset.output_dims, training_args).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
mse_loss = []#[] for i in test_iterator.dataset.actidx]
# if True:
for idx in test_iterator.dataset.actidx:
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
		if training_args.dataset=='NuiSIHH' or training_args.dataset =='BuetepageHH':
			pred_mse = ((r_out_h - x_out)**2).reshape((x_out.shape[0], 5, 6, 3)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy()

		# # # Handover HH
		elif training_args.dataset=='HandoverHH':
			pred_mse = ((r_out_h - x_out)**2).reshape((x_out.shape[0], 5, 12, 3)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy()
		
		# # BP/NuiSI Pepper
		elif training_args.dataset=='NuiSIPepper' or training_args.dataset =='BuetepagePepper':
			pred_mse = ((r_out_h - x_out)**2).reshape((x_out.shape[0], 5, 4, 1)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy()
		
		# # # BP/NuiSI Yumi
		elif training_args.dataset =='BuetepageYumi':
			pred_mse = ((r_out_h - x_out)**2).reshape((x_out.shape[0], 5, 7, 1)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy()

		mse_loss[-1] += pred_mse.tolist()
		# print(np.mean(pred_mse))
	# print(torch.sum(torch.vstack(alpha_dist), 0))
	mse_loss[-1] = np.array(mse_loss[-1])
	if training_args.dataset=='NuiSIHH' or training_args.dataset =='BuetepageHH' or training_args.dataset=='HandoverHH':
		mse_loss[-1] *= 100
	# print('')

	# print(mse_loss[-1].shape)
	print(f'{np.mean(mse_loss[-1]):.3f} $\pm$ {np.std(mse_loss[-1]):.3f}')