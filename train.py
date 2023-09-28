import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np

import os

from utils import *
from dataset import HumanHandoverDataset
import networks

def run_iteration(iterator:DataLoader, model:networks.BaseNet, optimizer:torch.optim.Optimizer, args:argparse.ArgumentParser, epoch:int):
	mse_loss, bce_loss, ae_loss = [], [], []
	for i, (x_in, x_out, label) in enumerate(iterator):
		if model.training:
			optimizer.zero_grad()

		x_in = torch.Tensor(x_in[0]).to(device)
		x_out = torch.Tensor(x_out[0]).to(device)
		label = torch.eye(3, device=device)[label[0]]
		if args.model == 'RMDVAE':
			h_mean, h_std, h_alpha, r_mean, r_std, r_out_r, r_out_h = model(x_in, x_out, label)
			
		else:
			policy, policy_std, segment_pred = model(x_in)
			if args.loss=='mse':
				mse = ((policy - x_out)**2).mean(-1)
			elif args.loss=='mc_mse':
				eps = torch.randn((5,)+x_out.shape)
				x_out_samples = policy[None] + eps*policy_std[None]
				mse = ((x_out_samples - x_out[None])**2).mean(-1)
			elif args.loss=='nll':
				mse = (((policy - x_out.repeat(1,args.num_components))/policy_std)**2).mean(-1)
			mse_loss += mse.tolist()

			bce = F.binary_cross_entropy(segment_pred, label, reduction='none').sum(-1)
			bce_loss += bce.tolist()

		loss = mse + bce
		loss = loss.mean()
		if model.training:
			loss.backward()
			optimizer.step()

	return mse_loss, bce_loss, ae_loss

args = training_argparse()
assert not (args.model == 'FeedFwdNet' and args.loss!='mse')
print('Random Seed',args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.autograd.set_detect_anomaly(True)


print("Reading Data")
train_iterator = DataLoader(HumanHandoverDataset(args, train=True), batch_size=1, shuffle=True)
test_iterator = DataLoader(HumanHandoverDataset(args, train=False), batch_size=1, shuffle=False)

print("Creating Paths")
MODELS_FOLDER = os.path.join(args.results, "models")
SUMMARIES_FOLDER = os.path.join(args.results, "summary")

if not os.path.exists(args.results):
	print("Creating Result Directory")
	os.makedirs(args.results)
if not os.path.exists(MODELS_FOLDER):
	print("Creating Model Directory")
	os.makedirs(MODELS_FOLDER)
if not os.path.exists(SUMMARIES_FOLDER):
	print("Creating Model Directory")
	os.makedirs(SUMMARIES_FOLDER)

global_epochs = 0

print("Creating Model and Optimizer")
model = getattr(networks, args.model)(train_iterator.dataset.input_dims, train_iterator.dataset.output_dims, args).to(device)
params = model.parameters()
torch.compile(model)
named_params = model.named_parameters()
optimizer = torch.optim.AdamW(params, lr=args.lr, fused=True)
if args.ckpt is not None:
	ckpt = torch.load(args.ckpt)
	model.load_state_dict(ckpt['model'])
	optimizer.load_state_dict(ckpt['optimizer'])
	# global_epochs = ckpt['epoch']

print("Building Writer")
writer = SummaryWriter(SUMMARIES_FOLDER)
s = ''
for k in args.__dict__:
	s += str(k) + ' : ' + str(args.__dict__[k]) + '\n'
writer.add_text('args', s)

writer.flush()

for epoch in range(global_epochs, args.epochs):
	model.train()
	mse_train, bce_train, ae_train = run_iteration(train_iterator, model, optimizer, args, epoch)
	
	if epoch % 10 == 0 or epoch==args.epochs-1:
		model.eval()
		with torch.no_grad():
			mse_test, bce_test, ae_test = run_iteration(test_iterator, model, optimizer, args, epoch)
		writer.add_scalar('train/pred_mse', np.mean(mse_train), epoch)
		writer.add_scalar('train/alpha_bce', np.mean(bce_train), epoch)
		writer.add_scalar('test/pred_mse', np.mean(mse_test), epoch)
		writer.add_scalar('test/alpha_bce', np.mean(bce_test), epoch)
		if args.model == 'RMDVAE':
			writer.add_scalar('train/recon', np.mean(ae_train), epoch)
			writer.add_scalar('test/recon', np.mean(ae_test), epoch)
		params = []
		grads = []
		for name, param in model.named_parameters():
			if param.grad is None:
				continue
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), epoch)
			writer.add_histogram('param/'+name, param.reshape(-1), epoch)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		checkpoint_file = os.path.join(MODELS_FOLDER, '%0.3d.pth'%(epoch))
		torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args':args}, checkpoint_file)

	print(epoch,'epochs done')

writer.flush()

checkpoint_file = os.path.join(MODELS_FOLDER, f'final_{epoch}.pth')
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args':args}, checkpoint_file)
