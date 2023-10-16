import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np

import os

from rmdn_hri.utils import *
import rmdn_hri.dataset
import rmdn_hri.networks

args = training_argparse()
print('Random Seed',args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.autograd.set_detect_anomaly(True)


print("Reading Data")
train_iterator = DataLoader(getattr(rmdn_hri.dataset,args.dataset)(train=True), batch_size=1, shuffle=True)
test_iterator = DataLoader(getattr(rmdn_hri.dataset,args.dataset)(train=False), batch_size=1, shuffle=False)

print(train_iterator.dataset.input_dims, train_iterator.dataset.output_dims)
print(test_iterator.dataset.input_dims, test_iterator.dataset.output_dims)

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
model = getattr(rmdn_hri.networks,args.model)(train_iterator.dataset.input_dims, train_iterator.dataset.output_dims, args).to(device)
params = model.parameters()
# torch.compile(model)
named_params = model.named_parameters()
optimizer = torch.optim.AdamW(params, lr=args.lr)#, fused=True)
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
	d_train = model.run_iteration(train_iterator, optimizer, args, epoch)
	
	if epoch % 10 == 0 or epoch==args.epochs-1:
		model.eval()
		with torch.no_grad():
			d_test = model.run_iteration(test_iterator, optimizer, args, epoch)
		for k in d_train:
			writer.add_scalar('train/'+k, torch.mean(d_train[k]), epoch)
		for k in d_test:
			writer.add_scalar('test/'+k, torch.mean(d_test[k]), epoch)
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
