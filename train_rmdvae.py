import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np

import os

from utils import *
from dataset import HumanHandoverDataset
import networks


def run_iteration(iterator:DataLoader, model:networks.RMDVAE, optimizer:torch.optim.Optimizer, args:argparse.ArgumentParser, epoch:int):
	mse_loss, bce_loss, ae_loss, kl_loss = [], [], [], []
	
	for i, (x_in, x_out, label) in enumerate(iterator):
		if model.training:
			optimizer.zero_grad()
			with torch.no_grad():
				model.transitions.copy_(nn.Sigmoid()(model.transitions))

		label_dist = np.array([np.sum(label[0].numpy()==0), np.sum(label[0].numpy()==1), np.sum(label[0].numpy()==2)])
		w = 1/label_dist
		w /= w.sum()
		w = torch.Tensor(w).to(device)
		x_in = torch.Tensor(x_in[0]).to(device)
		x_out = torch.Tensor(x_out[0]).to(device)
		label_onehot = torch.eye(3, device=device)[label[0]]
		label = torch.Tensor(label[0]).to(device)
		h_mean, h_alpha, r_mean, r_std, r_out_r, r_out_h = model(x_in, x_out)
		
		kld = kl_divergence(Normal(r_mean, r_std), Normal((h_mean*label_onehot[:, None]).sum(-1), (model.human_std*label_onehot[:, None]).sum(-1))).mean(-1)
		
		ae_recon = ((r_out_r - x_out)**2).sum(-1)
		pred_mse = ((r_out_h - x_out)**2).sum(-1)
		if model.training:
			ae_recon = ae_recon.mean(0)
			pred_mse = pred_mse.mean(0)

		interclass_dist = (
							torch.exp(-((h_mean[..., 0] - h_mean[..., 1])**2).sum(-1)) + \
							torch.exp(-((h_mean[..., 2] - h_mean[..., 1])**2).sum(-1)) + \
							torch.exp(-((h_mean[..., 0] - h_mean[..., 2])**2).sum(-1))
		)/3

		intraclass_dist = torch.exp(-((torch.diff(h_mean, dim=0, prepend=h_mean[0:1]))**2).sum(-2)).mean(-1)
		
		kld += intraclass_dist - interclass_dist
		if model.training:
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
		if model.training:
			loss.backward()
			optimizer.step()

	return mse_loss, bce_loss, ae_loss, kl_loss

args = training_argparse()
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
model = networks.RMDVAE(train_iterator.dataset.input_dims, train_iterator.dataset.output_dims, args).to(device)
params = model.parameters()
# torch.compile(model)
named_params = model.named_parameters()
optimizer = torch.optim.AdamW(params, lr=args.lr)#, fused=True)
if args.ckpt is not None:
	ckpt = torch.load(args.ckpt)
	model.load_state_dict(ckpt['model'])
	optimizer.load_state_dict(ckpt['optimizer'])
	global_epochs = ckpt['epoch']

print("Building Writer")
writer = SummaryWriter(SUMMARIES_FOLDER)
s = ''
for k in args.__dict__:
	s += str(k) + ' : ' + str(args.__dict__[k]) + '\n'
writer.add_text('args', s)

writer.flush()

for epoch in range(global_epochs, args.epochs):
	model.train()
	mse_train, bce_train, ae_train, kl_train = run_iteration(train_iterator, model, optimizer, args, epoch)
	
	if epoch % 10 == 0 or epoch==args.epochs-1:
		model.eval()
		with torch.no_grad():
			mse_test, bce_test, ae_test, kl_test = run_iteration(test_iterator, model, optimizer, args, epoch)
		writer.add_scalar('train/pred_mse', torch.mean(mse_train), epoch)
		writer.add_scalar('train/alpha_bce', torch.mean(bce_train), epoch)
		writer.add_scalar('test/pred_mse', torch.mean(mse_test), epoch)
		writer.add_scalar('test/accuracy', torch.mean(bce_test), epoch)
		# writer.add_scalar('train/recon', torch.mean(ae_train), epoch)
		# writer.add_scalar('test/recon', torch.mean(ae_test), epoch)
		writer.add_scalar('train/kld', torch.mean(kl_train), epoch)
		writer.add_scalar('test/kld', torch.mean(kl_test), epoch)
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
