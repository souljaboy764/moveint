import torch
from mild_hri.transformations import *

import argparse
import numpy as np

joints = ['Hip', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 'LFArm', 'LHand', 'RShoulder', 'RUArm', 'RFArm', 'RHand']
joints_dic = {joints[i]:i for i in range(len(joints))}
object_key = 'Rigid Body:object:Position'
label_idx = {'reach':0, 'transfer':1, 'retreat':2}

def angle(a,b):
	dot = np.dot(a,b)
	cos = dot/(np.linalg.norm(a)*np.linalg.norm(b))
	if np.allclose(cos, 1):
		cos = 1
	elif np.allclose(cos, -1):
		cos = -1
	return np.arccos(cos)

def joint_angle_extraction(skeleton): # Based on the Pepper Robot URDF, with the limits
	# Recreating arm with upper and under arm
	rightUpperArm = skeleton[1] - skeleton[0]
	rightUnderArm = skeleton[2] - skeleton[1]


	rightElbowAngle = np.clip(angle(rightUpperArm, rightUnderArm), 0.0087, 1.562)
	
	rightYaw = np.clip(np.arcsin(min(rightUpperArm[1],-0.0087)/np.linalg.norm(rightUpperArm)), -1.562, -0.0087)
	
	rightPitch = np.arctan2(max(rightUpperArm[0],0), rightUpperArm[2])
	rightPitch -= np.pi/2 # Needed for pepper frame
	
	# Recreating under Arm Position with known Angles(without roll)
	rightRotationAroundY = euler_matrix(0, rightPitch, 0,)[:3,:3]
	rightRotationAroundX = euler_matrix(0, 0, rightYaw)[:3,:3]
	rightElbowRotation = euler_matrix(0, 0, rightElbowAngle)[:3,:3]

	rightUnderArmInZeroPos = np.array([np.linalg.norm(rightUnderArm), 0, 0.])
	rightUnderArmWithoutRoll = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightElbowRotation,rightUnderArmInZeroPos)))

	# Calculating the angle betwenn actual under arm position and the one calculated without roll
	rightRoll = angle(rightUnderArmWithoutRoll, rightUnderArm)

	return np.array([rightPitch, rightYaw, rightRoll, rightElbowAngle]).astype(np.float32)

"""Implementation of the straight-through gumbel-rao estimator.
https://github.com/nshepperd/gumbel-rao-pytorch/

Paper: "Rao-Blackwellizing the Straight-Through Gumbel-Softmax
Gradient Estimator" <https://arxiv.org/abs/2010.04838>.

Note: the implementation here differs from the paper in that we DO NOT
propagate gradients through the conditional G_k | D
reparameterization. The paper states that:

	Note that the total derivative d(softmax_τ(θ + Gk))/dθ is taken
	through both θ and Gk. For the case K = 1, our estimator reduces to
	the standard ST-GS estimator.

I believe that this is a mistake - the expectation of this estimator
is only equal to that of ST-GS if the derivative is *not* taken
through G_k, as the ST-GS estimator ∂f(D)/dD d(softmax_τ(θ + G))/dθ
does not.

With the derivative ignored through G_k, the value of this estimator
with k=1 is numerically equal to that of ST-GS, and as k->∞ the
estimator for any given outcome D converges to the expectation of
ST-GS over G conditional on D.

"""

@torch.no_grad()
def conditional_gumbel(logits, D, k=1):
	"""Outputs k samples of Q = StandardGumbel(), such that argmax(logits
	+ Q) is given by D (one hot vector)."""
	# iid. exponential
	E = torch.distributions.exponential.Exponential(rate=torch.ones_like(logits)).sample([k])
	# E of the chosen class
	Ei = (D * E).sum(dim=-1, keepdim=True)
	# partition function (normalization constant)
	Z = logits.exp().sum(dim=-1, keepdim=True)
	# Sampled gumbel-adjusted logits
	adjusted = (D * (-torch.log(Ei) + torch.log(Z)) +
				(1 - D) * -torch.log(E/torch.exp(logits) + Ei / Z))
	return adjusted - logits


def exact_conditional_gumbel(logits, D, k=1):
	"""Same as conditional_gumbel but uses rejection sampling."""
	# Rejection sampling.
	idx = D.argmax(dim=-1)
	gumbels = []
	while len(gumbels) < k:
		gumbel = torch.rand_like(logits).log().neg().log().neg()
		if logits.add(gumbel).argmax() == idx:
			gumbels.append(gumbel)
	return torch.stack(gumbels)


def replace_gradient(value, surrogate):
	"""Returns `value` but backpropagates gradients through `surrogate`."""
	return surrogate + (value - surrogate).detach()


def gumbel_rao(logits, k, temp=0.1, I=None):
	"""Returns a categorical sample from logits (over axis=-1) as a
	one-hot vector, with gumbel-rao gradient.

	k: integer number of samples to use in the rao-blackwellization.
	1 sample reduces to straight-through gumbel-softmax.

	I: optional, categorical sample to use instead of drawing a new
	sample. Should be a tensor(shape=logits.shape[:-1], dtype=int64).

	"""
	num_classes = logits.shape[-1]
	if I is None:
		I = torch.distributions.categorical.Categorical(logits=logits).sample()
	D = torch.nn.functional.one_hot(I, num_classes).float()
	adjusted = logits + conditional_gumbel(logits, D, k=k)
	surrogate = torch.nn.functional.softmax(adjusted/temp, dim=-1).mean(dim=0)
	return replace_gradient(D, surrogate)

# >>> exact_conditional_gumbel(torch.tensor([[1.0,2.0, 3.0]]), torch.tensor([[0.0, 1.0, 0.0]]), k=10000).std(dim=0)
# tensor([[0.9952, 1.2695, 0.8132]])
# >>> conditional_gumbel(torch.tensor([[1.0,2.0, 3.0]]), torch.tensor([[0.0, 1.0, 0.0]]), k=10000).std(dim=0)
# tensor([[0.9905, 1.2951, 0.8148]])

def training_argparse(args=None):
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	# Results and Paths
	parser.add_argument('--results', type=str, default='./logs/debug',#+datetime.datetime.now().strftime("%m%d%H%M"),
						help='Path for saving results (default: ./logs/results/MMDDHHmm).', metavar='RES')
	parser.add_argument('--seed', type=int, default=np.random.randint(0,np.iinfo(np.int32).max), metavar='SEED',
						help='Random seed for training (randomized by default).')
	
	# Input data shapers
	parser.add_argument('--dataset', type=str, default='BuetepagePepper', metavar='DATSET',  choices=['HandoverHH', 'UnimanualHandover', 'BimanualHandover', 'BuetepageHH', 'BuetepageYumi', 'BuetepagePepper', 'NuiSIHH', 'NuiSIPepper'],
						help='Dataset to use: HandoverHH, UnimanualHandover, BimanualHandover, BuetepageHH, BuetepageYumi, BuetepagePepper, NuiSIHH or NuiSIPepper (default: HandoverHH)')
	
	# Model args
	parser.add_argument('--model', type=str, default='RMDVAE', metavar='MODEL', choices=['RMDN', 'RMDVAE'],
						help='Which VAE to use: RMDN or RMDVAE (default: RMDN).')
	parser.add_argument('--latent-dim', type=int, default=5, metavar='Z',
						help='Latent space dimension (default: 5)')
	parser.add_argument('--num-components', type=int, default=8, metavar='N_COMPONENTS',
						help='Number of components to use in MDN predictions (default: 3).')
	parser.add_argument('--std-reg', type=float, default=1e-6, metavar='EPS',
						help='Positive value to add to standard deviation predictions (default: 1e-3)')
	parser.add_argument('--hidden-sizes', default=[40,20], nargs='+', type=int,
			 			help='List of weights for the VAE layers (default: [250,150] )')
	parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
						help='Checkpoint to resume training from (default: None)')
	parser.add_argument('--activation', default='LeakyReLU', type=str,
			 			help='Activation Function for the hidden layers')
	
	# Training args
	parser.add_argument('--grad-clip', type=float, default=None, metavar='CLIP',
						help='Value to clip gradients at (default: None)')
	parser.add_argument('--epochs', type=int, default=200, metavar='EPOCHS',
						help='Number of epochs to train for (default: 200)')
	parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
						help='Starting Learning Rate (default: 5e-4)')
	parser.add_argument('--mce-samples', type=int, default=10, metavar='MCE',
						help='Number of Monte Carlo samples to draw (default: 10)')
	parser.add_argument('--beta', type=float, default=0.001, metavar='BETA',
						help='Scaling factor for KL divergence (default: 0.005)')
	return parser.parse_args(args)
