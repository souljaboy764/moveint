import numpy as np

from torch.utils.data import Dataset

from mild_hri.transformations import *
from mild_hri.dataloaders import *

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

# P1 - Giver, P2 - Receiver. Currently commented out object trajectories due to inconsistency in object marker location. Can be mitigated as post-processing but haven't done that yet.
class UnimanualDataset(Dataset):
	def __init__(self, args, train):
		with np.load(args.src, allow_pickle=True) as data:
			if train:
				p1_trajs, p2_trajs, object_trajs, self.labels = data['train_data']
			else:
				p1_trajs, p2_trajs, object_trajs, self.labels = data['test_data']

		self.input_data = []
		self.output_data = []
		for i in range(len(p1_trajs)):
			robot_rhand_trajs = np.array([joint_angle_extraction(p1_trajs[i][t, -3:]) for t in range(p1_trajs[i].shape[0])])
			
			p2_trajs[i][:, :, 0] *= -1
			p2_trajs[i][:, :, 1] *= -1
			p2_trajs[i] -= p2_trajs[i][0:1, -3] # right shoulder as origin
			p2_rhand_trajs = p2_trajs[i][::4, -3:].reshape((-1, 9))
			p2_rhand_vels = np.diff(p2_rhand_trajs, axis=0, prepend=p2_rhand_trajs[0:1])
			
			# p1r_p2l_dist = p2_rhand_trajs - p1_rhand_trajs
			
			# min_dist_idx = np.linalg.norm(p1r_p2l_dist, axis=-1).argmin()
			# goals_idx = np.ones(p2_rhand_vels.shape[0])
			# goals_idx[self.labels[i]==0] = min_dist_idx
			# goals_idx[self.labels[i]==1] = min_dist_idx
			# goals_idx[self.labels[i]==2] = -1

			# goals_idx = goals_idx.astype(int)


			self.input_data.append(np.concatenate([
									p2_rhand_trajs, 
									p2_rhand_vels, 
								], axis=-1))
			
			self.output_data.append(robot_rhand_trajs[::4])
		self.input_data = np.array(self.input_data, dtype=object)
		self.output_data = np.array(self.output_data, dtype=object)
		self.input_dims = self.input_data[0].shape[-1]
		self.output_dims = self.output_data[0].shape[-1]
		self.len = len(self.input_data)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.input_data[index], self.output_data[index]


# P1 - Giver, P2 - Receiver. Currently commented out object trajectories due to inconsistency in object marker location. Can be mitigated as post-processing but haven't done that yet.
class BimanualDataset(Dataset):
	def __init__(self, args, train):
		with np.load(args.src, allow_pickle=True) as data:
			if train:
				p1_trajs, p2_trajs, object_trajs, self.labels = data['train_data']
			else:
				p1_trajs, p2_trajs, object_trajs, self.labels = data['test_data']
			joints = data['joints']
			joints_dic = {joints[i]:i for i in range(len(joints))}
			self.input_data = []
			self.output_data = []
			for i in range(len(p1_trajs)):
				p1_rhand_trajs = p1_trajs[i][:, joints_dic['RHand']]
				p1_lhand_trajs = p1_trajs[i][:, joints_dic['LHand']]
				p2_rhand_trajs = p2_trajs[i][:, joints_dic['RHand']]
				p2_lhand_trajs = p2_trajs[i][:, joints_dic['LHand']]

				p1_rhand_vels = np.diff(p1_rhand_trajs, axis=0, prepend=p1_rhand_trajs[0:1])
				p1_lhand_vels = np.diff(p1_lhand_trajs, axis=0, prepend=p1_lhand_trajs[0:1])
				p2_rhand_vels = np.diff(p2_rhand_trajs, axis=0, prepend=p2_rhand_trajs[0:1])
				p2_lhand_vels = np.diff(p2_lhand_trajs, axis=0, prepend=p2_lhand_trajs[0:1])

				p1_rhand_objdist = p1_rhand_trajs - object_trajs[i]
				p2_lhand_objdist = p2_lhand_trajs - object_trajs[i]

				p1r_p2l_dist = p2_lhand_trajs - p1_rhand_trajs
				p1r_p2l_vels = np.diff(p1r_p2l_dist, axis=0, prepend=p1r_p2l_dist[0:1])

				min_dist_idx = np.linalg.norm(p1r_p2l_dist, axis=-1).argmin()
				goals_idx = np.ones(p2_lhand_trajs.shape[0])
				goals_idx[self.labels[i]==0] = min_dist_idx
				goals_idx[self.labels[i]==1] = min_dist_idx
				goals_idx[self.labels[i]==2] = -1

				goals_idx = goals_idx.astype(int)


				self.input_data.append(np.concatenate([
										p2_rhand_trajs, 
										p2_lhand_trajs, 
										p2_rhand_vels, 
										p2_lhand_vels, 
										# p1_rhand_objdist, 
										# p2_lhand_objdist, 
										# p1r_p2l_dist,
										# p1r_p2l_vels,
									], axis=-1))
				
				self.output_data.append(np.concatenate([
										p1_rhand_trajs, 
										p1_lhand_trajs,
										# p1_rhand_trajs[goals_idx],
										# p1_lhand_trajs[goals_idx],
										# p1_rhand_vels, 
										# p1_lhand_vels,
									], axis=-1))

			self.input_data = np.array(self.input_data, dtype=object)
			self.output_data = np.array(self.output_data, dtype=object)
			self.input_dims = self.input_data[0].shape[-1]
			self.output_dims = self.output_data[0].shape[-1]
			self.len = len(self.input_data)
			
	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.input_data[index], self.output_data[index]
