import numpy as np

from torch.utils.data import Dataset
from mild_hri.dataloaders import *
from rmdn_hri.utils import *

# P1 - Giver, P2 - Receiver. Currently commented out object trajectories due to inconsistency in object marker location. Can be mitigated as post-processing but haven't done that yet.
class UnimanualHandover(Dataset):
	def __init__(self, train, window_length=5):
		with np.load('data/alap_dataset_unimanual.npz', allow_pickle=True) as data:
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
class BimanualKobo(Dataset):
	def __init__(self, train, window_length=5):
		with np.load('data/alap_dataset_bimanual_kobo_p2frame.npz', allow_pickle=True) as data:
			if train:
				p1_trajs, p2_trajs, object_trajs, self.labels = data['train_data']
			else:
				p1_trajs, p2_trajs, object_trajs, self.labels = data['test_data']
			joints = data['joints']
			joints_dic = {joints[i]:i for i in range(len(joints))}
			self.input_data = []
			self.output_data = []
			for i in range(len(p1_trajs)):
				p1_rhand_trajs = p1_trajs[i][::4, joints_dic['RHand']]
				p1_lhand_trajs = p1_trajs[i][::4, joints_dic['LHand']]
				p2_rhand_trajs = p2_trajs[i][::4, joints_dic['RHand']]
				p2_lhand_trajs = p2_trajs[i][::4, joints_dic['LHand']]

				p1_rhand_vels = np.diff(p1_rhand_trajs, axis=0, prepend=p1_rhand_trajs[0:1])
				p1_lhand_vels = np.diff(p1_lhand_trajs, axis=0, prepend=p1_lhand_trajs[0:1])
				p2_rhand_vels = np.diff(p2_rhand_trajs, axis=0, prepend=p2_rhand_trajs[0:1])
				p2_lhand_vels = np.diff(p2_lhand_trajs, axis=0, prepend=p2_lhand_trajs[0:1])

				# p1_rhand_objdist = p1_rhand_trajs - object_trajs[i]
				# p2_lhand_objdist = p2_lhand_trajs - object_trajs[i]

				# p1r_p2l_dist = p2_lhand_trajs - p1_rhand_trajs
				# p1r_p2l_vels = np.diff(p1r_p2l_dist, axis=0, prepend=p1r_p2l_dist[0:1])

				# min_dist_idx = np.linalg.norm(p1r_p2l_dist, axis=-1).argmin()
				# goals_idx = np.ones(p2_lhand_trajs.shape[0])
				# goals_idx[self.labels[i]==0] = min_dist_idx
				# goals_idx[self.labels[i]==1] = min_dist_idx
				# goals_idx[self.labels[i]==2] = -1

				# goals_idx = goals_idx.astype(int)


				input_traj = np.concatenate([
										# p2_rhand_trajs, 
										p2_lhand_trajs, 
										# p2_rhand_vels, 
										p2_lhand_vels, 
										# p1_rhand_objdist, 
										# p2_lhand_objdist, 
										# p1r_p2l_dist,
										# p1r_p2l_vels,
									], axis=-1)
				
				output_traj = np.concatenate([
										p1_rhand_trajs, 
										p1_lhand_trajs,
										# p1_rhand_trajs[goals_idx],
										# p1_lhand_trajs[goals_idx],
										# p1_rhand_vels, 
										# p1_lhand_vels,
									], axis=-1)
				
				# seq_len = p1_rhand_trajs.shape[0]
				# input_dim = input_traj.shape[-1]
				# output_dim = output_traj.shape[-1]
				# idx = np.array([np.arange(i,i+window_length) for i in range(seq_len + 1 - 2*window_length)])
				self.input_data.append(input_traj)#[idx].reshape((seq_len + 1 - 2*window_length, window_length*input_dim)))
				# idx = np.array([np.arange(i,i+window_length) for i in range(window_length, seq_len + 1 - window_length)])
				self.output_data.append(output_traj)#[idx].reshape((seq_len + 1 - 2*window_length, window_length*output_dim)))

			self.input_data = np.array(self.input_data, dtype=object)
			self.output_data = np.array(self.output_data, dtype=object)
			self.input_dims = self.input_data[0].shape[-1]
			self.output_dims = self.output_data[0].shape[-1]

			self.len = len(self.input_data)
			
	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.input_data[index], self.output_data[index]
	
class HandoverHH:
	# mode: 1 - unimanual, 2 - bimanual, 3 - both
	def __init__(self, train, mode=3, window_length=5):
		with np.load('data/alap_dataset_combined.npz', allow_pickle=True) as data:
			if train:
				p1_trajs, p2_trajs, _, _ = data['train_data']
				self.actidx = np.array([[0,105], [105, 168]])
			else:
				p1_trajs, p2_trajs, _, _ = data['test_data']
				self.actidx = np.array([[0,12], [12, 24]])
			joints = data['joints']

		joints_dic = {joints[i]:i for i in range(len(joints))}
		# joints_idx = [joints_dic[i] for i in ['LUArm', 'LFArm', 'LHand', 'RUArm', 'RFArm', 'RHand']]
		p1_joints_idx = [joints_dic[i] for i in ['LHand', 'RHand']]
		p2_joints_idx = [joints_dic[i] for i in ['LUArm', 'LFArm', 'LHand', 'RUArm', 'RFArm', 'RHand']]
		self.input_data = []
		self.output_data = []
		for i in range(len(p1_trajs)):
			p1_pos = p1_trajs[i][::4, p1_joints_idx]
			p1_pos = p1_pos.reshape((p1_pos.shape[0], 3*len(p1_joints_idx)))
			p2_pos = p2_trajs[i][::4, p2_joints_idx]
			p2_pos = p2_pos.reshape((p2_pos.shape[0], 3*len(p2_joints_idx)))

			p1_vel = np.diff(p1_pos, axis=0, prepend=p1_pos[0:1])
			p2_vel = np.diff(p2_pos, axis=0, prepend=p2_pos[0:1])

			p1_traj = np.hstack([p1_pos, p1_vel])
			p2_traj = np.hstack([p2_pos, p2_vel])
			
			input_dim = output_dim = p1_traj.shape[-1]
			seq_len = p1_traj.shape[0]
			idx = np.array([np.arange(i,i+window_length) for i in range(seq_len + 1 - 2*window_length)])
			self.input_data.append(p2_traj[idx].reshape((seq_len + 1 - 2*window_length, window_length*input_dim)))
			idx = np.array([np.arange(i,i+window_length) for i in range(window_length, seq_len + 1 - window_length)])
			self.output_data.append(p1_traj[idx].reshape((seq_len + 1 - 2*window_length, window_length*output_dim)))

		self.input_data = np.array(self.input_data, dtype=object)
		self.output_data = np.array(self.output_data, dtype=object)
		self.input_dims = self.input_data[0].shape[-1]
		self.output_dims = self.output_data[0].shape[-1]

		self.len = len(self.input_data)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.input_data[index], self.output_data[index]


class HandoverKobo:
	# mode: 1 - unimanual, 2 - bimanual, 3 - both
	def __init__(self, train, mode=3, window_length=5):
		with np.load('data/alap_dataset_combined_kobo.npz', allow_pickle=True) as data:
			if train:
				p1_trajs, p2_trajs, _, _ = data['train_data']
				self.actidx = np.array([[0,105], [105, 168]])
			else:
				p1_trajs, p2_trajs, _, _ = data['test_data']
				self.actidx = np.array([[0,12], [12, 24]])
			joints = data['joints']

		joints_dic = {joints[i]:i for i in range(len(joints))}
		p1_joints_idx = [joints_dic[i] for i in ['LHand', 'RHand']]
		p2_joints_idx = [joints_dic[i] for i in ['LUArm', 'LFArm', 'LHand', 'RUArm', 'RFArm', 'RHand']]
		self.input_data = []
		self.output_data = []
		for i in range(len(p1_trajs)):
			p1_pos = p1_trajs[i][::4, p1_joints_idx]
			p1_pos = p1_pos.reshape((p1_pos.shape[0], 3*len(p1_joints_idx)))
			p2_pos = p2_trajs[i][::4, p2_joints_idx]
			p2_pos = p2_pos.reshape((p2_pos.shape[0], 3*len(p2_joints_idx)))

			# p1_vel = np.diff(p1_pos, axis=0, prepend=p1_pos[0:1])
			p2_vel = np.diff(p2_pos, axis=0, prepend=p2_pos[0:1])

			# p1_traj = np.hstack([p1_pos, p1_vel])
			p1_traj = p1_pos
			p2_traj = np.hstack([p2_pos, p2_vel])
			
			# input_dim = p2_traj.shape[-1]
			# output_dim = p1_traj.shape[-1]
			# seq_len = p1_traj.shape[0]
			# idx = np.array([np.arange(i,i+window_length) for i in range(seq_len + 1 - 2*window_length)])
			# self.input_data.append(p2_traj[idx].reshape((seq_len + 1 - 2*window_length, window_length*input_dim)))
			# idx = np.array([np.arange(i,i+window_length) for i in range(window_length, seq_len + 1 - window_length)])
			# self.output_data.append(p1_traj[idx].reshape((seq_len + 1 - 2*window_length, window_length*output_dim)))

			self.input_data.append(p2_traj)
			self.output_data.append(p1_traj)

		self.input_data = np.array(self.input_data, dtype=object)
		self.output_data = np.array(self.output_data, dtype=object)
		self.input_dims = self.input_data[0].shape[-1]
		self.output_dims = self.output_data[0].shape[-1]
		print(self.input_dims, self.output_dims)

		self.len = len(self.input_data)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.input_data[index], self.output_data[index]


class BuetepageHH:
	def __init__(self, train):
		self.dataset = buetepage.HHWindowDataset('/home/vignesh/playground/mild_hri/data/buetepage/traj_data.npz', train=train, window_length=5, downsample = 0.2)
		self.input_dims = self.output_dims = self.dataset.traj_data[0].shape[-1]//2

	def __len__(self):
		return self.dataset.len
		
	def __getitem__(self, index):
		traj_data, _ = self.dataset.__getitem__(index)
		dims = traj_data.shape[-1]
		return traj_data[:, :dims//2], traj_data[:, dims//2:]
	
class BuetepageYumi:
	def __init__(self, train):
		self.dataset = buetepage_hr.YumiWindowDataset('/home/vignesh/playground/mild_hri/data/buetepage_hr/traj_data.npz', train=train, window_length=5, downsample = 0.2)
		self.input_dims = self.dataset.traj_data[0].shape[-1] - 35
		self.output_dims = 35

	def __len__(self):
		return self.dataset.len
		
	def __getitem__(self, index):
		traj_data, _ = self.dataset.__getitem__(index)
		return traj_data[:, :-35], traj_data[:, -35:]
	
class BuetepagePepper:
	def __init__(self, train):
		self.dataset = buetepage.PepperWindowDataset('/home/vignesh/playground/mild_hri/data/buetepage/traj_data.npz', train=train, window_length=5, downsample = 0.2)
		print(self.dataset.traj_data[0].shape)
		self.input_dims = self.dataset.traj_data[0].shape[-1] - 20
		self.output_dims = 20

	def __len__(self):
		return self.dataset.len
			
	def __getitem__(self, index):
		traj_data, _ = self.dataset.__getitem__(index)
		return traj_data[:, :-20], traj_data[:, -20:]
	
class NuiSIHH:
	def __init__(self, train):
		self.dataset = nuisi.HHWindowDataset('/home/vignesh/playground/mild_hri/data/nuisi/traj_data.npz', train=train, window_length=5, downsample = 0.2)
		self.input_dims = self.output_dims = self.dataset.traj_data[0].shape[-1]//2

	def __len__(self):
		return self.dataset.len
					
	def __getitem__(self, index):
		traj_data, _ = self.dataset.__getitem__(index)
		dims = traj_data.shape[-1]
		return traj_data[:, :dims//2], traj_data[:, dims//2:]
	
class NuiSIPepper:
	def __init__(self, train):
		self.dataset = nuisi.PepperWindowDataset('/home/vignesh/playground/mild_hri/data/nuisi/traj_data.npz', train=train, window_length=5, downsample = 0.2)
		self.input_dims = self.dataset.traj_data[0].shape[-1] - 20
		self.output_dims = 20

	def __len__(self):
		return self.dataset.len
			
	def __getitem__(self, index):
		traj_data, _ = self.dataset.__getitem__(index)
		return traj_data[:, :-20], traj_data[:, -20:]