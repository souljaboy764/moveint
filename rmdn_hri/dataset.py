import numpy as np

from torch.utils.data import Dataset

# P1 - Giver, P2 - Receiver. Currently commented out object trajectories due to inconsistency in object marker location. Can be mitigated as post-processing but haven't done that yet.
class AlapDataset(Dataset):
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
				p2_rhand_vels = np.diff(p1_rhand_trajs, axis=0, prepend=p1_rhand_trajs[0:1])
				p2_lhand_vels = np.diff(p1_lhand_trajs, axis=0, prepend=p1_lhand_trajs[0:1])

				p1_rhand_objdist = p1_rhand_trajs - object_trajs[i]
				p2_lhand_objdist = p2_lhand_trajs - object_trajs[i]

				p1r_p2l_dist = p2_lhand_trajs - p1_rhand_trajs

				self.input_data.append(np.concatenate([
										p2_rhand_trajs, 
										p2_lhand_trajs, 
										p2_rhand_vels, 
										p2_lhand_vels, 
										# p1_rhand_objdist, 
										# p2_lhand_objdist, 
										# p1r_p2l_dist,
									], axis=-1))
				
				self.output_data.append(np.concatenate([
										p1_rhand_trajs, 
										p1_lhand_trajs,
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
		return self.input_data[index], self.output_data[index], self.labels[index]
