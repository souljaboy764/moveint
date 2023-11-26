# This file contains wrapped dataset classes from the phd_utils.dataloaders module

import numpy as np

from phd_utils.dataloaders import *
from phd_utils.data import *


class HandoverHH(alap.HHDataset):
	def __init__(self, train):
		super().__init__(train=train, window_length=5, downsample = 0.2)
		for i in range(len(self.output_data)):
			self.output_data[i] = np.array([joint_angle_extraction(self.output_data[i][t, 9:18].reshape((-1, 3, 3))) for t in range(self.output_data[i].shape[0])])
			p2_traj = self.input_data[i]
			_, dims = p2_traj.shape
			p2_rpos = p2_traj[:, dims//4:dims//2]
			p2_rvel = p2_traj[:, 3*dims//4:]
			self.input_data[i] = np.hstack([p2_rpos, p2_rvel])
			
		self.input_dims = self.input_data[0].shape[-1]
		self.output_dims = self.output_data[0].shape[-1]

	def __getitem__(self, index):
		return self.input_data[index], self.output_data[index]

class HandoverHH(alap.HHWindowDataset):
	def __init__(self, train):
		super().__init__(train=train, window_length=5, downsample = 0.2)
		
	def __getitem__(self, index):
		return self.input_data[index], self.output_data[index]

class HandoverKobo(alap.KoboWindowDataset):
	def __init__(self, train):
		super().__init__(train=train, window_length=5, downsample = 0.2)
		
	def __getitem__(self, index):
		return self.input_data[index], self.output_data[index]

class BuetepageHH(buetepage.HHWindowDataset):
	def __init__(self, train):
		super().__init__(train=train, window_length=5, downsample = 0.2)
		self.input_dims = self.output_dims = self.traj_data[0].shape[-1]//2
		
	def __getitem__(self, index):
		traj_data = self.traj_data[index].astype(np.float32)
		dims = traj_data.shape[-1]
		return traj_data[:, :dims//2], traj_data[:, dims//2:]
	
class BuetepageYumi(buetepage_hr.YumiWindowDataset):
	def __init__(self, train):
		super().__init__(train=train, window_length=5, downsample = 0.2)
		self.input_dims = self.traj_data[0].shape[-1] - 35
		self.output_dims = 35
		
	def __getitem__(self, index):
		traj_data = self.traj_data[index].astype(np.float32)
		return traj_data[:, :-35], traj_data[:, -35:]
	
class BuetepagePepper(buetepage.PepperWindowDataset):
	def __init__(self, train):
		super().__init__(train=train, window_length=5, downsample = 0.2)
		self.input_dims = self.traj_data[0].shape[-1] - 20
		self.output_dims = 20

	def __getitem__(self, index):
		traj_data = self.traj_data[index].astype(np.float32)
		return traj_data[:, :-20], traj_data[:, -20:]
	
class NuiSIHH(nuisi.HHWindowDataset):
	def __init__(self, train):
		super().__init__(train=train, window_length=5, downsample = 0.2)
		self.input_dims = self.output_dims = self.dataset.traj_data[0].shape[-1]//2

	def __getitem__(self, index):
		traj_data = self.traj_data[index].astype(np.float32)
		dims = traj_data.shape[-1]
		return traj_data[:, :dims//2], traj_data[:, dims//2:]
	
class NuiSIPepper(nuisi.PepperWindowDataset):
	def __init__(self, train):
		super().__init__(train=train, window_length=5, downsample = 0.2)
		self.input_dims = self.traj_data[0].shape[-1] - 20
		self.output_dims = 20

	def __getitem__(self, index):
		traj_data = self.traj_data[index].astype(np.float32)
		return traj_data[:, :-20], traj_data[:, -20:]