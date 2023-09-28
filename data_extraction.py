import numpy as np
import glob
import os
import csv

from utils import *

def preproc_files_list(files):
	p1_trajs = []
	p2_trajs = []
	object_trajs = []
	labels = []
	for f_csv in files:
		p1_name = os.path.basename(f_csv).split('_')[0]
		p2_name = os.path.basename(f_csv).split('_')[1]
		p1_body_keys = [f'Bone:{p1_name}:{joint}:Position' for joint in joints]
		p2_body_keys = [f'Bone:{p2_name}:{joint}:Position' for joint in joints]
		p1_traj = []
		p2_traj = []
		object_traj = []
		label = []
		reader = csv.DictReader(open(f_csv))
		try:
			for row in reader:
				p1_traj.append([])
				p2_traj.append([])
				for k in p1_body_keys:
					p1_traj[-1].append([-float(row[k+':X']), float(row[k+':Z']), float(row[k+':Y'])])
				for k in p2_body_keys:
					p2_traj[-1].append([-float(row[k+':X']), float(row[k+':Z']), float(row[k+':Y'])])
				object_traj.append([-float(row[object_key+':X']), float(row[object_key+':Z']), float(row[object_key+':Y'])])
				label.append(label_idx[row['Giver']])
			p1_traj = np.array(p1_traj, dtype=np.float32)
			p2_traj = np.array(p2_traj, dtype=np.float32)
			object_traj = np.array(object_traj, dtype=np.float32)
			origin = p1_traj[0:1, joints_dic["RUArm"]].copy()
			for t in range(p1_traj.shape[0]):
				p1_traj[t] = p1_traj[t] - origin
				p2_traj[t] = p2_traj[t] - origin
			object_traj = object_traj - origin
			p1_trajs.append(p1_traj)
			p2_trajs.append(p2_traj)
			object_trajs.append(object_traj)
			labels.append(np.array(label, dtype=int))
		except Exception as e:
			print(f'Error encountered: {e.__str__()}\nSkipping file {f_csv}')
	p1_trajs = np.array(p1_trajs, dtype=object)
	p2_trajs = np.array(p2_trajs, dtype=object)
	object_trajs = np.array(object_trajs, dtype=object)
	labels = np.array(labels, dtype=object)
	print(p1_trajs.shape, p2_trajs.shape, object_trajs.shape, labels.shape)
	return p1_trajs, p2_trajs, object_trajs, labels


train_dirs = ['P07_P08', 'P09_P10', 'P11_P12', 'P13_P14', 'P15_P16', 'P17_P18', 'P19_P20', 'P21_P22', 'P23_P24', 'P25_P26']
test_dirs = ['P27_P28', 'P29_P30']

train_files = []
for d in train_dirs:
	files = glob.glob(f'data/Bimanual Handovers Dataset/{d}/OptiTrack_Global_Frame/*double*.csv')
	files.sort()
	train_files += files
train_data = preproc_files_list(train_files)

test_files = []
for d in test_dirs:
	files = glob.glob(f'data/Bimanual Handovers Dataset/{d}/OptiTrack_Global_Frame/*double*.csv')
	files.sort()
	test_files += files
test_data = preproc_files_list(test_files)

np.savez_compressed('data/data_raw.npz', train_data=train_data, test_data=test_data, joints=joints)