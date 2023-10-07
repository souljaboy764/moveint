import numpy as np
import scipy.ndimage
import glob
import os
import csv

from rmdn_hri.utils import *
from mild_hri.utils import *
import zipfile

def rotation_normalization(skeleton):
	leftShoulder = skeleton[1]
	rightShoulder = skeleton[2]
	waist = skeleton[0]
	
	xAxisHelper = waist - rightShoulder
	yAxis = leftShoulder - rightShoulder # right to left
	xAxis = np.cross(xAxisHelper, yAxis) # out of the human(like an arrow in the back)
	zAxis = np.cross(xAxis, yAxis) # like spine, but straight
	
	xAxis /= np.linalg.norm(xAxis)
	yAxis /= np.linalg.norm(yAxis)
	zAxis /= np.linalg.norm(zAxis)

	return np.array([[xAxis[0], xAxis[1], xAxis[2]],
					 [yAxis[0], yAxis[1], yAxis[2]],
					 [zAxis[0], zAxis[1], zAxis[2]]])

# train_dirs = ['Pair'+str(i) for i in range(1,10)]
# test_dirs = ['Pair'+str(i) for i in range(10,13)]

# train_files = []
# for d in train_dirs:
# 	files = glob.glob(f'data/parag_dataset_git/dataset_pairwise/{d}/Setting1/*/*.zip')
# 	files.sort()
# 	train_files += files
# 	files = glob.glob(f'data/parag_dataset_git/dataset_pairwise/{d}/Setting2/*/*.zip')
# 	files.sort()
# 	train_files += files

# for f in train_files:
# 	with zipfile.ZipFile(f, 'r') as zip_ref:
# 		zip_ref.extractall('data/parag_dataset/')

# test_files = []
# for d in test_dirs:
# 	files = glob.glob(f'data/parag_dataset_git/dataset_pairwise/{d}/Setting1/*/*.zip')
# 	files.sort()
# 	test_files += files
# 	files = glob.glob(f'data/parag_dataset_git/dataset_pairwise/{d}/Setting2/*/*.zip')
# 	files.sort()
# 	test_files += files

# for f in test_files:
# 	with zipfile.ZipFile(f, 'r') as zip_ref:
# 		zip_ref.extractall('data/parag_dataset/')

train_dirs = ['*P'+str(i)+'*' for i in range(1,10)]
test_dirs = ['*P'+str(i)+'*' for i in range(10,13)]

joints = ['hip', 'LShoulder', 'RShoulder', 'RFArm', 'RUArm', 'RHand']

train_scenes = []
for d in train_dirs:
	files = glob.glob(f'data/parag_dataset/{d}/*')
	files.sort()
	train_scenes += files

test_scenes = []
for d in test_dirs:
	files = glob.glob(f'data/parag_dataset/{d}/*')
	files.sort()
	test_scenes += files

train_trajs = []
test_trajs = []
def preproc(scenes):
	giver_trajs = []
	taker_trajs = []
	robot_trajs = []
	labels = []
	for f in scenes:
		print(f)
		giver_traj = []
		taker_traj = []
		for j in joints:
			reader = csv.DictReader(open(os.path.join(f,f'giver_{j}_pose_saved.csv')))
			giver_traj.append([])
			for row in reader:
				giver_traj[-1].append([float(row['x']), float(row['y']), float(row['z'])])
			
			reader = csv.DictReader(open(os.path.join(f,f'taker_{j}_pose_saved.csv')))
			taker_traj.append([])
			for row in reader:
				taker_traj[-1].append([float(row['x']), float(row['y']), float(row['z'])])

		giver_traj = np.array(giver_traj).transpose((1,0,2))[::4]
		taker_traj = np.array(taker_traj).transpose((1,0,2))[::4]
		R_taker = rotation_normalization(taker_traj[0])
		o_taker = taker_traj[0, 2].copy()
		R_giver = rotation_normalization(taker_traj[0])
		o_giver = giver_traj[0, 2].copy()
		robot_traj = []
		T = giver_traj.shape[0]
		for t in range(T):
			robot_traj.append(joint_angle_extraction(R_giver.dot((giver_traj[t] - o_giver).T).T))

			giver_traj[t] = R_taker.dot((giver_traj[t] - o_taker).T).T
			taker_traj[t] = R_taker.dot((taker_traj[t] - o_taker).T).T
		robot_traj = np.array(robot_traj)

		giver_force_data = []
		reader = csv.DictReader(open(os.path.join(f,'Wrench_giver_saved.csv')))
		for row in reader:
			giver_force_data.append([float(row['Fx']), float(row['Fy']), float(row['Fz'])])

		giver_force_data = scipy.ndimage.uniform_filter1d(np.linalg.norm(np.array(giver_force_data), axis=-1),20)

		taker_force_data = []
		reader = csv.DictReader(open(os.path.join(f,'Wrench_taker_saved.csv')))
		for row in reader:
			taker_force_data.append([float(row['Fx']), float(row['Fy']), float(row['Fz'])])
		taker_force_data = scipy.ndimage.uniform_filter1d(np.linalg.norm(np.array(taker_force_data), axis=-1),20)
		
		force_threshold = 1
		start_idx = 300 + np.where(taker_force_data[300:400]>force_threshold)[0][0]
		end_idx = 400+np.where(giver_force_data[400:500]>force_threshold)[0][-1]
		
		label = np.zeros(T)
		label[start_idx:end_idx] = 1
		label[end_idx:] = 2

		giver_trajs.append(giver_traj)
		taker_trajs.append(taker_traj)
		robot_trajs.append(robot_traj)
		labels.append(label)
	giver_trajs = np.array(giver_trajs)
	taker_trajs = np.array(taker_trajs)
	robot_trajs = np.array(robot_trajs)
	labels = np.array(labels)
	print(giver_trajs.shape, taker_trajs.shape, robot_trajs.shape, labels.shape)
	return giver_trajs, taker_trajs, robot_trajs, labels

np.savez_compressed('data/parag_dataset.npz', train_data = preproc(train_scenes), test_data = preproc(test_scenes), joints=joints)