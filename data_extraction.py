import numpy as np
import glob
import pandas as pd
import os
import argparse
import csv

joints = ['Hip', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 'LFArm', 'LHand', 'RShoulder', 'RUArm', 'RFArm', 'RHand']
object_keys = ['Rigid Body:object:Position:X', 'Rigid Body:object:Position:Y', 'Rigid Body:object:Position:Z']
label_idx = {'reach':0, 'transfer':1, 'retreat':2}

def preproc_files_list(files):
    p1_trajs = []
    p2_trajs = []
    object_trajs = []
    labels = []
    for f_csv in files:
        p1_name = os.path.basename(f_csv).split('_')[0]
        p2_name = os.path.basename(f_csv).split('_')[1]
        p1_body_keys = [f'Bone:{p1_name}:{joint}:Position:{c}' for c in ['X', 'Y', 'Z'] for joint in joints]
        p2_body_keys = [f'Bone:{p2_name}:{joint}:Position:{c}' for c in ['X', 'Y', 'Z'] for joint in joints]
        p1_traj = []
        p2_traj = []
        object_traj = []
        label = []
        reader = csv.DictReader(open(f_csv))
        for row in reader:
            p1_traj.append([])
            p2_traj.append([])
            object_traj.append([])
            for k in p1_body_keys:
                p1_traj[-1].append(row[k])
            for k in p2_body_keys:
                p2_traj[-1].append(row[k])
            for k in object_keys:
                object_traj[-1].append(row[k])
            label.append(label_idx[row['Giver']])
        p1_trajs.append(np.array(p1_traj))
        p2_trajs.append(np.array(p2_traj))
        object_trajs.append(np.array(object_traj))
        labels.append(np.array(label))
    p1_trajs = np.array(p1_trajs, dtype=object)
    p2_trajs = np.array(p2_trajs, dtype=object)
    object_trajs = np.array(object_trajs, dtype=object)
    labels = np.array(labels, dtype=object)
    return p1_trajs, p2_trajs, object_trajs, labels


files = glob.glob('data/Bimanual Handovers Dataset/*/OptiTrack_Global_Frame/*.csv')
p1_trajs, p2_trajs, object_trajs, labels = preproc_files_list(files)
np.savez_compressed('data/data_raw.npz', p1_trajs=p1_trajs, p2_trajs=p2_trajs, object_trajs=object_trajs, labels=labels, joints=joints)