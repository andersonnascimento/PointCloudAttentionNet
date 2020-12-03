import os
import sys
import glob
# import math
import h5py
import numpy as np

from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

class ModelNet40(Dataset):

    def jitter_pointcloud(self, pointcloud, sigma=0.01, clip=0.02):
        N, C = pointcloud.shape
        pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
        return pointcloud

    def translate_pointcloud(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud

    def download(self):
        os.system('mkdir -p /tmp/dataset/data')
        BASE_DIR = os.path.dirname('/tmp/dataset/')
        # DATA_DIR = os.path.join(BASE_DIR, 'data')
        DATA_DIR = os.path.dirname('/tmp/dataset/data')
        # if not os.path.exists(DATA_DIR):
        #     os.mkdir(DATA_DIR)
        if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
            www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
            zipfile = os.path.basename(www)
            os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
            os.system('rm %s' % (zipfile))

    def load_data(self, partition):
        self.download()
        os.system('mkdir -p /tmp/dataset/data')
        # BASE_DIR = os.path.dirname('/tmp/dataset/')
        # DATA_DIR = os.path.join(BASE_DIR, 'data')
        DATA_DIR = os.path.dirname('/tmp/dataset/data')
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_*.h5')):
            f = h5py.File(h5_name, 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)

        label_description = np.loadtxt(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), dtype=np.str)
        train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size=0.30, random_state=self.random_state)
        if partition == 'train':
            return train_data, train_label, label_description
        else:
            validation_data, test_data, validation_label, test_label = train_test_split(test_data, test_label, test_size=0.50, random_state=self.random_state)
            if partition == 'validation':
                return validation_data, validation_label, label_description
            else:
                return test_data, test_label, label_description

    def __init__(self, num_points, partition='train', random_state=42):
        self.num_points = num_points
        self.partition = partition
        self.random_state = random_state
        self.data, self.label, self.label_description = self.load_data(partition)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = self.translate_pointcloud(pointcloud)
            pointcloud = self.jitter_pointcloud(pointcloud)

        return pointcloud, label

    def label_description(self):
        self.label_description

    def __len__(self):
        return self.data.shape[0]
