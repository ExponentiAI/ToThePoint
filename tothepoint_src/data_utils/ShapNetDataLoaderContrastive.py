import data_augment_utils as d_aug
import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import math
from data_utils.plyfile import load_ply
from PIL import Image
import torchvision.transforms as transforms


trans_1 = transforms.Compose(
    [
        d_aug.PointcloudToTensor(),
        d_aug.PointcloudNormalize(),
        d_aug.PointcloudScale(lo=0.5, hi=2, p=1),
        d_aug.PointcloudRotate(),
        d_aug.PointcloudTranslate(0.5, p=1),
        d_aug.PointcloudJitter(p=1),
        d_aug.PointcloudRandomInputDropout(p=1),
    ])

trans_2 = transforms.Compose(
    [
        d_aug.PointcloudToTensor(),
        d_aug.PointcloudNormalize(),
        d_aug.PointcloudScale(lo=0.5, hi=2, p=1),
        d_aug.PointcloudRotate(),
        d_aug.PointcloudTranslate(0.5, p=1),
        d_aug.PointcloudJitter(p=1),
        d_aug.PointcloudRandomInputDropout(p=1),
    ])


def load_shapenet_data():
    BASE_DIR = '../'
    all_filepath = []

    for cls in glob.glob(os.path.join(BASE_DIR, 'ShapeNet/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs

    return all_filepath

class ShapeNetConrastive(Dataset):
    def __init__(self,):
        self.data = load_shapenet_data()

    def __getitem__(self, item):
        # render_img_list.append(render_img)
        pointcloud = load_ply(self.data[item])  # 2048, 3
        np.random.shuffle(pointcloud)
        point_t1 = trans_1(pointcloud)
        point_t2 = trans_2(pointcloud)
        pointcloud = (point_t1, point_t2)
        return pointcloud

    def __len__(self):
        return len(self.data)

def load_shapenet_less_data():
    BASE_DIR = '../'
    all_filepath = []

    for cls in glob.glob(os.path.join(BASE_DIR, 'ShapeNetLess/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs

    return all_filepath

class ShapeNetConrastiveLess(Dataset):
    def __init__(self,):
        self.data = load_shapenet_less_data()

    def __getitem__(self, item):
        # render_img_list.append(render_img)
        pointcloud = load_ply(self.data[item])  # 2048, 3
        np.random.shuffle(pointcloud)
        point_t1 = trans_1(pointcloud)
        # print('point_t1',point_t1.shape)
        point_t2 = trans_2(pointcloud)
        pointcloud = (point_t1, point_t2)
        return pointcloud

    def __len__(self):
        return len(self.data)
def load_modelnet_data(partition):
    BASE_DIR = '../'
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_ScanObjectNN(partition):
    BASE_DIR = '../ScanObjectNN'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')

    return data, label

class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNNSVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train_dataset = ShapeNetConrastive()
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                                  num_workers=0, drop_last=True)
    for (i,j) in trainDataLoader:
        print(i.shape,j.shape)
