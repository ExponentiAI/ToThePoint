import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
from random import shuffle
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class ModelNet40C(Dataset):
    def __init__(self, num_points, corruption, severity=None,partition='train'):
        super(ModelNet40C, self).__init__()

        assert corruption in [
            "background", "cutout", "density", "density_inc", "distortion",
            "distortion_rbf", "distortion_rbf_inv", "gaussian", "impluse", "lidar",
            "occlusion", "rotation", "shear", "uniform", "upsampling", # 15 corruptions
            "original",
        ]

        if corruption == "original":
            assert severity is None
            fname = "../modelnet40_c/data_original.npy"
        else:
            assert severity is not None
            fname = f"../modelnet40_c/data_{corruption}_{severity}_{partition}.npy"

        self.data = np.load(fname)
        self.label = np.load(f"../modelnet40_c/label_{partition}.npy")
        self.num_points = num_points

        print('self.data.shape[0]',self.data.shape[0])

    def __getitem__(self, item):
        pointcloud = self.data[item]
        if pointcloud.shape[1] >= self.num_points:
            pointcloud = pointcloud[:self.num_points]
        else:
            choice = np.random.choice(len(pointcloud), self.num_points, replace=True)
            pointcloud = pointcloud[choice, :]

        label = np.squeeze(self.label[item])

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    fname = f"../../modelnet40_c/data_lidar_1.npy"
    data = np.load(fname)
    label = np.load("../../modelnet40_c/label.npy")

    # trainDataLoader = torch.utils.data.DataLoader(ModelNet40C(1024,"lidar",5), batch_size=1, shuffle=True,
    #                                               num_workers=0, drop_last=True)
    # for p,l in trainDataLoader:
    #     print(p.shape,l.shape,l)

    index=[i for i in range(data.shape[0])]
    # print(index)
    shuffle(index)
    # print(index)
    #
    data = data[index,:,:]
    label=label[index,:]
    train_size = int(data.shape[0]*0.8)
    test_size = data.shape[0] - train_size
    train_data = data[:train_size,:,:]
    train_label = label[:train_size,:]
    test_data = data[train_size:, :, :]
    test_label = label[train_size:,]
    print(len(label))
    print(len(train_label))
    print(len(test_label))
    # np.save("../../modelnet40_c/data_lidar_1_train.npy", train_data)
    # np.save("../../modelnet40_c/label_train.npy", train_label)#../modelnet40_c/label
    # np.save("../../modelnet40_c/data_lidar_1_test.npy", test_data)
    # np.save("../../modelnet40_c/label_test.npy", test_label)


    a = np.arange(0,20)
    a = np.reshape(a,(10,2))
    print(a)
    index = [i for i in range(10)]
    shuffle(index)
    print(index)
    a = a[index,:]
    print(a)