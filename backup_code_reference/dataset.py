from torch.utils.data import Dataset
import random
import torch
import os
import pandas as pd

import numpy as np
# from environ import PATH

# from utils import rotation, reflection, crop, random_center
# from utils.misc import _triple
# from utils.util import segment2n_segment


class ClfDataset(Dataset):

    def __init__(self, train=True):

        data_dir = './dataset/'
        # choose the dataset
        patients_train = os.listdir(data_dir+'train_val/')
        patients_test = os.listdir(data_dir+'test/')
        labels_df = pd.read_csv('./dataset/info.csv',index_col=0)

        self.data_train = []
        self.data_test = []
        self.label = []

        for num, patient in enumerate(patients_train):
            patient_name = patient[0:-4]
            label = labels_df.get_value(patient_name, 'lable')
            path = data_dir + 'train_val/' + patient
            img_data = np.load(path)
            voxel = img_data['voxel'].astype(np.int32)
            self.data_train.append([voxel])
            self.label.append(label)

        for num, patient in enumerate(patients_test):
            path = data_dir + 'test/' + patient
            img_data = np.load(path)
            voxel = img_data['voxel'].astype(np.int32)
            self.data_test.append([voxel])

    def __getitem__(self, item):
        if train:
            patient_data = self.data_train[0][i]
            patient_label = self.data_train[i]
            return patient_data, patient_label
        else:
            patient_data = self.data_test[0][i]
            return patient_data

    def __len__(self):
        return len(self.index)


# class Transform:
#     def __init__(self, size, move=None):
#         self.size = _triple(size)
#         self.move = move

#     def __call__(self, arr, aux=None):
#         shape = arr.shape
#         if self.move is not None:
#             center = random_center(shape, self.move)
#             angle = np.random.randint(4, size=3)
#             axis = np.random.randint(4) - 1

#             arr_ret = crop(arr, center, self.size)
#             arr_ret = rotation(arr_ret, angle=angle)
#             arr_ret = reflection(arr_ret, axis=axis)
#             arr_ret = np.expand_dims(arr_ret, axis=-1)
#             if aux is not None:
#                 aux_ret = crop(aux, center, self.size)
#                 aux_ret = rotation(aux_ret, angle=angle)
#                 aux_ret = reflection(aux_ret, axis=axis)
#                 aux_ret = np.expand_dims(aux_ret, axis=-1)
#                 return arr_ret, aux_ret
#             return arr_ret
#         else:
#             center = np.array(shape) // 2
#             arr_ret = crop(arr, center, self.size)
#             arr_ret = np.expand_dims(arr_ret, axis=-1)
#             if aux is not None:
#                 aux_ret = crop(aux, center, self.size)
#                 aux_ret = np.expand_dims(aux_ret, axis=-1)
#                 return arr_ret, aux_ret
#             return arr_ret


# def shuffle_repeat(lst):
#     # iterator should have limited size
#     total_size = len(lst)
#     i = 0

#     random.shuffle(lst)
#     while True:
#         yield lst[i]
#         i += 1
#         if i >= total_size:
#             i = 0
#             random.shuffle(lst)


# if __name__ == '__main__':
#     dataset = ClfDataset(crop_size=32, move=5, train=False, subset=[1, 2, 3, 4, 5], lidc=True, voxel_segment=True,
#                          output_segment=True)
#     x = dataset[0]
#     voxel = torch.Tensor(x[0]).unsqueeze(dim=0)
#     batch_segment = torch.Tensor(x[2]).unsqueeze(dim=0)
#     n_segment = segment2n_segment(batch_segment, n_sat=1024)
#     voxel_feature = voxel[n_segment]
#     print('Data shape is {}.'.format(x[0].shape))
#     print('label is {}.'.format(x[1]))
#     print('Data segment shape is {}.'.format(x[2].shape))
