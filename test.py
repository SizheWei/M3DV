import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import roc_auc_score
import tensorboard_logger as tb_logger
print(torch.cuda.is_available())

PARAMS = {
    "input_channels": 1,
    "output_channels": 2,
    "input_transform_fn": lambda x: x / 128. - 1.,
    "input_conv_channels": 32,
    'down_structure': [2,2,2],
#     'down_structure': [3, 8, 4],
    'activation_fn': lambda: nn.LeakyReLU(0.1, inplace=True),
    # "normalization_fn": lambda c: nn.GroupNorm(num_groups=4, num_channels=c),
    "normalization_fn": lambda c: nn.BatchNorm3d(num_features=c),
    "drop_rate": 0,
    'growth_rate': 32,
    'bottleneck': 4,
    'compression': 2,
    'use_memonger': False  # ~2.2x memory efficiency (batch_size), 25%~30% slower on GTX 1080.
}

def densenet3d(with_segment=False, snapshot=None, **kwargs):
    for k, v in kwargs.items():
        assert k in PARAMS
        PARAMS[k] = v
    print("Model hyper-parameters:", PARAMS)
    if with_segment:
        model = DenseSharp()
        print("Using DenseSharp model.")
    else:
        model = DenseNet()
        print("Using DenseNet model.")
    print(model)
    if snapshot is None:
        initialize(model.modules())
        print("Random initialized.")
    else:
        state_dict = torch.load(snapshot)
        model.load_state_dict(state_dict)
        print("Load weights from `%s`," % snapshot)
    return model

def initialize(modules):
    for m in modules:
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            m.bias.data.zero_()

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels):
        super(ConvBlock, self).__init__()

        growth_rate = PARAMS['growth_rate']
        bottleneck = PARAMS['bottleneck']
        activation_fn = PARAMS['activation_fn']
        normalization_fn = PARAMS['normalization_fn']

        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.use_memonger = PARAMS['use_memonger']
        self.drop_rate = PARAMS['drop_rate']

        # TODO: consider bias term in conv with GN
        self.add_module('norm_1', normalization_fn(in_channels))
        self.add_module('act_1', activation_fn())
        self.add_module('conv_1', nn.Conv3d(in_channels, bottleneck * growth_rate, kernel_size=1, stride=1,
                                            padding=0, bias=True))

        self.add_module('norm_2', normalization_fn(bottleneck * growth_rate))
        self.add_module('act_2', activation_fn())
        self.add_module('conv_2', nn.Conv3d(bottleneck * growth_rate, growth_rate, kernel_size=3, stride=1,
                                            padding=1, bias=True))

    def forward(self, x):
        super_forward = super(ConvBlock, self).forward
        if self.use_memonger:
            new_features = checkpoint(super_forward, x)
        else:
            new_features = super_forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

    @property
    def out_channels(self):
        return self.in_channels + self.growth_rate

class TransmitBlock(nn.Sequential):
    def __init__(self, in_channels, is_last_block):
        super(TransmitBlock, self).__init__()

        activation_fn = PARAMS['activation_fn']
        normalization_fn = PARAMS['normalization_fn']
        compression = PARAMS['compression']

        # print("in_channels: %s, compression: %s" % (in_channels, compression))
        assert in_channels % compression == 0

        self.in_channels = in_channels
        self.compression = compression

        self.add_module('norm', normalization_fn(in_channels))
        self.add_module('act', activation_fn())
        if not is_last_block:
            self.add_module('conv', nn.Conv3d(in_channels, in_channels // compression, kernel_size=(1, 1, 1),
                                              stride=1, padding=0, bias=True))
            self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2, padding=0))
        else:
            self.compression = 1

    @property
    def out_channels(self):
        return self.in_channels // self.compression

class Lambda(nn.Module):
    def __init__(self, lambda_fn):
        super(Lambda, self).__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        return self.lambda_fn(x)

class DenseNet(nn.Module):

    def __init__(self):

        super(DenseNet, self).__init__()

        input_channels = PARAMS['input_channels']
        input_transform_fn = PARAMS['input_transform_fn']
        input_conv_channels = PARAMS['input_conv_channels']
        normalization_fn = PARAMS['normalization_fn']
        activation_fn = PARAMS['activation_fn']
        down_structure = PARAMS['down_structure']
        output_channels = PARAMS['output_channels']

        self.features = nn.Sequential()
        if input_transform_fn is not None:
            self.features.add_module("input_transform", Lambda(input_transform_fn))
        self.features.add_module("init_conv", nn.Conv3d(input_channels, input_conv_channels, kernel_size=3,
                                                        stride=1, padding=1, bias=True))
        self.features.add_module("init_norm", normalization_fn(input_conv_channels))
        self.features.add_module("init_act", activation_fn())

        channels = input_conv_channels
        for i, num_layers in enumerate(down_structure):
            for j in range(num_layers):
                conv_layer = ConvBlock(channels)
                self.features.add_module('denseblock{}_layer{}'.format(i + 1, j + 1), conv_layer)
                channels = conv_layer.out_channels
                # print(i, j, channels)

            trans_layer = TransmitBlock(channels, is_last_block=i == len(down_structure) - 1)
            self.features.add_module('transition%d' % (i + 1), trans_layer)
            channels = trans_layer.out_channels

        self.classifier = nn.Linear(channels, output_channels)

    def forward(self, x, **return_opts):
        # o = x
        # for i, layer in enumerate(self.features):
        #     o = layer(o)
        #     print(i, layer, o.shape)

        batch_size, _, d, h, w = x.size()

        features = self.features(x)
        pooled = F.adaptive_avg_pool3d(features, 1).view(batch_size, -1)
        scores = self.classifier(pooled)

        if len(return_opts) == 0:
            return scores

        # return other features in one forward
        for opt in return_opts:
            assert opt in {"return_features", "return_cam"}
        # print(return_opts)

        ret = dict(scores=scores)

        if 'return_features' in return_opts and return_opts['return_features']:
            ret['features'] = features

        if 'return_cam' in return_opts and return_opts['return_cam']:
            weight = self.classifier.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            bias = self.classifier.bias
            cam_raw = F.conv3d(features, weight, bias)
            cam = F.interpolate(cam_raw, size=(d, h, w), mode='trilinear', align_corners=True)
            ret['cam'] = F.softmax(cam, dim=1)
            ret['cam_raw'] = F.softmax(cam_raw, dim=1)
        return ret

from torch.utils.data import Dataset
import random
import torch
import os
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np

class ClfDataset(Dataset):

    def __init__(self, train=True):
        self.train = train
        data_dir = './dataset/'
        # choose the dataset
        patients_train = os.listdir(data_dir+'train_val/')
        patients_train.sort(key= lambda x:int(x[9:-4]))
        patients_test = os.listdir(data_dir+'test/')
        patients_test.sort(key= lambda x:int(x[9:-4]))
        labels_df = pd.read_csv('./dataset/info.csv',index_col=0)

        self.data_train = []
        self.data_test = []
        self.labels = []
        self.names_train = []
        self.names_test = []

        for num_train, patient_train in enumerate(patients_train):
            patient_train_name = patient_train[0:-4]
            self.names_train.append(patient_train_name)
            label = labels_df.get_value(patient_train_name, 'lable')

            path_train = data_dir + 'train_val/' + patient_train

            img_data_train = np.load(path_train)
            voxel_train = img_data_train['voxel'].astype(np.int32)
#             voxel_train_crop = voxel_train[10:90,10:90,10:90]
            voxel_train_crop = voxel_train[20:80,20:80,20:80]
#             voxel_train_crop = voxel_train[25:75,25:75,25:75]
            self.data_train.append(voxel_train_crop)
            self.labels.append(label)

        for num_test, patient_test in enumerate(patients_test):
            self.names_test.append(patient_test[0:-4])
            path_test = data_dir + 'test/' + patient_test
            img_data_test = np.load(path_test)
            voxel_test = img_data_test['voxel'].astype(np.int32)
#             voxel_test_crop = voxel_test[10:90,10:90,10:90]
            voxel_test_crop = voxel_test[20:80,20:80,20:80]
#             voxel_test_crop = voxel_test[25:75,25:75,25:75]
            self.data_test.append(voxel_test_crop)
    
    def __getitem__(self, item):
        if self.train:
            patient_data_train = self.data_train[item]
            patient_label = self.labels[item]
            patient_name_train = self.names_train[item]
            return patient_data_train, patient_label, patient_name_train
        else:
            patient_data_test = self.data_test[item]
            patient_name_test = self.names_test[item]
            return patient_data_test, patient_name_test
        
    def __len__(self):
        if self.train:
            return len(self.labels)
        else:
            return len(self.data_test)

# '''[begin] load the train and validation dataset:'''
# import numpy as np
# # use np.load to import data, devide dataset into 2 parts: train_data & validation_data:
# # credit: cheez & Matthew Kerian
# # link: https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa/56243777
# '''use older to successfully load the data:'''
# # save np.load
# np_load_old = np.load

# # modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# data_ineed = np.load('muchdata.npy')

# # restore np.load for future normal usage
# np.load = np_load_old

# # If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
# train_data = data_ineed[:-60]
# validation_data = data_ineed[-60:]
# '''[end] load the train and validation dataset:'''

model = densenet3d(with_segment=False, use_memonger=True).to(device)
# model = nn.DataParallel(model, device_ids=[0, 1])
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9)

criterion = criterion.to(device)

from torch.utils.data import DataLoader

data_train = ClfDataset(train=True)
train_data_loader = DataLoader(dataset=data_train, batch_size=32, shuffle=True)
data_test = ClfDataset(train=False)
test_data_loader = DataLoader(dataset=data_test, batch_size=32, shuffle=False)


date = 9 # day
number = 2 # No.
sample_size = 32

PATH = './PKG/my_{}_{}.pth'.format(date,number)
model = densenet3d(with_segment=False, use_memonger=True).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()
predict_value = []
names_box_test = []
    
for index, (inputs,patient_name) in enumerate(tqdm(test_data_loader)):
    
    inputs = inputs.to(device)
    # forward + backward + optimize
    inputs = inputs.unsqueeze(dim=1).float()
        
    inputs = F.interpolate(inputs, size=[sample_size,sample_size,sample_size],mode='trilinear',align_corners=False)
        
    outputs = model(inputs)
    
    test_value = F.softmax(outputs)
    test_value_1 = test_value[:,1]
    test_value_1 = test_value_1.cpu()
    predict_value.extend(test_value_1.detach().numpy().tolist())

    names_box_test.extend(patient_name)
        
import csv
box_train = zip(names_box_test, predict_value)
title = ('Id', 'Predicted')
    # print(box_train)
with open('submission.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerow(title)
    for row in box_train:
        wr.writerow(row)
        
print('---------Congratulations! Please check file named: submission.csv---------')