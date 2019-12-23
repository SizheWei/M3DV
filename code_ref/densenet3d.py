import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


PARAMS = {
    "input_channels": 1,
    "output_channels": 2,
    "input_transform_fn": lambda x: x / 128. - 1.,
    "input_conv_channels": 32,
    'down_structure': [3, 8, 4],
    'activation_fn': lambda: nn.LeakyReLU(0.1, inplace=True),
    # "normalization_fn": lambda c: nn.GroupNorm(num_groups=4, num_channels=c),
    "normalization_fn": lambda c: nn.BatchNorm3d(num_features=c),
    "drop_rate": 0,
    'growth_rate': 32,
    'bottleneck': 4,
    'compression': 2,
    'use_memonger': True  # ~2.2x memory efficiency (batch_size), 25%~30% slower on GTX 1080.
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


class DenseSharp(nn.Module):

    def __init__(self):

        super(DenseSharp, self).__init__()

        input_channels = PARAMS['input_channels']
        input_transform_fn = PARAMS['input_transform_fn']
        input_conv_channels = PARAMS['input_conv_channels']
        normalization_fn = PARAMS['normalization_fn']
        activation_fn = PARAMS['activation_fn']
        down_structure = PARAMS['down_structure']
        output_channels = PARAMS['output_channels']

        self.input_features = nn.Sequential()
        if input_transform_fn is not None:
            self.input_features.add_module("input_transform", Lambda(input_transform_fn))
        self.input_features.add_module("init_conv", nn.Conv3d(input_channels, input_conv_channels, kernel_size=3,
                                                              stride=1, padding=1, bias=True))
        self.input_features.add_module("init_norm", normalization_fn(input_conv_channels))
        self.input_features.add_module("init_act", activation_fn())

        self.dense_blocks = nn.ModuleList()
        self.transmit_blocks = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()

        channels = input_conv_channels
        for i, num_layers in enumerate(down_structure):
            dense_block = nn.Sequential()
            for j in range(num_layers):
                conv_layer = ConvBlock(channels)
                dense_block.add_module('denseblock{}_layer{}'.format(i + 1, j + 1), conv_layer)
                channels = conv_layer.out_channels
            self.dense_blocks.append(dense_block)

            if i == 0:
                segmentation_channels = channels
            else:
                # TODO: a pre-activated top-down path
                self.top_down_blocks.append(nn.Sequential(normalization_fn(channels), activation_fn(),
                                                          nn.ConvTranspose3d(channels, up_channels,
                                                                             kernel_size=2, stride=2)))

            trans_layer = TransmitBlock(channels, is_last_block=i == len(down_structure) - 1)
            self.transmit_blocks.append(trans_layer)
            up_channels = channels
            channels = trans_layer.out_channels

            self.classification_head = nn.Linear(channels, output_channels)
            self.segmentation_head = nn.Conv3d(segmentation_channels, 1, kernel_size=1, stride=1)

    def forward(self, x, **return_opts):
        batch_size, _, d, h, w = x.size()
        conv = self.input_features(x)

        # bottom-up for classification
        bottom_up_feats = []
        for dense_block, transmit_block in zip(self.dense_blocks, self.transmit_blocks):
            bottom_up = dense_block(conv)
            bottom_up_feats.append(bottom_up)
            conv = transmit_block(bottom_up)

        # top-down for segmentation
        top_down = bottom_up_feats[-1]
        for bottom_up_feat, top_down_block in zip(reversed(bottom_up_feats[:-1]), reversed(self.top_down_blocks)):
            deconv = top_down_block(top_down)
            top_down = bottom_up_feat + deconv
        logit = self.segmentation_head(top_down)
        seg = torch.sigmoid(logit)

        pooled = F.adaptive_avg_pool3d(conv, 1).view(batch_size, -1)
        clf = self.classification_head(pooled)

        ret = dict(clf=clf, seg=seg)

        # return other features in one forward
        for opt in return_opts:
            assert opt in {"return_features", "return_cam"}
        # print(return_opts)

        if 'return_features' in return_opts and return_opts['return_features']:
            ret['features'] = conv

        if 'return_cam' in return_opts and return_opts['return_cam']:
            weight = self.classification_head.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            bias = self.classification_head.bias
            cam_raw = F.conv3d(conv, weight, bias)
            cam = F.interpolate(cam_raw, size=(d, h, w), mode='trilinear', align_corners=True)
            ret['cam'] = F.softmax(cam, dim=1)
            ret['cam_raw'] = F.softmax(cam_raw, dim=1)

        return ret


if __name__ == '__main__':

    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    from tqdm import trange

    # densenet
    # batch_size = 8
    # model = densenet3d(use_memonger=False).cuda()
    # batch_size = 20
    # model = densenet3d(use_memonger=True).cuda()

    # densesharp
    # batch_size = 6
    # model = densenet3d(with_segment=True, use_memonger=False).cuda()
    batch_size = 1
    model = densenet3d(with_segment=False, use_memonger=True).cuda()

    # torch.save(model.state_dict(), 'tmp.pth')

    voxel = torch.randn(batch_size, 1, 32, 32, 32).cuda()

    output = model(voxel)
    print(output['clf'].shape)
    # if segmentation:
    #     print(segmentation[0].shape)

    # with torch.no_grad():
    #     ret = model(voxel, return_features=True)
    #     for k, v in ret.items():
    #         print(k, v.shape)
    #
    #     ret = model(voxel, return_features=False, return_cam=False)
    #
    #     for k, v in ret.items():
    #         print(k, v.shape)
    #
    #     ret = model(voxel, return_cam=True)
    #     for k, v in ret.items():
    #         print(k, v.shape)
        # print(ret['cam_raw'].sum(dim=1))
        # print(ret['cam'].sum(dim=1))

    # exit()

    total_size = 1000
    for _ in trange(total_size // batch_size + 1):
        o = model(voxel)
        (o[0].mean()+o[1].mean()).backward()

    #
    # output, features = model(input)
    # print(output.size())
    # print(features.size())
