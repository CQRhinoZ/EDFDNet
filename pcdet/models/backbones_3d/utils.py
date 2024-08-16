import torch
from torch import nn
from ...utils.spconv_utils import replace_feature, spconv


class WeightFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat1, feat2):
        features1 = feat1.features * self.weight1
        features2 = feat2.features * self.weight2
        # indices = torch.cat([feat1.indices, feat2.indices])
        feat2 = feat2.replace_feature(torch.cat([features1, features2]))
        feat2.indices = torch.cat([feat1.indices, feat2.indices])

        indices_cat = feat2.indices[:, [0, 2, 3]]
        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)

        features_cat = feat2.features
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        spatial_shape = feat2.spatial_shape[1:]

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=feat2.batch_size
        )
        return x_out

# 大核分组卷积
class SpatialGroupConv(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False):
        super(SpatialGroupConv, self).__init__()
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int(kernel_size // 2),
            bias=bias,
            indice_key=indice_key,
        )

        self.conv3x3_1 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=int(kernel_size // 2) - 1,
            bias=bias,
            dilation=int(kernel_size // 2) - 1,
            indice_key=indice_key + 'conv_3x3_1',
        )
        self._indice_list = []

        if kernel_size == 7:
            _list = [0, 3, 4, 7]
        elif kernel_size == 5:
            _list = [0, 2, 3, 5]
        else:
            raise ValueError('Unknown kernel size %d' % kernel_size)
        for i in range(len(_list) - 1):
            for j in range(len(_list) - 1):
                for k in range(len(_list) - 1):
                    a = torch.zeros((kernel_size, kernel_size, kernel_size)).long()
                    a[_list[i]:_list[i + 1], _list[j]:_list[j + 1], _list[k]:_list[k + 1]] = 1
                    b = torch.range(0, kernel_size ** 3 - 1, 1)[a.reshape(-1).bool()]
                    self._indice_list.append(b.long())

    # 此处可能会因为稀疏卷积版本的不同，而有不同的转置顺序
    def _convert_weight(self, weight):
        weight_reshape = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels,
                                                                          -1).clone()
        weight_return = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels,
                                                                         -1).clone()
        for _indice in self._indice_list:
            _mean_weight = torch.mean(weight_reshape[:, :, _indice], dim=-1, keepdim=True)
            weight_return[:, :, _indice] = _mean_weight
        return weight_return.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size,
                                     self.kernel_size).permute(2, 3, 4, 0, 1)

    def forward(self, x_conv):
        if self.training:
            self.block.weight.data = self._convert_weight(self.block.weight.data)
        x_conv_block = self.block(x_conv)
        x_conv_conv3x3_1 = self.conv3x3_1(x_conv)
        x_conv_block = x_conv_block.replace_feature(x_conv_block.features + x_conv_conv3x3_1.features)
        return x_conv_block
